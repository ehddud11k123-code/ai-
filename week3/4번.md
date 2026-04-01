import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QSlider, QLabel, QPushButton, QTextBrowser, QFrame)
from PySide6.QtCore import Qt, QTimer

class PureNumpyMLP:
    """순수 NumPy로 구현한 2-4-1 다층 퍼셉트론 (역전파 포함)"""
    def __init__(self, lr=0.5):
        self.lr = lr
        self.init_weights()
        
    def init_weights(self):
        np.random.seed() # 매 리셋마다 다른 시작점
        # Input(2) -> Hidden(4)
        self.W1 = np.random.uniform(-1, 1, (2, 4))
        self.b1 = np.zeros((1, 4))
        # Hidden(4) -> Output(1)
        self.W2 = np.random.uniform(-1, 1, (4, 1))
        self.b2 = np.zeros((1, 1))

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x) # 여기서 x는 이미 활성화 함수를 통과한 A값(출력값)이라고 가정

    def forward(self, X):
        self.Z1 = np.dot(X, self.W1) + self.b1
        self.A1 = self.sigmoid(self.Z1)
        
        self.Z2 = np.dot(self.A1, self.W2) + self.b2
        self.A2 = self.sigmoid(self.Z2)
        return self.A2, self.A1

    def backward(self, X, Y, output):
        m = X.shape[0] # 배치 사이즈 (4)
        
        # 1. Output Layer Gradients
        # Loss = 1/2 * (output - Y)^2 
        # dLoss/dOutput = output - Y
        dZ2 = (output - Y) * self.sigmoid_derivative(output)
        dW2 = np.dot(self.A1.T, dZ2) / m
        db2 = np.sum(dZ2, axis=0, keepdims=True) / m

        # 2. Hidden Layer Gradients (Chain Rule)
        dA1 = np.dot(dZ2, self.W2.T)
        dZ1 = dA1 * self.sigmoid_derivative(self.A1)
        dW1 = np.dot(X.T, dZ1) / m
        db1 = np.sum(dZ1, axis=0, keepdims=True) / m

        # 3. Update Weights
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1
        
        # MSE Loss 반환
        return np.mean(0.5 * (Y - output)**2)

class VisualizerCanvas(FigureCanvas):
    def __init__(self):
        self.fig, self.axs = plt.subplots(1, 3, figsize=(15, 5), dpi=100)
        self.fig.tight_layout(pad=3.0)
        super().__init__(self.fig)

class MLPBackpropApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Lab 4: Multi-Layer Perceptron (XOR Solver)")
        self.resize(1500, 600)

        # XOR 데이터셋
        self.X = np.array([[0,0], [0,1], [1,0], [1,1]])
        self.Y = np.array([[0], [1], [1], [0]])
        
        self.model = PureNumpyMLP(lr=0.5)
        self.epoch = 0
        self.loss_history = []
        
        # 애니메이션을 위한 타이머 설정
        self.timer = QTimer()
        self.timer.timeout.connect(self.train_step)
        self.is_training = False

        self.init_ui()
        self.update_plots() # 초기 상태 그리기

    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QHBoxLayout(central_widget)

        # --- Left Panel: 3분할 그래프 ---
        self.canvas = VisualizerCanvas()
        layout.addWidget(self.canvas, 7)

        # --- Right Panel: 컨트롤 & 텍스트 설명 ---
        right_panel = QVBoxLayout()
        
        # 1. 제어부
        control_frame = QFrame()
        control_frame.setStyleSheet("background-color: #F8F9F9; border-radius: 5px; padding: 10px;")
        control_layout = QVBoxLayout(control_frame)
        
        title = QLabel("⚙️ Training Controls")
        title.setStyleSheet("font-size: 16px; font-weight: bold; color: #2C3E50;")
        control_layout.addWidget(title)

        # 상태 표시기
        self.status_lbl = QLabel("Epoch: 0 | Loss: 0.0000")
        self.status_lbl.setStyleSheet("font-size: 14px; font-weight: bold; color: #E74C3C;")
        control_layout.addWidget(self.status_lbl)

        # Learning Rate 슬라이더
        self.lbl_lr = QLabel(f"Learning Rate: {self.model.lr:.2f}")
        self.slider_lr = QSlider(Qt.Horizontal)
        self.slider_lr.setRange(1, 200) # 0.01 ~ 2.00
        self.slider_lr.setValue(int(self.model.lr * 100))
        self.slider_lr.valueChanged.connect(self.on_lr_change)
        control_layout.addWidget(self.lbl_lr)
        control_layout.addWidget(self.slider_lr)

        # 버튼 그룹
        btn_layout = QHBoxLayout()
        self.btn_train = QPushButton("▶ Start Training")
        self.btn_train.setStyleSheet("background-color: #2ECC71; color: white; font-weight: bold; height: 35px;")
        self.btn_train.clicked.connect(self.toggle_training)
        
        self.btn_reset = QPushButton("🔄 Reset Model")
        self.btn_reset.setStyleSheet("background-color: #95A5A6; color: white; font-weight: bold; height: 35px;")
        self.btn_reset.clicked.connect(self.reset_model)
        
        btn_layout.addWidget(self.btn_train)
        btn_layout.addWidget(self.btn_reset)
        control_layout.addLayout(btn_layout)

        right_panel.addWidget(control_frame)

        # 2. 텍스트 설명부
        self.text_browser = QTextBrowser()
        self.text_browser.setStyleSheet("background-color: #FFFFFF; font-size: 14px; padding: 10px; border: 1px solid #BDC3C7;")
        self.set_explanation_text()
        right_panel.addWidget(self.text_browser)

        layout.addLayout(right_panel, 3)

    def set_explanation_text(self):
        html = """
        <h2 style='color: #2980B9;'>💡 MLP와 Backpropagation</h2>
        <p>단층 퍼셉트론은 XOR(배타적 논리합) 문제를 선형적으로 분리할 수 없었습니다. 하지만 <b>은닉층(Hidden Layer)</b>을 추가한 다층 퍼셉트론(MLP)은 비선형 함수를 통해 공간을 구부려 이 문제를 해결합니다.</p>
        
        <h3>그래프 설명</h3>
        <ul>
            <li><b>좌 (Training Loss):</b> 모델이 정답을 찾아가며 오차(MSE)가 감소하는 곡선입니다.</li>
            <li><b>중 (Decision Boundary):</b> 2D 평면을 어떻게 분류하는지 보여줍니다. 선이 곡선으로 변하는 것을 확인하세요! (파란색 영역: 0, 빨간색 영역: 1)</li>
            <li><b>우 (Hidden Activations):</b> 4개의 입력 데이터(행)에 대해 은닉층의 4개 뉴런(열)이 반응하는 강도(히트맵)입니다. 각 뉴런이 다른 특징을 담당하고 있습니다.</li>
        </ul>
        """
        self.text_browser.setHtml(html)

    def on_lr_change(self):
        self.model.lr = self.slider_lr.value() / 100.0
        self.lbl_lr.setText(f"Learning Rate: {self.model.lr:.2f}")

    def toggle_training(self):
        if not self.is_training:
            self.is_training = True
            self.btn_train.setText("⏸ Pause Training")
            self.btn_train.setStyleSheet("background-color: #E67E22; color: white; font-weight: bold; height: 35px;")
            self.timer.start(50) # 50ms 마다 train_step 실행 (애니메이션 속도)
        else:
            self.is_training = False
            self.btn_train.setText("▶ Resume Training")
            self.btn_train.setStyleSheet("background-color: #2ECC71; color: white; font-weight: bold; height: 35px;")
            self.timer.stop()

    def reset_model(self):
        self.is_training = False
        self.timer.stop()
        self.btn_train.setText("▶ Start Training")
        self.btn_train.setStyleSheet("background-color: #2ECC71; color: white; font-weight: bold; height: 35px;")
        
        self.model.init_weights()
        self.epoch = 0
        self.loss_history = []
        self.update_plots()

    def train_step(self):
        """타이머에 의해 주기적으로 호출되어 N번 학습을 진행하고 UI 업데이트"""
        epochs_per_tick = 50 # 한 번 화면이 갱신될 때 50 에포크씩 진행 (너무 느리지 않게)
        current_loss = 0
        
        for _ in range(epochs_per_tick):
            output, _ = self.model.forward(self.X)
            loss = self.model.backward(self.X, self.Y, output)
            self.epoch += 1
            if self.epoch % 10 == 0: # 데이터 저장 주기 조절
                self.loss_history.append((self.epoch, loss))
            current_loss = loss

        self.status_lbl.setText(f"Epoch: {self.epoch} | Loss: {current_loss:.4f}")
        
        # 오차가 충분히 작아지면 자동 정지
        if current_loss < 0.005:
            self.toggle_training()
            self.status_lbl.setText(f"Epoch: {self.epoch} | Loss: {current_loss:.4f} (Converged!)")
            
        self.update_plots()

    def update_plots(self):
        ax_loss, ax_bound, ax_heat = self.axs.flatten()
        
        # [1] Training Loss
        ax_loss.clear()
        if self.loss_history:
            epochs, losses = zip(*self.loss_history)
            ax_loss.plot(epochs, losses, color='#2980B9', lw=2)
            # Y축을 로그 스케일로 설정하여 지수적 감소를 명확히 보여줌
            ax_loss.set_yscale('log')
        ax_loss.set_title("Training Loss (MSE)")
        ax_loss.set_xlabel("Epoch")
        ax_loss.set_ylabel("Loss")
        ax_loss.grid(True, linestyle='--', alpha=0.5)

        # [2] XOR Decision Boundary
        ax_bound.clear()
        x_min, x_max = -0.5, 1.5
        y_min, y_max = -0.5, 1.5
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.05), np.arange(y_min, y_max, 0.05))
        grid = np.c_[xx.ravel(), yy.ravel()]
        
        preds, _ = self.model.forward(grid)
        preds = preds.reshape(xx.shape)
        
        # 과제 예시와 유사한 파란-빨강(RdYlBu) 컬러맵 사용
        contour = ax_bound.contourf(xx, yy, preds, alpha=0.8, cmap='RdYlBu_r', levels=20)
        
        # 실제 데이터 포인트 (XOR)
        for i in range(len(self.Y)):
            marker = 'x' if self.Y[i][0] == 1 else 'o'
            color = 'blue' if self.Y[i][0] == 1 else 'red'
            ax_bound.scatter(self.X[i,0], self.X[i,1], c=color, marker=marker, s=100, edgecolors='black', linewidths=2)
            
        ax_bound.set_title("XOR Decision Boundary")
        ax_bound.set_xlabel("x1")
        ax_bound.set_ylabel("x2")

        # [3] Hidden Layer Activations (Heatmap)
        ax_heat.clear()
        _, a1_out = self.model.forward(self.X)
        
        # a1_out의 모양은 (4 데이터, 4 은닉노드)
        im = ax_heat.imshow(a1_out, cmap='viridis', aspect='auto', vmin=0, vmax=1)
        ax_heat.set_title("Hidden Layer Activations")
        ax_heat.set_xlabel("Hidden Neuron")
        ax_heat.set_ylabel("Input Pattern (00, 01, 10, 11)")
        ax_heat.set_xticks(np.arange(4))
        ax_heat.set_yticks(np.arange(4))
        ax_heat.set_xticklabels(['h1', 'h2', 'h3', 'h4'])
        ax_heat.set_yticklabels(['0,0', '0,1', '1,0', '1,1'])
        
        # 우측에 컬러바 추가 (한 번만 추가되도록 처리)
        if not hasattr(self, 'colorbar_added'):
            self.fig.colorbar(im, ax=ax_heat, orientation='vertical')
            self.colorbar_added = True

        self.canvas.draw()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyleSheet("QMainWindow { background-color: white; }")
    window = MLPBackpropApp()
    window.show()
    sys.exit(app.exec())
