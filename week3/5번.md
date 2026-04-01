import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QSlider, QLabel, QTextBrowser, QFrame)
from PySide6.QtCore import Qt

class FastUniversalApproximator:
    """실시간 시각화를 위해 최소제곱법(Pseudo-inverse)을 사용하는 신경망 모델"""
    def __init__(self, max_neurons=500):
        # 애니메이션의 부드러움을 위해 최대 뉴런의 가중치를 미리 생성해 둠 (고정된 무작위 값)
        np.random.seed(42)
        self.max_neurons = max_neurons
        # 가중치 범위를 넓게 주어 다양한 주파수 커버
        self.W1_pool = np.random.randn(1, max_neurons) * 10 
        self.b1_pool = np.random.randn(1, max_neurons) * 10

    def fit_predict(self, x_train, y_train, x_test, n_neurons):
        # 슬라이더 값(n_neurons)만큼 가중치를 잘라서 사용 (스무스한 변화 보장)
        W1 = self.W1_pool[:, :n_neurons]
        b1 = self.b1_pool[:, :n_neurons]

        # 1. Hidden Layer Output (ReLU) - Training Set
        H_train = np.maximum(0, np.dot(x_train, W1) + b1)
        H_train_aug = np.hstack([H_train, np.ones((x_train.shape[0], 1))]) # Bias 항 추가

        # 2. Output Weights (Pseudo-inverse를 통한 1-step 즉시 학습)
        W2 = np.linalg.pinv(H_train_aug).dot(y_train)

        # 3. Predict on Test Set (부드러운 곡선 그리기용)
        H_test = np.maximum(0, np.dot(x_test, W1) + b1)
        H_test_aug = np.hstack([H_test, np.ones((x_test.shape[0], 1))])
        y_pred = np.dot(H_test_aug, W2)

        # MSE 계산
        mse = np.mean((y_train - np.dot(H_train_aug, W2))**2)
        return y_pred, mse

class VisualizerCanvas(FigureCanvas):
    def __init__(self):
        self.fig, self.axs = plt.subplots(1, 3, figsize=(15, 5), dpi=100)
        self.fig.tight_layout(pad=4.0)
        super().__init__(self.fig)

class UATApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Lab 5: Universal Approximation Theorem Visualizer")
        self.resize(1600, 600)

        self.model = FastUniversalApproximator()
        self.n_neurons = 10 # 초기 뉴런 수

        self.generate_data()
        self.init_ui()
        self.update_plots()

    def generate_data(self):
        # 평가를 위한 촘촘한 X 값 (곡선 그리기 용)
        self.x_dense = np.linspace(0, 1, 200).reshape(-1, 1)
        # 실제 학습에 사용될 듬성듬성한 X 값 (빨간 점)
        self.x_train = np.linspace(0.02, 0.98, 25).reshape(-1, 1)

        # 1. Sine Wave
        self.y_true_sine = np.sin(2 * np.pi * self.x_dense)
        self.y_train_sine = np.sin(2 * np.pi * self.x_train)

        # 2. Step Function
        self.y_true_step = np.where(self.x_dense >= 0.5, 1.0, -1.0)
        self.y_train_step = np.where(self.x_train >= 0.5, 1.0, -1.0)

        # 3. Complex Function
        self.y_true_comp = np.sin(2 * np.pi * self.x_dense) + 0.5 * np.sin(6 * np.pi * self.x_dense)
        self.y_train_comp = np.sin(2 * np.pi * self.x_train) + 0.5 * np.sin(6 * np.pi * self.x_train)

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
        control_frame.setStyleSheet("background-color: #F8F9F9; border-radius: 5px; padding: 15px;")
        control_layout = QVBoxLayout(control_frame)
        
        title = QLabel("⚙️ Hyperparameter")
        title.setStyleSheet("font-size: 16px; font-weight: bold; color: #2C3E50;")
        control_layout.addWidget(title)

        # 뉴런 수 슬라이더
        self.lbl_neurons = QLabel(f"Number of Hidden Neurons: {self.n_neurons}")
        self.lbl_neurons.setStyleSheet("font-size: 14px; font-weight: bold; color: #E74C3C;")
        self.slider_neurons = QSlider(Qt.Horizontal)
        self.slider_neurons.setRange(1, 200) # 1개부터 200개까지
        self.slider_neurons.setValue(self.n_neurons)
        self.slider_neurons.valueChanged.connect(self.on_slider_change)
        
        control_layout.addWidget(self.lbl_neurons)
        control_layout.addWidget(self.slider_neurons)

        right_panel.addWidget(control_frame)

        # 2. 텍스트 설명부
        self.text_browser = QTextBrowser()
        self.text_browser.setStyleSheet("background-color: #FFFFFF; font-size: 14px; padding: 15px; border: 1px solid #BDC3C7;")
        self.set_explanation_text()
        right_panel.addWidget(self.text_browser)

        layout.addLayout(right_panel, 3)

    def set_explanation_text(self):
        html = """
        <h2 style='color: #2980B9;'>🌌 Universal Approximation Theorem</h2>
        <p><b>"1개의 은닉층(Hidden Layer)을 가진 신경망이라도, 뉴런의 개수(Width)가 충분히 많다면 세상의 어떤 연속 함수든 근사(Approximate)할 수 있다."</b></p>
        
        <h3>슬라이더를 움직여보세요!</h3>
        <ul>
            <li><b>적은 뉴런 (N < 5):</b> 선이 직선의 조합으로 꺾이며 매우 거칠게 근사됩니다.</li>
            <li><b>많은 뉴런 (N > 50):</b> 곡선과 다중 주파수 함수까지 완벽하게 따라갑니다. 계단 함수의 불연속점도 거의 직각으로 묘사해 냅니다.</li>
        </ul>

        <h3 style='color: #8E44AD;'>💡 핵심 통찰 (Width vs Depth)</h3>
        <p>이론적으로는 은닉층이 1개만 있어도 뉴런을 무한히 늘리면(Width) 모든 문제를 풀 수 있습니다. <br><br>
        하지만 복잡한 문제일수록 필요한 뉴런의 수가 <b>기하급수적(Exponential)</b>으로 폭발합니다. 따라서 층을 1개가 아닌 여러 층(Depth)으로 쌓는 <b>Deep Learning</b>이 실용적으로 훨씬 효율적인 구조라는 것이 밝혀졌습니다.</p>
        """
        self.text_browser.setHtml(html)

    def on_slider_change(self):
        self.n_neurons = self.slider_neurons.value()
        self.lbl_neurons.setText(f"Number of Hidden Neurons: {self.n_neurons}")
        self.update_plots()

    def update_plots(self):
        ax_sine, ax_step, ax_comp = self.axs.flatten()
        
        # 각 함수별 예측값 계산
        pred_sine, mse_sine = self.model.fit_predict(self.x_train, self.y_train_sine, self.x_dense, self.n_neurons)
        pred_step, mse_step = self.model.fit_predict(self.x_train, self.y_train_step, self.x_dense, self.n_neurons)
        pred_comp, mse_comp = self.model.fit_predict(self.x_train, self.y_train_comp, self.x_dense, self.n_neurons)

        # 그리기 공통 함수
        def draw_plot(ax, title, y_true, y_train, y_pred, mse):
            ax.clear()
            # 실제 함수 (파란 실선)
            ax.plot(self.x_dense, y_true, label='True Function', color='#3498DB', lw=2, alpha=0.7)
            # 학습 데이터 (빨간 점)
            ax.scatter(self.x_train, y_train, color='red', s=40, label='Training Data', zorder=5)
            # 신경망 근사 함수 (빨간 점선)
            ax.plot(self.x_dense, y_pred, label=f'NN ({self.n_neurons} neurons)', color='#E74C3C', lw=2, linestyle='--')
            
            ax.set_title(f"{title} (MSE: {mse:.4f})")
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_ylim(-1.5, 1.5)
            ax.grid(True, linestyle='--', alpha=0.5)
            ax.legend(loc='lower right', fontsize=8)

        # 3개의 플롯 업데이트
        draw_plot(ax_sine, "Sine Wave", self.y_true_sine, self.y_train_sine, pred_sine, mse_sine)
        draw_plot(ax_step, "Step Function", self.y_true_step, self.y_train_step, pred_step, mse_step)
        draw_plot(ax_comp, "Complex Function", self.y_true_comp, self.y_train_comp, pred_comp, mse_comp)

        self.canvas.draw()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyleSheet("QMainWindow { background-color: white; }")
    window = UATApp()
    window.show()
    sys.exit(app.exec())
