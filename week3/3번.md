import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QSlider, QLabel, QPushButton, QTextBrowser, QFrame)
from PySide6.QtCore import Qt

class NeuralNetwork:
    """2-3-1 신경망의 순전파 연산을 담당하는 모델 클래스"""
    def __init__(self):
        self.init_weights()

    def init_weights(self):
        # 2-3-1 구조의 가중치 및 편향 초기화
        np.random.seed() # 무작위성 확보
        self.W1 = np.random.randn(2, 3)
        self.b1 = np.random.randn(3)
        self.W2 = np.random.randn(3, 1)
        self.b2 = np.random.randn(1)

    def relu(self, z):
        return np.maximum(0, z)

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def forward(self, x):
        """순전파 연산 수행 및 중간 과정 반환"""
        # Layer 1
        z1 = np.dot(x, self.W1) + self.b1
        a1 = self.relu(z1)
        
        # Layer 2
        z2 = np.dot(a1, self.W2) + self.b2
        a2 = self.sigmoid(z2)
        
        return z1, a1, z2, a2

class VisualizerCanvas(FigureCanvas):
    def __init__(self):
        # 4분할 플롯 생성
        self.fig, self.axs = plt.subplots(2, 2, figsize=(10, 8), dpi=100)
        self.fig.tight_layout(pad=4.0)
        super().__init__(self.fig)

class ForwardPropApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Lab 3: Forward Propagation Visualizer (2-3-1)")
        self.resize(1400, 850)
        
        self.nn = NeuralNetwork()
        self.x_input = np.array([0.5, 0.8]) # 초기 입력값
        
        self.init_ui()
        self.update_visualization()

    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QHBoxLayout(central_widget)

        # --- Left Panel: 4분할 그래프 ---
        self.canvas = VisualizerCanvas()
        layout.addWidget(self.canvas, 7) 

        # --- Right Panel: 컨트롤 & 텍스트 설명 ---
        right_panel = QVBoxLayout()
        
        # 1. 제어부
        control_frame = QFrame()
        control_frame.setStyleSheet("background-color: #F8F9F9; border-radius: 5px; padding: 10px;")
        control_layout = QVBoxLayout(control_frame)
        
        title = QLabel("⚙️ Inputs & Controls")
        title.setStyleSheet("font-size: 16px; font-weight: bold; color: #2C3E50;")
        control_layout.addWidget(title)

        # X1 슬라이더
        self.lbl_x1 = QLabel(f"Input X1: {self.x_input[0]:.2f}")
        self.slider_x1 = QSlider(Qt.Horizontal)
        self.slider_x1.setRange(-50, 50)
        self.slider_x1.setValue(int(self.x_input[0]*10))
        self.slider_x1.valueChanged.connect(self.on_input_change)
        control_layout.addWidget(self.lbl_x1)
        control_layout.addWidget(self.slider_x1)

        # X2 슬라이더
        self.lbl_x2 = QLabel(f"Input X2: {self.x_input[1]:.2f}")
        self.slider_x2 = QSlider(Qt.Horizontal)
        self.slider_x2.setRange(-50, 50)
        self.slider_x2.setValue(int(self.x_input[1]*10))
        self.slider_x2.valueChanged.connect(self.on_input_change)
        control_layout.addWidget(self.lbl_x2)
        control_layout.addWidget(self.slider_x2)

        # 무작위 가중치 버튼
        self.btn_random = QPushButton("🎲 Randomize Weights")
        self.btn_random.setMinimumHeight(40)
        self.btn_random.setStyleSheet("background-color: #9B59B6; color: white; font-weight: bold;")
        self.btn_random.clicked.connect(self.randomize_weights)
        control_layout.addWidget(self.btn_random)

        right_panel.addWidget(control_frame)

        # 2. 텍스트 설명부
        self.text_browser = QTextBrowser()
        self.text_browser.setStyleSheet("background-color: #FFFFFF; font-size: 14px; padding: 10px; border: 1px solid #BDC3C7;")
        self.set_explanation_text()
        right_panel.addWidget(self.text_browser)

        layout.addLayout(right_panel, 3)

    def set_explanation_text(self):
        html = """
        <h2 style='color: #2980B9;'>🧠 Forward Propagation 구조</h2>
        <p>순전파는 입력에서 출력까지 데이터가 <b>앞으로(Forward) 흐르는 과정</b>입니다.</p>
        
        <h3>1. Layer 1 (Input &rarr; Hidden)</h3>
        <ul>
            <li><b>선형 결합:</b> <code style='background:#E8DAEF; padding:2px;'>Z_1 = X @ W_1 + b_1</code></li>
            <li><b>활성화 (ReLU):</b> <code style='background:#E8DAEF; padding:2px;'>A_1 = ReLU(Z_1)</code></li>
            <li><i>* ReLU는 0 이하의 값(음수)을 0으로 차단합니다.</i></li>
        </ul>

        <h3>2. Layer 2 (Hidden &rarr; Output)</h3>
        <ul>
            <li><b>선형 결합:</b> <code style='background:#D4E6F1; padding:2px;'>Z_2 = A_1 @ W_2 + b_2</code></li>
            <li><b>최종 활성화 (Sigmoid):</b> <code style='background:#D4E6F1; padding:2px;'>A_2 = Sigmoid(Z_2)</code></li>
            <li><i>* Sigmoid는 최종 출력을 0과 1 사이의 확률값으로 압축합니다.</i></li>
        </ul>
        <p>슬라이더를 움직이거나 가중치를 초기화하여 막대그래프와 수식이 어떻게 변하는지 확인해보세요!</p>
        """
        self.text_browser.setHtml(html)

    def on_input_change(self):
        self.x_input[0] = self.slider_x1.value() / 10.0
        self.x_input[1] = self.slider_x2.value() / 10.0
        self.lbl_x1.setText(f"Input X1: {self.x_input[0]:.2f}")
        self.lbl_x2.setText(f"Input X2: {self.x_input[1]:.2f}")
        self.update_visualization()

    def randomize_weights(self):
        self.nn.init_weights()
        self.update_visualization()

    def draw_network_diagram(self, ax):
        ax.clear()
        ax.axis('off')
        
        # 노드 좌표 설정 (2-3-1)
        layer_x = [0.1, 0.5, 0.9]
        nodes_y = [
            [0.35, 0.65],           # Input 2개
            [0.2, 0.5, 0.8],        # Hidden 3개
            [0.5]                   # Output 1개
        ]
        colors = ['#3498DB', '#2ECC71', '#E74C3C'] # 파랑, 초록, 빨강
        labels = [['x1', 'x2'], ['h1', 'h2', 'h3'], ['y']]

        # 선 그리기
        for i in range(2):
            for y1 in nodes_y[i]:
                for y2 in nodes_y[i+1]:
                    ax.plot([layer_x[i], layer_x[i+1]], [y1, y2], 'k-', alpha=0.3)

        # 노드 그리기
        for i, (x_pos, y_positions) in enumerate(zip(layer_x, nodes_y)):
            for j, y_pos in enumerate(y_positions):
                circle = plt.Circle((x_pos, y_pos), 0.08, color=colors[i], ec='black', zorder=5)
                ax.add_patch(circle)
                ax.text(x_pos, y_pos, labels[i][j], color='white', weight='bold', 
                        ha='center', va='center', zorder=6)
                
        ax.set_title("Neural Network Architecture\n(2 - 3 - 1)", fontsize=10)

    def draw_matrix_operations(self, ax, z1, a1, z2, a2):
        ax.clear()
        ax.axis('off')
        ax.set_title("Matrix Operations (Real-time)", fontsize=10, weight='bold')

        # 텍스트로 행렬 연산 과정 출력
        text = f"Forward Pass:\n\n"
        text += f"X = [{self.x_input[0]:.2f}, {self.x_input[1]:.2f}]\n\n"
        
        text += f"Layer 1 (Input -> Hidden):\n"
        text += f"W1 = \n{np.array2string(self.nn.W1, precision=2, floatmode='fixed')}\n"
        text += f"Z1 = X @ W1 + b1\n"
        text += f"   = [{z1[0]:.2f}, {z1[1]:.2f}, {z1[2]:.2f}]\n"
        text += f"A1 = ReLU(Z1)\n"
        text += f"   = [{a1[0]:.2f}, {a1[1]:.2f}, {a1[2]:.2f}]\n\n"
        
        text += f"Layer 2 (Hidden -> Output):\n"
        text += f"Z2 = A1 @ W2 + b2\n"
        text += f"   = [{z2[0]:.2f}]\n"
        text += f"A2 = Sigmoid(Z2)\n"
        text += f"   = [{a2[0]:.2f}]\n"

        ax.text(0.05, 0.95, text, fontsize=9, family='monospace', va='top', 
                bbox=dict(facecolor='#F4F6F6', edgecolor='#BDC3C7', boxstyle='round,pad=0.5'))

    def update_visualization(self):
        # 1. 순전파 계산
        z1, a1, z2, a2 = self.nn.forward(self.x_input)
        
        # 4개의 축 가져오기
        ax_net, ax_l1, ax_l2, ax_math = self.axs.flatten()

        # [1] Top-Left: 네트워크 다이어그램
        self.draw_network_diagram(ax_net)

        # [2] Top-Right: Layer 1 Bar Chart (Z1 vs A1)
        ax_l1.clear()
        indices = np.arange(3)
        width = 0.35
        ax_l1.bar(indices - width/2, z1, width, label='Z1 (Before ReLU)', color='#F39C12')
        ax_l1.bar(indices + width/2, a1, width, label='A1 (After ReLU)', color='#3498DB')
        ax_l1.set_xticks(indices)
        ax_l1.set_xticklabels(['Neuron 1', 'Neuron 2', 'Neuron 3'])
        ax_l1.set_title("Layer 1: Input -> Hidden (ReLU)")
        ax_l1.legend(fontsize=8)
        ax_l1.grid(True, axis='y', linestyle='--', alpha=0.5)

        # [3] Bottom-Left: Layer 2 Bar Chart (Z2 vs A2)
        ax_l2.clear()
        ax_l2.barh(['Z2 (Before Sigmoid)'], [z2[0]], color='#E74C3C', height=0.4)
        ax_l2.barh(['A2 (Final Output)'], [a2[0]], color='#2ECC71', height=0.4)
        ax_l2.set_xlim(min(-2, z2[0]-1), max(2, z2[0]+1)) # 보기 좋게 범위 조정
        ax_l2.set_title("Layer 2: Hidden -> Output (Sigmoid)")
        ax_l2.grid(True, axis='x', linestyle='--', alpha=0.5)

        # [4] Bottom-Right: 행렬 연산 실시간 출력
        self.draw_matrix_operations(ax_math, z1, a1, z2, a2)

        self.canvas.draw()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyleSheet("QMainWindow { background-color: white; }")
    window = ForwardPropApp()
    window.show()
    sys.exit(app.exec())
