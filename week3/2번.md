import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QSlider, QLabel, QTextBrowser, QFrame)
from PySide6.QtCore import Qt

class ActivationMath:
    """활성화 함수와 도함수를 계산하는 클래스"""
    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))
    
    @staticmethod
    def sigmoid_deriv(x):
        s = ActivationMath.sigmoid(x)
        return s * (1 - s)

    @staticmethod
    def tanh(x):
        return np.tanh(x)
    
    @staticmethod
    def tanh_deriv(x):
        return 1 - np.tanh(x)**2

    @staticmethod
    def relu(x):
        return np.maximum(0, x)
    
    @staticmethod
    def relu_deriv(x):
        return np.where(x > 0, 1.0, 0.0)

    @staticmethod
    def leaky_relu(x, alpha=0.1):
        return np.where(x > 0, x, alpha * x)
    
    @staticmethod
    def leaky_relu_deriv(x, alpha=0.1):
        return np.where(x > 0, 1.0, alpha)


class VisualizerCanvas(FigureCanvas):
    def __init__(self):
        # 2x2 그리드로 플롯 생성
        self.fig, self.axs = plt.subplots(2, 2, figsize=(10, 8), dpi=100)
        self.fig.tight_layout(pad=4.0)
        super().__init__(self.fig)


class ActivationApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Neural Network: Activation Functions Visualizer")
        self.resize(1400, 800)
        
        self.x_range = 5.0 # 초기 X축 범위 (-5 to 5)
        self.alpha = 0.1   # 초기 Leaky ReLU alpha 값
        
        self.init_ui()
        self.render_plots()

    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QHBoxLayout(central_widget)

        # --- Left Panel: 4분할 그래프 ---
        self.canvas = VisualizerCanvas()
        layout.addWidget(self.canvas, 7) # 비율 7

        # --- Right Panel: 컨트롤 & 텍스트 설명 ---
        right_panel = QVBoxLayout()
        
        # 1. 제어부 (Controls)
        control_frame = QFrame()
        control_frame.setStyleSheet("background-color: #F8F9F9; border-radius: 5px; padding: 10px;")
        control_layout = QVBoxLayout(control_frame)
        
        title = QLabel("⚙️ Parameters")
        title.setStyleSheet("font-size: 16px; font-weight: bold; color: #2C3E50;")
        control_layout.addWidget(title)

        # X Range 슬라이더
        self.lbl_x = QLabel(f"X Axis Range: ±{self.x_range:.1f}")
        self.slider_x = QSlider(Qt.Horizontal)
        self.slider_x.setRange(20, 100) # 2.0 ~ 10.0
        self.slider_x.setValue(50)
        self.slider_x.valueChanged.connect(self.on_slider_change)
        control_layout.addWidget(self.lbl_x)
        control_layout.addWidget(self.slider_x)

        # Leaky ReLU Alpha 슬라이더
        self.lbl_alpha = QLabel(f"Leaky ReLU Alpha: {self.alpha:.2f}")
        self.slider_alpha = QSlider(Qt.Horizontal)
        self.slider_alpha.setRange(0, 50) # 0.0 ~ 0.5
        self.slider_alpha.setValue(10)
        self.slider_alpha.valueChanged.connect(self.on_slider_change)
        control_layout.addWidget(self.lbl_alpha)
        control_layout.addWidget(self.slider_alpha)

        right_panel.addWidget(control_frame)

        # 2. 설명부 (Text Panel)
        self.text_browser = QTextBrowser()
        self.text_browser.setStyleSheet("background-color: #FFFFFF; font-size: 14px; padding: 10px; border: 1px solid #BDC3C7;")
        self.update_text_panel()
        right_panel.addWidget(self.text_browser)

        layout.addLayout(right_panel, 3) # 비율 3

    def on_slider_change(self):
        self.x_range = self.slider_x.value() / 10.0
        self.alpha = self.slider_alpha.value() / 100.0
        
        self.lbl_x.setText(f"X Axis Range: ±{self.x_range:.1f}")
        self.lbl_alpha.setText(f"Leaky ReLU Alpha: {self.alpha:.2f}")
        self.render_plots()

    def update_text_panel(self):
        html_content = """
        <h2 style='color: #2980B9;'>📊 결과 해석 및 요약</h2>
        
        <h3>1. Activation Functions (좌상단)</h3>
        <ul>
            <li>신경망에 비선형성을 부여하는 핵심 함수들입니다.</li>
            <li>입력값 x에 따라 출력값 y의 형태가 어떻게 변하는지 한눈에 비교할 수 있습니다.</li>
        </ul>

        <h3>2. Derivatives / Gradients (우상단)</h3>
        <ul style='color: #C0392B;'>
            <li><b>그래디언트 소실(Vanishing Gradient) 문제:</b> Sigmoid와 Tanh는 입력값(x)이 커지거나 작아지면 도함수(기울기)가 0에 수렴합니다. 이는 깊은 신경망에서 학습이 멈추는 원인이 됩니다.</li>
            <li>반면 ReLU 계열은 양수 구간에서 기울기가 항상 1로 유지되어 이 문제를 해결합니다.</li>
        </ul>

        <h3>3. Sigmoid vs Tanh (좌하단)</h3>
        <ul>
            <li><b>Sigmoid:</b> 출력 범위가 (0, 1)이며, 중심이 0.5입니다.</li>
            <li><b>Tanh:</b> 출력 범위가 (-1, 1)이며, <b>중심이 0(Zero-centered)</b>입니다. 최적화 과정에서 Sigmoid보다 더 나은 성능을 보이는 경우가 많습니다.</li>
        </ul>

        <h3>4. ReLU vs Leaky ReLU (우하단)</h3>
        <ul>
            <li><b>Dying ReLU 문제:</b> 일반 ReLU는 x < 0 구간에서 기울기가 완전히 0이 되어, 해당 뉴런이 영원히 학습되지 않고 '죽는' 현상이 발생할 수 있습니다.</li>
            <li><b>Leaky ReLU:</b> 음수 구간에 약간의 기울기(현재 설정된 Alpha 값)를 주어 이 문제를 방지합니다. 슬라이더를 움직여 차이를 확인해보세요.</li>
        </ul>
        """
        self.text_browser.setHtml(html_content)

    def render_plots(self):
        x = np.linspace(-self.x_range, self.x_range, 400)
        
        # 4개의 축 초기화
        for ax in self.axs.flat:
            ax.clear()
            ax.grid(True, linestyle='--', alpha=0.6)
            ax.axhline(0, color='black', lw=0.8)
            ax.axvline(0, color='black', lw=0.8)

        ax1, ax2, ax3, ax4 = self.axs.flatten()

        # [1] Activation Functions
        ax1.plot(x, ActivationMath.sigmoid(x), label='Sigmoid', lw=2)
        ax1.plot(x, ActivationMath.tanh(x), label='Tanh', lw=2)
        ax1.plot(x, ActivationMath.relu(x), label='ReLU', lw=2)
        ax1.plot(x, ActivationMath.leaky_relu(x, self.alpha), '--', label='Leaky ReLU', lw=2)
        ax1.set_title("Activation Functions")
        ax1.legend(loc='upper left')

        # [2] Derivatives
        ax2.plot(x, ActivationMath.sigmoid_deriv(x), label="Sigmoid'", lw=2)
        ax2.plot(x, ActivationMath.tanh_deriv(x), label="Tanh'", lw=2)
        ax2.plot(x, ActivationMath.relu_deriv(x), label="ReLU'", lw=2)
        ax2.plot(x, ActivationMath.leaky_relu_deriv(x, self.alpha), '--', label="Leaky ReLU'", lw=2)
        ax2.set_title("Derivatives (Gradients)")
        ax2.legend(loc='upper left')

        # [3] Sigmoid vs Tanh (중심 비교)
        ax3.plot(x, ActivationMath.sigmoid(x), label='Sigmoid: (0, 1)', lw=2.5)
        ax3.plot(x, ActivationMath.tanh(x), label='Tanh: (-1, 1)', lw=2.5)
        ax3.axhline(0.5, color='blue', linestyle=':', alpha=0.5, label='Sigmoid center')
        ax3.set_title("Sigmoid vs Tanh (Center Difference)")
        ax3.legend(loc='upper left')

        # [4] ReLU vs Leaky ReLU (Dying ReLU)
        ax4.plot(x, ActivationMath.relu(x), label='ReLU (x < 0: Dead)', lw=2.5)
        ax4.plot(x, ActivationMath.leaky_relu(x, self.alpha), label=f'Leaky ReLU (alpha={self.alpha:.2f})', lw=2.5)
        ax4.set_title("ReLU vs Leaky ReLU (Dying ReLU Issue)")
        ax4.legend(loc='upper left')

        # 공통 설정
        for ax in self.axs.flat:
            ax.set_xlim(-self.x_range, self.x_range)

        # y축 범위 고정 (시각적 안정성을 위해)
        ax1.set_ylim(-1.5, self.x_range if self.x_range <= 5 else 5)
        ax2.set_ylim(-0.1, 1.1)
        ax3.set_ylim(-1.1, 1.1)
        ax4.set_ylim(-1.5, self.x_range if self.x_range <= 5 else 5)

        self.canvas.draw()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyleSheet("QMainWindow { background-color: white; }")
    window = ActivationApp()
    window.show()
    sys.exit(app.exec())
