import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.colors import ListedColormap
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QSlider, QLabel, QPushButton, QComboBox)
from PySide6.QtCore import Qt

class PerceptronEngine:
    """퍼셉트론의 수학적 로직을 담당하는 클래스"""
    def __init__(self, lr=0.1):
        # 가중치 초기화 (예시와 달리 임의 초기화로 학습 과정 노출)
        self.w = np.random.randn(2)
        self.b = np.random.randn()
        self.lr = lr

    def predict(self, x):
        # 수식: w1*x1 + w2*x2 + b
        # 활성화 함수: Heaviside Step Function (0 또는 1)
        return 1 if np.dot(self.w, x) + self.b >= 0 else 0

    def update(self, x, target):
        """학습 규칙에 따라 가중치와 편향 업데이트"""
        prediction = self.predict(x)
        error = target - prediction
        # 학습 공식: delta_w = lr * error * x
        self.w += self.lr * error * x
        self.b += self.lr * error
        return error

class VisualizerCanvas(FigureCanvas):
    """Matplotlib 그래프를 Qt 위젯으로 변환"""
    def __init__(self):
        # 교수님 요구사항에 맞춰 스타일 조정 가능
        self.fig, self.ax = plt.subplots(figsize=(6, 6), dpi=100)
        super().__init__(self.fig)

class PerceptronApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Visual Perceptron Simulator (with Decision Regions)")
        self.resize(1000, 650)
        
        # 모델 및 데이터 설정
        self.engine = PerceptronEngine()
        self.data_x = np.array([[0,0], [0,1], [1,0], [1,1]])
        self.data_y = np.array([0, 1, 1, 1]) # 기본 OR Gate
        
        # 컬러맵 설정 (결정 영역 색칠용)
        self.cmap_light = ListedColormap(['#FFAAAA', '#AAAAFF']) # 0영역(연빨강), 1영역(연파랑)
        
        self.init_ui()
        self.render_plot() # 초기 그래프 그리기

    def init_ui(self):
        # 메인 레이아웃
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QHBoxLayout(central_widget)

        # --- Sidebar (Controls) ---
        sidebar = QVBoxLayout()
        sidebar.setContentsMargins(10, 10, 10, 10)
        
        # 제목 및 설명
        title = QLabel("Perceptron Control Panel")
        title.setStyleSheet("font-size: 20px; font-weight: bold; color: #34495E; padding-bottom: 10px;")
        sidebar.addWidget(title)

        # 문제 데이터 선택 (콤보 박스)
        self.combo = QComboBox()
        self.combo.addItems(["OR Gate", "AND Gate", "NAND Gate", "XOR Gate"])
        self.combo.currentIndexChanged.connect(self.on_data_change)
        sidebar.addWidget(QLabel("Select Problem:"))
        sidebar.addWidget(self.combo)
        sidebar.addSpacing(15)

        # 모델 파라미터 조절 (슬라이더)
        # 생성 함수 사용: (이름, 레이아웃, 범위)
        self.w1_label, self.w1_slider = self.create_slider("Weight 1", sidebar)
        self.w2_label, self.w2_slider = self.create_slider("Weight 2", sidebar)
        self.b_label, self.b_slider = self.create_slider("Bias", sidebar)
        sidebar.addSpacing(15)

        # 학습 실행 버튼
        self.btn_train = QPushButton("Run 10 Epochs (Auto Train)")
        self.btn_train.setMinimumHeight(45)
        self.btn_train.setStyleSheet("background-color: #3498DB; color: white; font-size: 14px; font-weight: bold; border-radius: 5px;")
        self.btn_train.clicked.connect(self.train_model)
        sidebar.addWidget(self.btn_train)
        
        sidebar.addStretch() # 아래 여백 채우기
        layout.addLayout(sidebar, 1)

        # --- Main Display (Graph) ---
        self.canvas = VisualizerCanvas()
        layout.addWidget(self.canvas, 2)

    def create_slider(self, name, layout):
        """슬라이더 위젯 세트를 만들어주는 보조 함수"""
        label = QLabel(f"{name}: 0.0")
        slider = QSlider(Qt.Horizontal)
        slider.setRange(-20, 20) # 내부적으론 정수값 (-2.0 ~ 2.0 범위를 10배로 표현)
        slider.setValue(0)
        slider.valueChanged.connect(self.on_slider_move)
        layout.addWidget(label)
        layout.addWidget(slider)
        return label, slider

    def on_slider_move(self):
        """슬라이더 조절 시 실시간으로 퍼셉트론 상태 업데이트 및 그래프 그리기"""
        self.engine.w[0] = self.w1_slider.value() / 10
        self.engine.w[1] = self.w2_slider.value() / 10
        self.engine.b = self.b_slider.value() / 10
        self.update_labels()
        self.render_plot()

    def update_labels(self):
        """슬라이더 라벨에 현재 수치 표시"""
        self.w1_label.setText(f"Weight 1: {self.engine.w[0]:.1f}")
        self.w2_label.setText(f"Weight 2: {self.engine.w[1]:.1f}")
        self.b_label.setText(f"Bias: {self.engine.b:.1f}")

    def on_data_change(self, index):
        """콤보박스 선택에 따라 목표 데이터셋 변경"""
        gates = {
            0: [0, 1, 1, 1], # OR
            1: [0, 0, 0, 1], # AND
            2: [1, 1, 1, 0], # NAND
            3: [0, 1, 1, 0]  # XOR (분류 불가)
        }
        self.data_y = np.array(gates[index])
        self.render_plot()

    def train_model(self):
        """학습 버튼 클릭 시 10 에포크 동안 끊어서 학습 (GUI 응답성 유지)"""
        for _ in range(10):
            # 4개 데이터 포인트 전체에 대해 학습 규칙 적용
            for i in range(len(self.data_x)):
                self.engine.update(self.data_x[i], self.data_y[i])
        
        # 슬라이더 위치를 학습된 결과값으로 동기화
        self.w1_slider.setValue(int(self.engine.w[0]*10))
        self.w2_slider.setValue(int(self.engine.w[1]*10))
        self.b_slider.setValue(int(self.engine.b*10))
        self.update_labels()
        self.render_plot()

    def render_plot(self):
        """(핵심 변경 부분) 데이터 포인트와 퍼셉트론의 결정 영역을 한 그래프에 그리기"""
        self.canvas.ax.clear() # 기존 그래프 지우기
        
        # --- 1단계: 전체 결정 영역 색칠 (핵심) ---
        # 그래프 영역의 격자망 생성
        x_min, x_max = -0.5, 1.5
        y_min, y_max = -0.5, 1.5
        h = 0.01 # 격자 촘촘함 (작을수록 부드럽지만 느려짐)
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                             np.arange(y_min, y_max, h))
        
        # 격자의 모든 점에 대해 퍼셉트론 예측값 계산
        grid_points = np.c_[xx.ravel(), yy.ravel()]
        Z = np.array([self.engine.predict(p) for p in grid_points])
        Z = Z.reshape(xx.shape)
        
        # 예측값(0, 1)에 따라 컬러맵으로 영역 채우기
        self.canvas.ax.contourf(xx, yy, Z, cmap=self.cmap_light, alpha=0.8, zorder=1)
        
        # --- 2단계: 결정 경계선 그리기 (보조) ---
        # 직선 방정식: w1*x + w2*y + b = 0 => y = -(w1*x + b) / w2
        x_range = np.array([-0.5, 1.5])
        if self.engine.w[1] != 0:
            y_range = -(self.engine.w[0] * x_range + self.engine.b) / self.engine.w[1]
            self.canvas.ax.plot(x_range, y_range, color='#2ECC71', lw=3, label='Decision Line', zorder=2)
        
        # --- 3단계: 실제 데이터 포인트 플로팅 ---
        for i in range(len(self.data_y)):
            # 1은 파란색, 0은 빨간색 점
            color = '#3498DB' if self.data_y[i] == 1 else '#E74C3C'
            self.canvas.ax.scatter(self.data_x[i,0], self.data_x[i,1], c=color, s=200, edgecolors='white', zorder=5)
            # 포인트 옆에 좌표 텍스트 추가
            self.canvas.ax.text(self.data_x[i,0]+0.05, self.data_x[i,1]+0.05, f"({self.data_x[i,0]},{self.data_x[i,1]})")

        # --- 4단계: 그래프 설정 ---
        self.canvas.ax.set_xlim(-0.5, 1.5)
        self.canvas.ax.set_ylim(-0.5, 1.5)
        self.canvas.ax.set_aspect('equal') # 비율 고정 (정사각형 그래프)
        self.canvas.ax.grid(True, linestyle='--', alpha=0.3)
        self.canvas.ax.set_title("Perceptron Classification (colored areas)", pad=20)
        self.canvas.draw() # 변경 사항 반영하여 그리기

if __name__ == "__main__":
    app = QApplication(sys.argv)
    # 앱 전체 스타일 시트 적용
    app.setStyleSheet("QMainWindow { background-color: white; }")
    window = PerceptronApp()
    window.show()
    sys.exit(app.exec())
