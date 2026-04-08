import sys
import os
import datetime
import numpy as np
import tensorflow as tf
from tensorflow import keras
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                               QHBoxLayout, QPushButton, QComboBox, QTextEdit, QLabel, QSlider)
from PySide6.QtCore import QThread, Signal, Qt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

# GitHub 관리를 위한 출력 디렉토리 설정 (PRD 요구사항)
OUTPUT_DIR = "outputs"
LOG_FILE = "experiment_history.md"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------------------------------------------
# [Superpowers: 실시간 학습 모니터링 콜백]
# ---------------------------------------------------------
class RealtimeCallback(keras.callbacks.Callback):
    def __init__(self, thread_obj, x_test, y_true):
        super().__init__()
        self.thread_obj = thread_obj
        self.x_test = x_test
        self.y_true = y_true

    def on_epoch_end(self, epoch, logs=None):
        loss = logs.get('loss')
        self.thread_obj.loss_signal.emit(epoch, loss)
        
        # UI 과부하 방지 및 시각적 애니메이션 효과 (Superpower)
        if epoch % 5 == 0 or epoch == self.params['epochs'] - 1:
            y_pred = self.model.predict(self.x_test, verbose=0)
            self.thread_obj.pred_signal.emit(self.x_test, self.y_true, y_pred)

# ---------------------------------------------------------
# [TRD 연동: Keras 학습 백그라운드 스레드]
# ---------------------------------------------------------
class TrainingThread(QThread):
    log_signal = Signal(str)
    loss_signal = Signal(int, float)
    pred_signal = Signal(object, object, object)
    finished_signal = Signal(float) # 최종 Loss 값 전달

    def __init__(self, lab_name, epochs, learning_rate):
        super().__init__()
        self.lab_name = lab_name
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.final_loss = 0.0

    def run(self):
        if "Lab 1" in self.lab_name:
            self.run_lab1()
        elif "Lab 3" in self.lab_name:
            self.run_lab3()
        else:
            self.log_signal.emit("해당 Lab은 구현 중입니다.")
            
        self.finished_signal.emit(self.final_loss)

    def run_lab1(self):
        self.log_signal.emit(f"[{self.lab_name}] 학습 시작... (Epochs: {self.epochs}, LR: {self.learning_rate})")
        x = np.linspace(-np.pi, np.pi, 200).reshape(-1, 1)
        y = np.sin(x)
        
        # TRD 요구사항: np.random.permutation을 통한 셔플링 (Validation Split 이슈 방지)
        indices = np.random.permutation(len(x))
        x_shuffled, y_shuffled = x[indices], y[indices]

        model = keras.Sequential([
            keras.layers.Dense(64, activation='tanh', input_shape=(1,)),
            keras.layers.Dense(64, activation='tanh'),
            keras.layers.Dense(1, activation='linear')
        ])
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate), loss='mse')
        
        realtime_cb = RealtimeCallback(self, x, y)
        history = model.fit(x_shuffled, y_shuffled, epochs=self.epochs, verbose=0, callbacks=[realtime_cb])
        self.final_loss = history.history['loss'][-1]

    def run_lab3(self):
        self.log_signal.emit(f"[{self.lab_name}] 과적합 데모 시작... (Epochs: {self.epochs}, LR: {self.learning_rate})")
        x = np.linspace(-2, 2, 200).reshape(-1, 1)
        y = np.sin(2 * x) + 0.5 * x + np.random.normal(0, 0.3, x.shape) # 노이즈 추가

        model = keras.Sequential([
            keras.layers.Dense(128, activation='relu', input_shape=(1,)),
            keras.layers.Dense(64, activation='relu'),
            keras.layers.Dense(32, activation='relu'),
            keras.layers.Dense(1, activation='linear')
        ])
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate), loss='mse')
        
        x_test = np.linspace(-2.5, 2.5, 200).reshape(-1, 1)
        y_true = np.sin(2 * x_test) + 0.5 * x_test
        
        realtime_cb = RealtimeCallback(self, x_test, y_true)
        history = model.fit(x, y, epochs=self.epochs, verbose=0, callbacks=[realtime_cb])
        self.final_loss = history.history['loss'][-1]

# ---------------------------------------------------------
# [GUI 및 PRD 산출물 저장 로직]
# ---------------------------------------------------------
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AI Physics Course - GitHub Linked Simulator")
        self.resize(1100, 700)

        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QHBoxLayout(main_widget)

        # 컨트롤 패널
        control_layout = QVBoxLayout()
        control_layout.addWidget(QLabel("<b>실험 (Lab) 선택</b>"))
        self.lab_combo = QComboBox()
        self.lab_combo.addItems(["Lab 1: 완벽한 1D 함수 근사", "Lab 2: 포물선 운동 회귀 (준비중)", "Lab 3: 과적합(Overfitting) 데모"])
        control_layout.addWidget(self.lab_combo)

        # 파라미터 조절 슬라이더 (Superpowers)
        self.epoch_label = QLabel("<b>Epochs: 100</b>")
        control_layout.addWidget(self.epoch_label)
        self.epoch_slider = QSlider(Qt.Horizontal)
        self.epoch_slider.setRange(10, 500)
        self.epoch_slider.setValue(100)
        self.epoch_slider.valueChanged.connect(lambda v: self.epoch_label.setText(f"<b>Epochs: {v}</b>"))
        control_layout.addWidget(self.epoch_slider)

        control_layout.addWidget(QLabel("<b>Learning Rate</b>"))
        self.lr_combo = QComboBox()
        self.lr_combo.addItems(["0.01", "0.001", "0.0001"])
        self.lr_combo.setCurrentIndex(1)
        control_layout.addWidget(self.lr_combo)

        self.run_btn = QPushButton("▶ 시뮬레이션 실행 및 GitHub 아티팩트 생성")
        self.run_btn.setMinimumHeight(50)
        self.run_btn.clicked.connect(self.start_training)
        control_layout.addWidget(self.run_btn)

        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        control_layout.addWidget(QLabel("<b>터미널 로그</b>"))
        control_layout.addWidget(self.log_text)

        # 그래프 영역
        self.figure = Figure(figsize=(10, 5), dpi=100)
        self.canvas = FigureCanvas(self.figure)
        self.ax_loss = self.figure.add_subplot(121)
        self.ax_pred = self.figure.add_subplot(122)
        self.init_plots()

        layout.addLayout(control_layout, 1)
        layout.addWidget(self.canvas, 3)

        self.thread = None
        self.epoch_history = []
        self.loss_history = []

    def init_plots(self):
        self.ax_loss.set_title("실시간 MSE Loss")
        self.ax_loss.grid(True, linestyle='--')
        self.ax_pred.set_title("예측 모델 피팅 과정")
        self.ax_pred.grid(True, linestyle='--')
        self.figure.tight_layout()

    def start_training(self):
        self.run_btn.setEnabled(False)
        self.log_text.clear()
        self.epoch_history.clear()
        self.loss_history.clear()
        
        self.ax_loss.clear()
        self.ax_pred.clear()
        self.init_plots()
        self.canvas.draw()

        lab_name = self.lab_combo.currentText()
        epochs = self.epoch_slider.value()
        lr = float(self.lr_combo.currentText())

        self.thread = TrainingThread(lab_name, epochs, lr)
        self.thread.log_signal.connect(self.log_text.append)
        self.thread.loss_signal.connect(self.update_loss_plot)
        self.thread.pred_signal.connect(self.update_pred_plot)
        self.thread.finished_signal.connect(self.training_finished)
        self.thread.start()

    def update_loss_plot(self, epoch, loss):
        self.epoch_history.append(epoch)
        self.loss_history.append(loss)
        self.ax_loss.clear()
        self.ax_loss.plot(self.epoch_history, self.loss_history, color='purple', linewidth=2)
        self.ax_loss.set_title(f"Loss: {loss:.4f}")
        self.ax_loss.grid(True, linestyle='--')
        self.canvas.draw()

    def update_pred_plot(self, x, y_true, y_pred):
        self.ax_pred.clear()
        self.ax_pred.plot(x, y_true, label='True Data', color='blue', alpha=0.5)
        self.ax_pred.plot(x, y_pred, label='AI Predict', color='red', linestyle='dashed')
        self.ax_pred.legend(loc='upper left')
        self.ax_pred.grid(True, linestyle='--')
        self.canvas.draw()

    # PRD 목표: 학습 종료 시 깃허브 업로드용 파일 자동 생성
    def training_finished(self, final_loss):
        self.run_btn.setEnabled(True)
        lab_name = self.lab_combo.currentText().split(":")[0]
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 1. 그래프를 outputs/ 폴더에 PNG로 자동 저장
        img_filename = os.path.join(OUTPUT_DIR, f"{lab_name.replace(' ', '_')}_{timestamp}.png")
        self.figure.savefig(img_filename)
        self.log_text.append(f"✅ 그래프 저장 완료: {img_filename}")
        
        # 2. 브레인스토밍 목표: 실험 결과를 마크다운 로그(테이블)로 누적 저장
        log_entry = f"| {timestamp} | {lab_name} | {self.epoch_slider.value()} | {self.lr_combo.currentText()} | {final_loss:.6f} | [Graph](./{img_filename}) |\n"
        
        if not os.path.exists(LOG_FILE):
            with open(LOG_FILE, 'w', encoding='utf-8') as f:
                f.write("# 물리 데이터 학습 실험 로그 (Auto-generated)\n\n")
                f.write("| 시간 | 실험명 | Epochs | Learning Rate | 최종 Loss (MSE) | 결과 이미지 |\n")
                f.write("|---|---|---|---|---|---|\n")
                
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            f.write(log_entry)
            
        self.log_text.append(f"✅ 마크다운 로그 업데이트 완료: {LOG_FILE}")
        self.log_text.append("🚀 이제 `git commit` 및 `push`를 진행할 수 있습니다.")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
