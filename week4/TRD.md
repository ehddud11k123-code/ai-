# TRD (Technical Requirements Document)
# AI Physics Course — GitHub Linked Simulator


---

## 1. 기술 스택

| 구분 | 라이브러리 | 버전 | 용도 |
|---|---|---|---|
| GUI | PySide6 | 최신 | 메인 윈도우, 슬라이더, 버튼 |
| 딥러닝 | TensorFlow / Keras | 2.x | 신경망 정의 및 학습 |
| 수치 연산 | NumPy | 최신 | 데이터 생성, 셔플링 |
| 시각화 | Matplotlib | 최신 | Loss 곡선, 예측 피팅 그래프 |
| Qt-Matplotlib 연동 | matplotlib.backends.backend_qtagg | — | PySide6 내 캔버스 임베딩 |

---

## 2. 아키텍처

```
MainWindow (QMainWindow)
├── 컨트롤 패널 (QVBoxLayout)
│   ├── QComboBox      — Lab 선택
│   ├── QSlider        — Epochs (10~500)
│   ├── QComboBox      — Learning Rate
│   ├── QPushButton    — 실행 버튼
│   └── QTextEdit      — 터미널 로그 (읽기 전용)
└── 그래프 영역
    └── FigureCanvas (Matplotlib Figure)
        ├── ax_loss    — 실시간 MSE Loss 곡선
        └── ax_pred    — 예측 모델 피팅

TrainingThread (QThread)          ← 학습 전담 백그라운드 스레드
├── run_lab1()
├── run_lab3()
└── Signal 방출
    ├── log_signal(str)
    ├── loss_signal(epoch, loss)
    ├── pred_signal(x, y_true, y_pred)
    └── finished_signal(final_loss)

RealtimeCallback (keras.callbacks.Callback)
└── on_epoch_end() → loss_signal, pred_signal 방출 (5 epoch마다)
```

---

## 3. 핵심 구현 세부사항

### 3.1 데이터 셔플링

순차 데이터를 그대로 `validation_split`에 넣으면 뒷부분 데이터만 검증셋이 되어 편향 발생.
반드시 `np.random.permutation`으로 사전 셔플링한다.

```python
indices = np.random.permutation(len(x))
x_shuffled, y_shuffled = x[indices], y[indices]
```

### 3.2 Lab별 모델 아키텍처

| Lab | 레이어 구성 | 활성화 함수 | 목적 |
|---|---|---|---|
| Lab 1 | Dense(64) × 2 + Dense(1) | tanh / linear | 사인 함수 근사 |
| Lab 3 | Dense(128) → Dense(64) → Dense(32) → Dense(1) | relu / linear | 과적합 유도 (깊은 망 + 노이즈 데이터) |

### 3.3 실시간 UI 업데이트 설계

- `TrainingThread`는 `QThread` 상속 — 학습이 UI 스레드를 블로킹하지 않음
- Signal/Slot 메커니즘으로 스레드 간 안전하게 데이터 전달
- `RealtimeCallback.on_epoch_end()`에서 5 epoch마다 `y_pred` 계산 후 `pred_signal` 방출
  - 매 epoch 방출 시 Matplotlib `canvas.draw()` 호출 비용이 누적되어 UI 느려짐 방지

### 3.4 GitHub 아티팩트 저장

```python
# PNG 저장
img_path = f"outputs/{lab}_{timestamp}.png"
figure.savefig(img_path)

# 마크다운 로그 누적
log_entry = f"| {timestamp} | {lab} | {epochs} | {lr} | {final_loss:.6f} | [Graph](./{img_path}) |\n"
with open("experiment_history.md", "a", encoding="utf-8") as f:
    f.write(log_entry)
```

`experiment_history.md`가 없을 경우 헤더 포함해서 새로 생성, 있을 경우 append.

---

## 4. Lab 4 (진자 운동) 구현 명세 — 예정

RK4(Runge-Kutta 4차) 수치 적분으로 진자 운동 데이터를 생성한다.

```python
def rk4_pendulum(theta0, omega0, dt, steps, g=9.8, L=1.0):
    def deriv(state):
        theta, omega = state
        return np.array([omega, -(g/L) * np.sin(theta)])

    states = [np.array([theta0, omega0])]
    for _ in range(steps - 1):
        s = states[-1]
        k1 = deriv(s)
        k2 = deriv(s + 0.5*dt*k1)
        k3 = deriv(s + 0.5*dt*k2)
        k4 = deriv(s + dt*k3)
        states.append(s + (dt/6)*(k1 + 2*k2 + 2*k3 + k4))
    return np.array(states)
```

입력: 시간 `t`, 출력: 각도 `θ(t)` — 이 데이터로 신경망 학습.

---

## 5. 파일 구조

```
week4/
├── main.py                  # 메인 애플리케이션 (GUI + 학습 스레드)
├── PRD.md                   # 제품 요구사항 문서
├── TRD.md                   # 기술 요구사항 문서
├── experiment_history.md    # 실험 로그 (자동 생성)
└── outputs/                 # PNG 그래프 (자동 생성)
    └── Lab_N_YYYYMMDD_HHMMSS.png
```
