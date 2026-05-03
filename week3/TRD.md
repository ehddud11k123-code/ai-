# TRD (Technical Requirements Document)
# AI Physics Course — Week 3 Neural Network Visualizers

---

## 1. 기술 스택

| 구분 | 라이브러리 | 용도 |
|---|---|---|
| GUI | PySide6 | 윈도우, 슬라이더, 버튼, 레이아웃 |
| 시각화 | Matplotlib | 그래프 렌더링 |
| Qt-Matplotlib 연동 | FigureCanvasQTAgg | Matplotlib Figure를 Qt 위젯으로 임베딩 |
| 수치 연산 | NumPy | 행렬 연산, 활성화 함수, 역전파 |

---

## 2. 공통 아키텍처 패턴

모든 Lab이 동일한 3-레이어 구조를 따른다:

```
[수학/모델 클래스]      — 순수 NumPy 연산, UI 의존성 없음
        ↓  Signal/Slot 또는 직접 호출
[VisualizerCanvas]     — FigureCanvasQTAgg 상속, Matplotlib Figure 보유
        ↓
[MainWindow]           — QMainWindow, 슬라이더·버튼 이벤트 → 캔버스 갱신
```

---

## 3. Lab별 핵심 구현

### 과제 1 — Perceptron

```python
class PerceptronEngine:
    def predict(self, x):
        return 1 if np.dot(self.w, x) + self.b >= 0 else 0

    def update(self, x, target):
        error = target - self.predict(x)
        self.w += self.lr * error * x   # 퍼셉트론 학습 규칙
        self.b += self.lr * error
        return error
```

- 결정 경계: `w1*x1 + w2*x2 + b = 0` 직선을 `np.meshgrid`로 전체 공간에 걸쳐 색칠
- XOR는 선형 분리 불가 → 학습이 수렴하지 않음을 시각적으로 확인

---

### Lab 2 — Activation Functions

```python
class ActivationMath:
    @staticmethod
    def sigmoid(x):       return 1 / (1 + np.exp(-x))
    def sigmoid_deriv(x): s = sigmoid(x); return s * (1 - s)

    @staticmethod
    def leaky_relu(x, alpha=0.1):
        return np.where(x > 0, x, alpha * x)
```

- 2×2 subplot 구성: 각 활성화 함수와 도함수를 같은 축에 오버레이
- `alpha` 슬라이더 변경 시 `leaky_relu` 재계산 후 `canvas.draw()`

---

### Lab 3 — Forward Propagation (2-3-1)

- 가중치 행렬: `W1 (3×2)`, `W2 (1×3)` — 슬라이더로 각 원소 직접 조절
- 순전파 수식:
  ```
  z1 = W1 @ x + b1
  a1 = tanh(z1)
  z2 = W2 @ a1 + b2
  output = sigmoid(z2)
  ```
- 각 노드 값을 화살표 + 텍스트로 그래프에 오버레이 표시

---

### Lab 4 — MLP with Backpropagation (XOR)

```python
class MLP:
    # 순수 NumPy 역전파, 2-4-1 구조
    def forward(self, x):
        self.z1 = self.W1 @ x + self.b1
        self.a1 = sigmoid(self.z1)
        self.z2 = self.W2 @ self.a1 + self.b2
        return sigmoid(self.z2)

    def backward(self, x, y):
        # MSE 기반 역전파
        delta2 = (self.output - y) * sigmoid_deriv(self.z2)
        delta1 = (self.W2.T @ delta2) * sigmoid_deriv(self.z1)
        self.W2 -= lr * delta2 @ self.a1.T
        self.W1 -= lr * delta1 @ x.T
```

---

### Lab 5 — Universal Approximation (Pseudo-inverse)

- 은닉층 활성화 행렬 `H` 계산 후 `W_out = pinv(H) @ y_target`
- `np.linalg.pinv` 사용 — 경사하강법 없이 즉시 최적 가중치 계산
- 노드 수 슬라이더 변경 시 H 재계산 → 근사 곡선 즉시 업데이트

---

## 4. 파일 구조

```
week3/
├── 과제1번.py    — Perceptron Simulator (OR/AND/NAND/XOR)
├── 2번.py        — Activation Functions Visualizer
├── 3번.py        — Forward Propagation Visualizer (2-3-1)
├── 4번.py        — MLP XOR Solver (역전파, NumPy)
├── 5번.py        — Universal Approximation Visualizer
├── PRD.md        — 제품 요구사항 문서
└── TRD.md        — 기술 요구사항 문서
```
