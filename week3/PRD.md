# PRD (Product Requirements Document)
# AI Physics Course — Week 3 Neural Network Visualizers

---

## 1. 제품 개요

### 1.1 목적
신경망의 핵심 개념(퍼셉트론, 활성화 함수, 순전파, 역전파, 범용 근사)을 **인터랙티브 GUI**로 직접 조작하며 학습하는 교육용 시뮬레이터 모음.

### 1.2 핵심 가치
| 가치 | 설명 |
|---|---|
| 즉각적 시각 피드백 | 슬라이더·버튼 조작 즉시 그래프 반영 |
| 개념별 독립 실행 | 각 Lab이 독립 스크립트 — 하나만 실행해도 동작 |
| 수식과 시각의 연결 | 수식 계산 결과를 그래프로 동시 표시 |

---

## 2. Lab 목록 및 기능 요구사항

### 과제 1 — Visual Perceptron Simulator
**파일:** `과제1번.py`

| 기능 | 설명 |
|---|---|
| 문제 선택 | OR / AND / NAND / XOR Gate 콤보박스 전환 |
| 수동 파라미터 조절 | Weight1, Weight2, Bias 슬라이더로 결정 경계 실시간 조작 |
| 자동 학습 | "Run 10 Epochs" 버튼으로 퍼셉트론 학습 규칙 자동 실행 |
| 결정 영역 시각화 | 0영역(연빨강) / 1영역(연파랑) 배경색으로 분류 경계 표시 |
| XOR 한계 표시 | XOR 선택 시 선형 분리 불가 상황 시각적 확인 가능 |

---

### Lab 2 — Activation Functions Visualizer
**파일:** `2번.py`

| 기능 | 설명 |
|---|---|
| 4분할 그래프 | Sigmoid / Tanh / ReLU / Leaky ReLU 동시 표시 |
| X축 범위 조절 | 슬라이더로 입력 범위 동적 변경 |
| Leaky ReLU alpha 조절 | alpha 슬라이더로 음수 기울기 실시간 변경 |
| 도함수 오버레이 | 각 함수의 미분값을 같은 그래프에 표시 |

---

### Lab 3 — Forward Propagation Visualizer (2-3-1)
**파일:** `3번.py`

| 기능 | 설명 |
|---|---|
| 2-3-1 구조 순전파 | 입력 2개 → 은닉층 3개 → 출력 1개 계산 과정 시각화 |
| 가중치 슬라이더 | 각 연결 가중치를 슬라이더로 직접 조절 |
| 수식 패널 | 각 노드의 가중합·활성화 결과를 수식과 함께 표시 |

---

### Lab 4 — Multi-Layer Perceptron XOR Solver
**파일:** `4번.py`

| 기능 | 설명 |
|---|---|
| XOR 문제 해결 | 순수 NumPy 역전파로 2-4-1 MLP 학습 |
| 학습 과정 시각화 | Epoch별 Loss 감소 곡선 실시간 플롯 |
| 결정 경계 변화 | 학습 진행에 따라 결정 경계가 변하는 애니메이션 |

---

### Lab 5 — Universal Approximation Theorem Visualizer
**파일:** `5번.py`

| 기능 | 설명 |
|---|---|
| 임의 함수 근사 | 최소제곱법(Pseudo-inverse) 기반 신경망으로 함수 근사 |
| 은닉 노드 수 조절 | 노드 수 변경 시 근사 정밀도 변화 실시간 확인 |
| 범용 근사 정리 시연 | 노드가 충분하면 임의 함수를 근사할 수 있음을 직관적으로 시연 |

---

## 3. 공통 비기능 요구사항

| 항목 | 요구사항 |
|---|---|
| 독립 실행 | 각 파일을 `python 파일명.py`로 단독 실행 가능 |
| UI 반응성 | 슬라이더 조작 시 즉각 렌더링 (블로킹 없음) |
| 플랫폼 | PySide6 + Matplotlib — Windows/macOS/Linux 지원 |
