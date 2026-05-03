import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import os

# 0. 환경 설정
output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'outputs')
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

print("=== SciPy를 이용한 선형 회귀 (curve_fit) ===")

# 1. 데이터 준비 (01_linear_regression_spring.py와 동일)
weights = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=float)
true_lengths = 2 * weights + 10
np.random.seed(42)
noise = np.random.normal(loc=0.0, scale=1.5, size=len(weights))
measured_lengths = true_lengths + noise

# 2. 모델 함수 정의
def linear_func(x, a, b):
    return a * x + b

# 3. 최적화 (curve_fit: 최소자승법 기반 즉시 계산)
popt, pcov = curve_fit(linear_func, weights, measured_lengths)
learned_a, learned_b = popt

print(f"\n[학습 결과 (SciPy)]")
print(f"예측된 식: 길이 = {learned_a:.2f} * 무게 + {learned_b:.2f}")
print(f"실제 식  : 길이 = 2.00 * 무게 + 10.00")

new_weight = 15.0
print(f"\n[예측 테스트]")
print(f"15kg 예측 길이: {linear_func(new_weight, learned_a, learned_b):.2f} cm")

# 4. 시각화
plt.figure(figsize=(10, 6))
plt.scatter(weights, measured_lengths, color='blue', label='Measured Data (Noisy)')
plt.plot(weights, true_lengths, 'g--', label='True Law (y=2x+10)')

x_range = np.linspace(0, 15, 100)
plt.plot(x_range, linear_func(x_range, learned_a, learned_b), 'r-', label='SciPy Fit')

plt.title("Hooke's Law Regression (SciPy curve_fit)")
plt.xlabel('Weight (kg)')
plt.ylabel('Spring Length (cm)')
plt.legend()
plt.grid(True)

save_path = os.path.join(output_dir, 'ex_01_spring_scipy.png')
plt.savefig(save_path)
print(f"\n그래프가 저장되었습니다: {save_path}")
