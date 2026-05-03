import numpy as np
import matplotlib.pyplot as plt
import os

# 0. 환경 설정
output_dir = 'outputs'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

print("=== 최적화: 경사 하강법 시각화 (Gradient Descent) ===")

# 1. 손실 함수 및 기울기 정의
def loss_function(x):
    return x**2          # y = x², 최소값: x=0

def gradient(x):
    return 2 * x         # dy/dx = 2x

# 2. 경사 하강법 시뮬레이션
current_x = -4.0
learning_rate = 0.1
steps = []

print(f"시작 위치: x = {current_x}")

for i in range(20):
    current_loss = loss_function(current_x)
    steps.append((current_x, current_loss))
    grad = gradient(current_x)
    current_x = current_x - learning_rate * grad
    print(f"Step {i+1}: x = {current_x:.4f}, Loss = {current_loss:.4f}")

print(f"\n최종 위치: x = {current_x:.4f} (목표값 0.0에 매우 가까움)")

# 3. 시각화
plt.figure(figsize=(10, 6))

x_range = np.linspace(-5, 5, 100)
plt.plot(x_range, loss_function(x_range), 'k-', label='Loss Function (y=x^2)')

steps_arr = np.array(steps)
plt.scatter(steps_arr[:, 0], steps_arr[:, 1], color='red', s=100, zorder=5)
plt.plot(steps_arr[:, 0], steps_arr[:, 1], 'r--', label='Gradient Descent Path')

plt.text(steps_arr[0, 0],  steps_arr[0, 1]  + 1, 'Start', ha='center', color='red', fontweight='bold')
plt.text(steps_arr[-1, 0], steps_arr[-1, 1] + 1, 'End',   ha='center', color='red', fontweight='bold')

plt.title('Optimization: Gradient Descent (Ball rolling down)')
plt.xlabel('Parameter x')
plt.ylabel('Loss y')
plt.legend()
plt.grid(True)

save_path = os.path.join(output_dir, '04_gradient_descent.png')
plt.savefig(save_path)
print(f"\n그래프가 저장되었습니다: {save_path}")
