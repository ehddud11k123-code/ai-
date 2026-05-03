import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import os

# 0. 환경 설정
output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'outputs')
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

print("=== SciPy를 이용한 최적화 (minimize / BFGS) ===")

# 1. 목적 함수: y = (x-2)^2 + 1, 최소값 x=2, y=1
def objective_func(x):
    return (x - 2)**2 + 1

# 2. 최적화
x0 = -3.0
print(f"시작 위치: x = {x0}")

history = []
def callback(x):
    history.append(x[0])

result = minimize(objective_func, x0, method='BFGS', callback=callback)

print(f"\n[최적화 결과]")
print(f"성공 여부: {result.success}")
print(f"최적의 x: {result.x[0]:.4f}")
print(f"최소값 y: {result.fun:.4f}")
print(f"반복 횟수: {result.nit}")

# 3. 시각화
plt.figure(figsize=(10, 6))

x_range = np.linspace(-4, 6, 100)
plt.plot(x_range, objective_func(x_range), 'k-', label='Objective Function y=(x-2)^2 + 1')

path_x = np.array([x0] + history)
path_y = np.array([objective_func(x) for x in path_x])

plt.scatter(path_x, path_y, color='red', s=100, zorder=5)
plt.plot(path_x, path_y, 'r--', label='Optimization Path (BFGS)')
plt.text(path_x[0],  path_y[0]  + 1, 'Start', ha='center', color='red', fontweight='bold')
plt.text(path_x[-1], path_y[-1] + 1, 'End',   ha='center', color='red', fontweight='bold')

plt.title('Optimization using SciPy minimize (BFGS)')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True)

save_path = os.path.join(output_dir, 'ex_04_optimization_scipy.png')
plt.savefig(save_path)
print(f"\n그래프가 저장되었습니다: {save_path}")
