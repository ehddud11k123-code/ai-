import numpy as np
import matplotlib.pyplot as plt
import os

# 0. 환경 설정
output_dir = 'outputs'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

print("=== 데이터 전처리: Min-Max 정규화 (Normalization) ===")

# 1. 데이터 생성 (단위가 극단적으로 다른 두 특성)
np.random.seed(42)
n_samples = 50
salary = np.random.uniform(30000000, 100000000, n_samples)  # 연봉 (3천만~1억)
age    = np.random.uniform(20, 60, n_samples)                # 나이 (20~60세)

# 2. Min-Max 정규화: (x - min) / (max - min)
salary_normalized = (salary - salary.min()) / (salary.max() - salary.min())
age_normalized    = (age    - age.min())    / (age.max()    - age.min())

print("\n[데이터 비교]")
print(f"연봉(원본): 최소 {salary.min():,.0f}, 최대 {salary.max():,.0f}")
print(f"연봉(변환): 최소 {salary_normalized.min():.1f}, 최대 {salary_normalized.max():.1f}")
print(f"나이(원본): 최소 {age.min():.0f}, 최대 {age.max():.0f}")
print(f"나이(변환): 최소 {age_normalized.min():.1f}, 최대 {age_normalized.max():.1f}")

# 3. 시각화
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(age, salary, color='orange')
plt.title('Before Scaling (Raw Data)')
plt.xlabel('Age (Years)')
plt.ylabel('Salary (Won)')
plt.grid(True)

plt.subplot(1, 2, 2)
plt.scatter(age_normalized, salary_normalized, color='blue')
plt.title('After Scaling (Normalized)')
plt.xlabel('Age (0~1)')
plt.ylabel('Salary (0~1)')
plt.grid(True)
plt.axis('square')

save_path = os.path.join(output_dir, '03_preprocessing.png')
plt.savefig(save_path)
print(f"\n그래프가 저장되었습니다: {save_path}")
