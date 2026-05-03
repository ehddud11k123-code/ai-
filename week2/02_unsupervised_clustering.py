import numpy as np
import matplotlib.pyplot as plt
import os

# 0. 환경 설정
output_dir = 'outputs'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

print("=== 비지도 학습: 군집화 (K-Means Clustering) ===")

# 1. 데이터 생성 (정답 없음)
np.random.seed(42)
group1 = np.random.normal(loc=[2, 2], scale=0.5, size=(30, 2))
group2 = np.random.normal(loc=[8, 3], scale=0.5, size=(30, 2))
group3 = np.random.normal(loc=[5, 8], scale=0.5, size=(30, 2))
X = np.vstack([group1, group2, group3])

print(f"데이터 개수: {len(X)}개")

# 2. K-Means 구현
k = 3
centers = X[np.random.choice(len(X), k, replace=False)]

for i in range(10):
    distances = np.sqrt(((X - centers[:, np.newaxis])**2).sum(axis=2))
    closest_cluster = np.argmin(distances, axis=0)
    new_centers = np.array([X[closest_cluster == j].mean(axis=0) for j in range(k)])
    if np.all(centers == new_centers):
        break
    centers = new_centers

print("\n[학습 완료된 중심점]")
print(centers)

# 3. 시각화
plt.figure(figsize=(8, 6))
colors = ['red', 'green', 'blue']
for j in range(k):
    cluster_data = X[closest_cluster == j]
    plt.scatter(cluster_data[:, 0], cluster_data[:, 1], c=colors[j], label=f'Group {j+1}', alpha=0.6)
plt.scatter(centers[:, 0], centers[:, 1], c='black', marker='*', s=300, label='Centroids')

plt.title('Unsupervised Learning: K-Means Clustering')
plt.xlabel('Feature 1 (e.g., Purchase Amount)')
plt.ylabel('Feature 2 (e.g., Visit Count)')
plt.legend()
plt.grid(True)

save_path = os.path.join(output_dir, '02_clustering.png')
plt.savefig(save_path)
print(f"\n그래프가 저장되었습니다: {save_path}")
