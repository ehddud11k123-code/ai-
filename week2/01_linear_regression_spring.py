import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os

# 0. 환경 설정
output_dir = 'outputs'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

print(f"TensorFlow Version: {tf.__version__}")

# 1. 데이터 준비
# 훅의 법칙: Length = 2 * Weight + 10
weights = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=float)
true_lengths = 2 * weights + 10

np.random.seed(42)
noise = np.random.normal(loc=0.0, scale=1.5, size=len(weights))
measured_lengths = true_lengths + noise

print("\n[데이터 확인]")
print("무게(kg):", weights)
print("측정된 길이(cm):", np.round(measured_lengths, 2))

# 2. 모델 구성
model = tf.keras.Sequential([
    tf.keras.layers.Dense(units=1, input_shape=[1])
])

# 3. 모델 컴파일
model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.01),
              loss='mean_squared_error')

# 4. 모델 학습
print("\n[학습 시작]")
history = model.fit(weights, measured_lengths, epochs=500, verbose=0)
print("학습 완료!")

# 5. 결과 확인
learned_w = float(model.layers[0].get_weights()[0][0])
learned_b = float(model.layers[0].get_weights()[1][0])

print(f"\n[학습 결과]")
print(f"예측된 식: 길이 = {learned_w:.2f} * 무게 + {learned_b:.2f}")
print(f"실제 식  : 길이 = 2.00 * 무게 + 10.00")

new_weight = 15.0
predicted_length = float(model.predict(np.array([[new_weight]]), verbose=0)[0][0])
print(f"\n[예측 테스트]")
print(f"15kg 추를 매달았을 때 예측 길이: {predicted_length:.2f} cm")
print(f"이론상 실제 길이: {2 * new_weight + 10:.2f} cm")

# 6. 시각화
plt.figure(figsize=(10, 6))
plt.scatter(weights, measured_lengths, color='blue', label='Measured Data (Noisy)')
plt.plot(weights, true_lengths, 'g--', label='True Law (y=2x+10)')

plot_weights = np.linspace(0, 15, 100)
plot_lengths = model.predict(plot_weights.reshape(-1, 1), verbose=0)
plt.plot(plot_weights, plot_lengths, 'r-', label='AI Prediction')

plt.title("Hooke's Law Regression (Spring Experiment)")
plt.xlabel('Weight (kg)')
plt.ylabel('Spring Length (cm)')
plt.legend()
plt.grid(True)

save_path = os.path.join(output_dir, 'spring_fitting.png')
plt.savefig(save_path)
print(f"\n그래프가 저장되었습니다: {save_path}")
