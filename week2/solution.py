import tensorflow as tf

def build_cnn_model():
    model = tf.keras.Sequential([
        # 첫 번째 합성곱 + 풀링
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        tf.keras.layers.MaxPooling2D((2, 2)),

        # 두 번째 합성곱 + 풀링
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),

        # 전결합 층
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    return model

# 과제 확인용 문구
if __name__ == "__main__":
    cnn_model = build_cnn_model()
    cnn_model.summary()
    print("\n[성공] MIT 6.S191 Lab 2 CNN 모델이 생성되었습니다.")
