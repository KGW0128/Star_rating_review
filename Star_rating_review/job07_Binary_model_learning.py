import matplotlib.pyplot as plt
from keras.src.layers import Bidirectional
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, MaxPool1D, GRU, Dropout, Dense,Flatten
import numpy as np
from keras.callbacks import EarlyStopping

# =============================
# 📊 1. 데이터 로드
# =============================

X_train = np.load('C:/workspace/Star_rating_review/Star_rating_review/crawling_data/review_data_X_train_max_129_wordsize_15852.npy', allow_pickle=True)
X_test = np.load('C:/workspace/Star_rating_review/Star_rating_review/crawling_data/review_data_X_test_max_129_wordsize_15852.npy', allow_pickle=True)
Y_train = np.load('C:/workspace/Star_rating_review/Star_rating_review/crawling_data/review_data_Y_train_max_129_wordsize_15852.npy', allow_pickle=True)
Y_test = np.load('C:/workspace/Star_rating_review/Star_rating_review/crawling_data/review_data_Y_test_max_129_wordsize_15852.npy', allow_pickle=True)

# 원-핫 인코딩 → 이진 레이블 변환
Y_train = np.argmax(Y_train, axis=1)
Y_test = np.argmax(Y_test, axis=1)

print(X_train.shape, Y_train.shape)  # 학습 데이터 크기 확인
print(X_test.shape, Y_test.shape)  # 테스트 데이터 크기 확인

# =============================
# 🛠️ 2. 이진 분류 모델 정의
# =============================

model = Sequential()

model.add(Embedding(input_dim=15852, output_dim=300))
model.add(Conv1D(32, kernel_size=3, padding='same', activation='relu'))
model.add(MaxPool1D(pool_size=1))
model.add(Bidirectional(GRU(256, activation='tanh', return_sequences=True)))
model.add(Dropout(0.3))
model.add(Conv1D(32, kernel_size=3, padding='same', activation='relu'))
model.add(MaxPool1D(pool_size=1))
model.add(Bidirectional(GRU(64, activation='tanh')))

model.add(Dense(1, activation='sigmoid'))

model.build(input_shape=(None, 129))
model.summary()

# =============================
# 🚦 3. 학습 설정 및 실행
# =============================

early_stopping = EarlyStopping(monitor='val_accuracy', patience=2, restore_best_weights=True)

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

fit_hist = model.fit(X_train, Y_train, batch_size=128, epochs=3,
                     validation_data=(X_test, Y_test), callbacks=[early_stopping])

# =============================
# 📊 4. 모델 평가 및 저장
# =============================

score = model.evaluate(X_test, Y_test, verbose=0)
print('Test Loss:', score[0])
print('Test Accuracy:', score[1])

model.save('./models/review_data_binary_classification_model_Bisigmoid_{:.3f}.h5'.format(score[1]))

# =============================
# 📈 5. 학습 결과 시각화
# =============================

plt.plot(fit_hist.history['accuracy'], label='accuracy')
plt.plot(fit_hist.history['val_accuracy'], label='val_accuracy')
plt.legend()
plt.show()

plt.plot(fit_hist.history['loss'], label='loss')
plt.plot(fit_hist.history['val_loss'], label='val_loss')
plt.legend()
plt.show()
