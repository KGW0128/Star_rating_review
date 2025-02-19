import matplotlib.pyplot as plt
from keras.src.layers import Bidirectional
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, MaxPool1D, GRU, Dropout, Dense, BatchNormalization
import numpy as np
from keras.callbacks import EarlyStopping

# 모델 학습 CNN(이중분류기)

# 데이터 로드
X_train = np.load('C:/PyCharm_workspace/Star_rating_review/Star_rating_review/crawling_data/review_data_X_train_max_129_wordsize_15845.npy', allow_pickle=True)
X_test = np.load('C:/PyCharm_workspace/Star_rating_review/Star_rating_review/crawling_data/review_data_X_test_max_129_wordsize_15845.npy', allow_pickle=True)
Y_train = np.load('C:/PyCharm_workspace/Star_rating_review/Star_rating_review/crawling_data/review_data_Y_train_max_129_wordsize_15845.npy', allow_pickle=True)
Y_test = np.load('C:/PyCharm_workspace/Star_rating_review/Star_rating_review/crawling_data/review_data_Y_test_max_129_wordsize_15845.npy', allow_pickle=True)

# 원-핫 인코딩된 Y 데이터를 이진 레이블로 변환 (가장 큰 값을 가진 인덱스를 선택)
Y_train = np.argmax(Y_train, axis=1)
Y_test = np.argmax(Y_test, axis=1)

print(X_train.shape, Y_train.shape)  # 학습 데이터 크기 확인
print(X_test.shape, Y_test.shape)  # 테스트 데이터 크기 확인

# 모델 정의 (Bidirectional GRU와 Conv1D 층을 사용한 이진 분류기)

model = Sequential()

# 단어 임베딩: 단어를 벡터로 변환
model.add(Embedding(input_dim=15852, output_dim=300))

# 첫 번째 Conv1D 층: 특징 추출
model.add(Conv1D(32, kernel_size=3, padding='same', activation='relu'))
model.add(MaxPool1D(pool_size=2))  # MaxPool의 크기 증가
model.add(BatchNormalization())  # 배치 정규화 추가

# 두 번째 Conv1D 층: 특징 추출
model.add(Conv1D(64, kernel_size=3, padding='same', activation='relu'))
model.add(MaxPool1D(pool_size=2))  # MaxPool의 크기 증가
model.add(BatchNormalization())  # 배치 정규화 추가

# 세 번째 Conv1D 층: 특징 추출
model.add(Conv1D(128, kernel_size=3, padding='same', activation='relu'))
model.add(MaxPool1D(pool_size=2))  # MaxPool의 크기 증가
model.add(BatchNormalization())  # 배치 정규화 추가

# Bidirectional GRU 층: 순차적인 데이터를 처리하고, 양방향으로 학습
model.add(Bidirectional(GRU(256, activation='tanh', return_sequences=True)))
model.add(Dropout(0.4))  # 드롭아웃 비율 증가
model.add(BatchNormalization())  # 배치 정규화 추가

# 네 번째 Conv1D 층: 특징 추출
model.add(Conv1D(256, kernel_size=3, padding='same', activation='relu'))
model.add(MaxPool1D(pool_size=2))  # MaxPool의 크기 증가
model.add(BatchNormalization())  # 배치 정규화 추가

# 두 번째 Bidirectional GRU 층
model.add(Bidirectional(GRU(128, activation='tanh')))
model.add(Dropout(0.5))  # 드롭아웃 비율 증가

# 출력층: 이진 분류를 위한 Sigmoid 활성화 함수
model.add(Dense(1, activation='sigmoid'))

# 모델 입력 형태 정의 및 요약 출력
model.build(input_shape=(None, 129))
model.summary()

# 모델 학습 설정 (EarlyStopping을 사용하여 3번까지 성능 향상 없을 시 학습 종료)
early_stopping = EarlyStopping(monitor='val_accuracy', patience=3, restore_best_weights=True)

# 옵티마이저 학습률 조정
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 모델 학습
fit_hist = model.fit(X_train, Y_train, batch_size=128, epochs=50,
                     validation_data=(X_test, Y_test), callbacks=[early_stopping])

# 모델 평가 및 저장
score = model.evaluate(X_test, Y_test, verbose=0)
print('Test Loss:', score[0])
print('Test Accuracy:', score[1])

# 모델 저장
model.save('./models/review_model_{:.3f}.h5'.format(score[1]))

# 학습 결과 시각화

# 정확도 그래프
plt.plot(fit_hist.history['accuracy'], label='accuracy')
plt.plot(fit_hist.history['val_accuracy'], label='val_accuracy')
plt.legend()
plt.show()

# 손실 그래프
plt.plot(fit_hist.history['loss'], label='loss')
plt.plot(fit_hist.history['val_loss'], label='val_loss')
plt.legend()
plt.show()
