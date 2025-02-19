import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
from konlpy.tag import Okt
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


# 학습된 모델 예측해보기



# -------------------------------
# 📝 **1. 데이터 로드 및 전처리**
# -------------------------------

# 리뷰 데이터 로드
df = pd.read_csv('C:/PyCharm_workspace/Star_rating_review/test/_Four_star_고등어.csv')
df.drop_duplicates(inplace=True)  # 중복 데이터 제거
df.reset_index(drop=True, inplace=True)  # 인덱스 재설정

# 이진 분류 라벨 생성 (Five, Four => 'positive' / 그 외 => 'negative')
df['binary_category'] = df['category'].apply(lambda x: 'positive' if x in ['Five', 'Four'] else 'negative')

# X: 리뷰 제목, Y: 이진 카테고리
X = df['titles']
Y = df['binary_category']

# 라벨 인코딩 (positive -> 1, negative -> 0)
encoder = LabelEncoder()
labeled_y = encoder.fit_transform(Y)
onehot_Y = to_categorical(labeled_y)  # 원-핫 인코딩
print("이진 라벨:", encoder.classes_)

# -------------------------------
# 📝 **2. 형태소 분석 및 불용어 처리**
# -------------------------------

okt = Okt()
stopwords = pd.read_csv('C:/PyCharm_workspace/Star_rating_review/Star_rating_review/stopwords_data/stopwords.csv', index_col=0)

# 리뷰 데이터를 형태소 분석 후 불용어 제거
X_processed = X.copy()
for i in range(len(X_processed)):
    if i % 100 == 0:
        print('형태소 처리:', i)
    processed_text = okt.morphs(X_processed[i], stem=True)  # 형태소 분석
    X_processed[i] = ' '.join([word for word in processed_text if word not in stopwords['stopword'].values and len(word) > 1])

print("전처리 결과 확인:", X_processed[:5])

# -------------------------------
# 📝 **3. 토큰화 및 패딩**
# -------------------------------

# 미리 저장된 토크나이저 로드
with open('C:/PyCharm_workspace/Star_rating_review/Star_rating_review/models/review_token_MAX_129.pickle', 'rb') as f:
    token = pickle.load(f)

# 리뷰 텍스트를 시퀀스로 변환 후 패딩
tokened_X = token.texts_to_sequences(X)
X_pad = pad_sequences(tokened_X, maxlen=100, padding='post', truncating='post')  # 패딩 및 자르기
print("패딩 확인:", X_pad[:5])

# -------------------------------
# 📝 **4. 모델 로드 및 예측**
# -------------------------------

# 저장된 이진 분류 모델 로드
binary_model = load_model('C:/PyCharm_workspace/Star_rating_review/Star_rating_review/models/review_model_0.911.h5')

# 예측 수행 (0~1 사이 값 출력)
binary_preds = binary_model.predict(X_pad)

# -------------------------------
# 📝 **5. 예측 결과 비교**
# -------------------------------

# 0.5를 기준으로 이진 분류 예측 (0.5보다 크면 'positive', 작으면 'negative')
binary_predicts = ['positive' if pred > 0.5 else 'negative' for pred in binary_preds]
df['binary_predict'] = binary_predicts  # 예측 결과 저장

# -------------------------------
# 📝 **6. 정확도 및 성능 평가**
# -------------------------------

# 이진 분류 성능 평가 (정확도, 정밀도, 재현율, F1 스코어)
binary_accuracy = accuracy_score(df['binary_category'], df['binary_predict'])
binary_precision = precision_score(df['binary_category'], df['binary_predict'], pos_label='positive')
binary_recall = recall_score(df['binary_category'], df['binary_predict'], pos_label='positive')
binary_f1 = f1_score(df['binary_category'], df['binary_predict'], pos_label='positive')

print("이진 분류 정확도:", binary_accuracy)
print("이진 분류 Precision:", binary_precision)
print("이진 분류 Recall:", binary_recall)
print("이진 분류 F1 Score:", binary_f1)

# 혼돈 행렬 출력
print("혼돈 행렬:")
print(confusion_matrix(df['binary_category'], df['binary_predict']))
