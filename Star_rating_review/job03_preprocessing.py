# 필요한 라이브러리 및 모듈 불러오기
import pickle  # 객체 저장 및 로드
import pandas as pd  # 데이터프레임 처리
import numpy as np  # 수치 연산
from sklearn.model_selection import train_test_split  # 데이터 분리
from keras.utils import to_categorical  # 원-핫 인코딩
from konlpy.tag import Okt, Kkma  # 형태소 분석
from tensorflow.keras.preprocessing.text import Tokenizer  # 텍스트 토큰화
from tensorflow.keras.preprocessing.sequence import pad_sequences  # 패딩 처리

# 데이터 전처리 과정

# CSV 파일 불러오기 및 중복 제거
df = pd.read_csv('C:/workspace/Star_rating_review/Star_rating_review/Star_All_Datas/All_Data.csv')
df.drop_duplicates(inplace=True)  # 중복 데이터 제거
df.reset_index(drop=True, inplace=True)  # 인덱스 재정렬

# 데이터 프레임 정보 확인
print(df.head())  # 상위 5개 데이터 확인
df.info()  # 데이터프레임 정보 확인
print(df.category.value_counts())  # 카테고리별 데이터 수 확인

# X: 뉴스 제목, Y: 뉴스 카테고리로 분리
X = df['titles'].tolist()  # 제목을 리스트로 변환
Y = df['category'].values  # 카테고리를 numpy 배열로 변환

print()
print('Y values:', np.unique(Y))  # 고유한 Y 값 확인

# 원-핫 인코딩
onehot_Y = to_categorical(Y)

# 형태소 분석
okt = Okt()
okt_x = okt.morphs(X[0], stem=True)  # 첫 번째 제목을 형태소 분석
print('Okt: ', okt_x)

# 형태소 분석을 전체 데이터에 적용
for i in range(len(X)):
    if (i % 1000) == 0:
        print(i)  # 진행 상황 출력
    X[i] = okt.morphs(X[i], stem=True)  # 형태소 분석 및 어간 추출

print('X: ', X[:5])  # 형태소 분석된 데이터 일부 확인

# 불용어 처리
stopwords = pd.read_csv('C:/workspace/Star_rating_review/Star_rating_review/stopwords_data/stopwords.csv', index_col=0)
print(stopwords)  # 불용어 리스트 확인

# 불용어 및 한 글자 단어 제거
for sentence in range(len(X)):
    words = []
    for word in range(len(X[sentence])):
        if len(X[sentence][word]) > 1:  # 두 글자 이상 단어만 사용
            if X[sentence][word] not in list(stopwords['stopword']):  # 불용어 제거
                words.append(X[sentence][word])
    X[sentence] = ' '.join(words)  # 단어들을 다시 문장으로 결합

print(X[:5])  # 전처리된 데이터 일부 확인

# 텍스트 데이터 숫자 라벨링
token = Tokenizer()
token.fit_on_texts(X)  # 텍스트 데이터를 토큰화
tokened_X = token.texts_to_sequences(X)  # 토큰화된 데이터를 숫자로 변환
wordsize = len(token.word_index) + 1  # 전체 단어 수 확인
print(wordsize)  # 단어 사전 크기 출력

# max 길이 조정 (최대 길이 129로 설정)
for i in range(len(tokened_X)):
    if len(tokened_X[i]) > 129:
        tokened_X[i] = tokened_X[i][:129]

X_pad = pad_sequences(tokened_X, 129)  # 패딩 적용
print(tokened_X[:5])  # 토큰화된 데이터 일부 확인

# 최대 길이 확인
max = 0
for i in range(len(tokened_X)):
    if max < len(tokened_X[i]):
        max = len(tokened_X[i])
print(max)  # 최대 길이 출력

# 토큰 저장
with open('./models/review_token_MAX_{}.pickle'.format(max), 'wb') as f:
    pickle.dump(token, f)  # 토크나이저 저장

X_pad = pad_sequences(tokened_X, max)  # 패딩 재적용
print(X_pad)  # 패딩된 데이터 확인
print(len(X_pad[0]))  # 패딩된 데이터의 길이 확인

# 학습 및 테스트 데이터 분리
X_train, X_test, Y_train, Y_test = train_test_split(X_pad, onehot_Y, test_size=0.1)  # 데이터 9:1로 분리
print(X_train.shape, Y_train.shape)  # 학습 데이터 형태 확인
print(X_test.shape, Y_test.shape)  # 테스트 데이터 형태 확인

# 데이터 저장
np.save('C:/workspace/Star_rating_review/Star_rating_review/crawling_data/review_data_X_train_max_{}_wordsize_{}'.format(max, wordsize), X_train)
np.save('C:/workspace/Star_rating_review/Star_rating_review/crawling_data/review_data_Y_train_max_{}_wordsize_{}'.format(max, wordsize), Y_train)
np.save('C:/workspace/Star_rating_review/Star_rating_review/crawling_data/review_data_X_test_max_{}_wordsize_{}'.format(max, wordsize), X_test)
np.save('C:/workspace/Star_rating_review/Star_rating_review/crawling_data/review_data_Y_test_max_{}_wordsize_{}'.format(max, wordsize), Y_test)
