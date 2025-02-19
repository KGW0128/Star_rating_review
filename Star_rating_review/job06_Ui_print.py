import sys
from PyQt5.QtWidgets import *  # PyQt5 위젯 모듈 임포트
from PyQt5 import uic  # UI 파일을 불러오기 위한 uic 임포트
from PyQt5.QtCore import QCoreApplication  # 앱 종료를 위한 모듈
from keras.models import load_model  # Keras 모델 불러오기
import numpy as np  # 수치 연산 라이브러리
import pickle  # 객체 직렬화 및 역직렬화
from tensorflow.keras.preprocessing.text import Tokenizer  # 텍스트 토크나이저
from tensorflow.keras.preprocessing.sequence import pad_sequences  # 시퀀스 패딩

# UI 파일 불러오기
form_window = uic.loadUiType('C:/PyCharm_workspace/Star_rating_review/review_ui.ui')[0]

# 토크나이저 모델 불러오기
with open('C:/PyCharm_workspace/Star_rating_review/Star_rating_review/models/review_token_MAX_129.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

# 메인 클래스 정의
class Exam(QWidget, form_window):
    def __init__(self):
        super().__init__()
        self.setupUi(self)  # UI 세팅

        # 학습된 모델 불러오기
        self.model = load_model(
            'C:/PyCharm_workspace/Star_rating_review/Star_rating_review/models/review_data_classfication_model_0.9109051823616028.h5')

        # 버튼 클릭 이벤트 연결
        self.review_btn.clicked.connect(self.btn_clicked_slot)
        self.text_edit.clear()  # 텍스트 에디터 초기화
        self.stars_progressBar.setValue(0)  # 프로그레스바 초기화
        self.AI_star_print_label.setText('')  # AI 별점 라벨 초기화
        self.my_star_label.setText('')  # 사용자 별점 라벨 초기화

    def preprocess_text(self, text):
        """리뷰 텍스트를 전처리하여 모델 입력 형태로 변환"""
        sequences = tokenizer.texts_to_sequences([text])  # 텍스트를 시퀀스로 변환
        padded = pad_sequences(sequences, maxlen=129)  # 패딩 적용
        return padded

    def btn_clicked_slot(self):
        try:
            # 사용자 입력 텍스트 및 별점 가져오기
            review_text = self.text_edit.toPlainText()
            review_star = self.my_star_spinBox.value()
            print("입력된 리뷰:", review_text)
            print("내가 준 별점:", review_star)

            # 리뷰 텍스트 전처리
            review_text_vector = self.preprocess_text(review_text)
            print("리뷰 벡터화 결과:", review_text_vector)

            # AI 모델 예측
            pred = self.model.predict(review_text_vector).item()
            print("AI 예측 확률:", pred)

            # 예측 확률을 별점으로 변환
            if pred <= 0.2:
                predicted_star = 1
                label = "매우 부정적"
            elif pred <= 0.4:
                predicted_star = 2
                label = "부정적"
            elif pred <= 0.6:
                predicted_star = 3
                label = "중립적"
            elif pred <= 0.8:
                predicted_star = 4
                label = "긍정적"
            else:
                predicted_star = 5
                label = "매우 긍정적"

            print("AI 예측 별점:", predicted_star)

            # AI 별점 출력
            ai_stars = "★" * predicted_star + "☆" * (5 - predicted_star)
            self.AI_star_print_label.setText(ai_stars)

            # 예측 확률을 프로그레스바에 반영
            self.stars_progressBar.setValue(abs(int(pred * 100)))

            # 내가 준 별점 출력
            my_stars = "★" * review_star + "☆" * (5 - review_star)
            self.my_star_label.setText(my_stars)

        except Exception as e:
            print(f"예측 중 오류 발생: {e}")
            QMessageBox.warning(self, "오류", "예측 중 오류가 발생했습니다.")  # 오류 메시지 출력

# 프로그램 진입점
if __name__ == '__main__':
    try:
        app = QApplication(sys.argv)  # 애플리케이션 객체 생성
        mainWindow = Exam()  # 메인 윈도우 객체 생성
        mainWindow.show()  # 윈도우 표시
        sys.exit(app.exec_())  # 앱 실행 및 종료 처리
    except Exception as e:
        print(f"오류 발생: {e}")
    finally:
        print("프로그램 종료")  # 프로그램 종료 메시지 출력
