import pandas as pd  # 데이터프레임 처리를 위한 라이브러리
import os  # 파일 및 디렉터리 경로를 다루기 위한 라이브러리


#파일 일괄 병합 코드


# 병합할 CSV 파일이 있는 폴더 경로 설정
folder_path = 'C:/PyCharm_workspace/Star_rating_review/test'

# 폴더 내 파일 확인
try:
    files_in_folder = os.listdir(folder_path)  # 폴더 내 파일 목록 가져오기
    print(f"폴더 내 파일: {files_in_folder}")
except FileNotFoundError:  # 폴더가 존재하지 않을 경우 예외 처리
    print(f"폴더 경로가 올바르지 않습니다: {folder_path}")

# CSV 파일 필터링
csv_files = [file for file in files_in_folder if file.endswith('.csv')]  # CSV 파일만 필터링
print(f"CSV 파일 목록: {csv_files}")

# CSV 파일 병합
if csv_files:
    # 모든 CSV 파일을 읽어와 병합
    dataframes = [pd.read_csv(os.path.join(folder_path, file)) for file in csv_files]
    merged_df = pd.concat(dataframes, ignore_index=True)  # 데이터프레임 병합

    # 병합된 결과를 저장
    output_path = os.path.join(folder_path, "all_Coupang_Review.csv")  # 병합된 CSV 저장 경로 설정
    merged_df.to_csv(output_path, index=False)  # CSV 파일로 저장
    print(f"병합된 파일이 저장되었습니다: {output_path}")
else:
    print("CSV 파일이 폴더에 없습니다.")

# 병합된 CSV 파일 확인
df = pd.read_csv('C:/PyCharm_workspace/Star_rating_review/test/all_Coupang_Review.csv')  # 병합된 CSV 파일 불러오기

# 중복 데이터 제거 및 정보 출력
df.drop_duplicates(inplace=True)  # 중복 데이터 제거
print(df.head())  # 데이터프레임 상위 5개 행 출력
df.info()  # 데이터프레임 정보 출력
print(df.category.value_counts())  # 카테고리별 데이터 수 출력
