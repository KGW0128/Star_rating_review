from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.chrome.options import Options as ChromeOptions
from setuptools.package_index import user_agent
from webdriver_manager.chrome import ChromeDriverManager
from selenium.common.exceptions import NoSuchElementException
from selenium.common.exceptions import StaleElementReferenceException
import pandas as pd
import re
import time
import datetime



#쿠팡 리뷰 크롤링 코드


# 크롬에서 연다
# 열어볼 주소
options = ChromeOptions()
user_agent = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36'

# 한글만 긁어옴
options.add_argument('--disable-blink-features=AutomationControlled')
options.add_argument('user_agent=' + user_agent)
options.add_argument('lang=ko_KR')

service = ChromeService(executable_path=ChromeDriverManager().install())
driver = webdriver.Chrome(service=service, options=options)


# 별점 카테고리 리스트 정의
category = ['Five','Four','Three','Two','One']



# 1~5점까지 반복
for star_i in range(4, 6):

    # 데이터 프레임 초기화
    df_titles = pd.DataFrame()

    url = ('https://www.coupang.com/vp/products/6357346236?itemId=13404238888&vendorItemId=80658954948&q=%EC%A3%BC%EA%BE%B8%EB%AF%B8&itemsCount=36&searchId=0bab7a2aaefe4e0e95475f873875ef35&rank=0&searchRank=0&isAddedCart=')

    driver.get(url)  # 브라우저 띄우기
    time.sleep(3)  # 버튼 생성이 될때까지 기다리는 딜레이

    # 리뷰 버튼
    Review_button_xpath = '//*[@id="btfTab"]/ul[1]/li[2]'
    time.sleep(2)
    driver.find_element(By.XPATH, Review_button_xpath).click()

    # 별점 보기 버튼
    Star_button_xpath = '// *[ @ id = "btfTab"] / ul[2] / li[2] / div / div[6] / section[2] / div[3]'
    time.sleep(2)
    driver.find_element(By.XPATH, Star_button_xpath).click()


    #각 별점별 리뷰 갯수 확인
    element = driver.find_element('xpath', '// *[ @ id = "btfTab"] / ul[2] / li[2] / div / div[6] / section[2] / div[3] / div[2] / ul / li[{}] / div[3]'.format(star_i))
    text = element.text
    print('리뷰갯수: ',text)

    #받아온 리뷰갯수 페이지 수만큼 나누기
    pass_num = int(text.replace(',', ''))/50

    if pass_num>30:
        pass_num = 30

    print('넘길 페이지: ', int(pass_num))


    # 별점 보기 안에 별갯수 버튼
    Star_in_button_xpath = ' // *[ @ id = "btfTab"] / ul[2] / li[2] / div / div[6] / section[2] / div[3] / div[2] / ul / li[{}]'.format(star_i)
    time.sleep(0.5)
    driver.find_element(By.XPATH, Star_in_button_xpath).click()
    time.sleep(2)




    titles = []
    #전체 페이지 갯수(옆으로 넘기기 버튼 횟수)
    for next_page_i in range(int(pass_num)):

        if next_page_i!=0:
            next_page_xpath = '//*[@id="btfTab"]/ul[2]/li[2]/div/div[6]/section[4]/div[3]/button[12]'
            time.sleep(2)
            driver.find_element(By.XPATH, next_page_xpath).click()

        print('next_page_i: ', next_page_i,'page')

        time.sleep(1)
        #1~10버튼
        for page_num_i in range(2,12):

            print('page_num_i: ', page_num_i-1)
            page_xpath = '//*[@id="btfTab"]/ul[2]/li[2]/div/div[6]/section[4]/div[3]/button[{}]'.format(page_num_i)
            time.sleep(3)
            driver.find_element(By.XPATH, page_xpath).click()

            for text_i in range(1, 5):
                
                #쿠팡이 막아놓은 주소 변동 뚫기
                for article_index in [3, 4]:
                    title_xpath = '//*[@id="btfTab"]/ul[2]/li[2]/div/div[6]/section[4]/article[{}]/div[{}]/div'.format(text_i, article_index)

                    try:
                        # 뷰 크롤링 및 한글 외 문자 제거
                        title = driver.find_element(By.XPATH, title_xpath).text
                        title = re.compile('[^가-힣 ]').sub(' ', title)  # 한글과 공백만 남김
                        title = re.sub(' +', ' ', title).strip()  # 여러 공백을 하나로 줄이고 양 끝 공백 제거

                        # title이 비어있지 않을 때만 추가
                        if title:
                            if title and all(exclude not in title for exclude in ["명에게 도움 됨", "신선도 적당해요", "신선도 아주 신선해요", "생각보다 덜 신선해요"]):
                                titles.append(title)
                                print('text저장:', titles)
                            else:
                                print('pass (trash):', text_i, title)
                        else:
                            print('pass (Null):', text_i)

                    except:  # 예외 처리 (존재하지 않는 항목 무시)
                        print('pass:', text_i)

                # 크롤링된 제목을 데이터프레임에 저장
                df_section_titles = pd.DataFrame(titles, columns=['titles'])
                df_section_titles['category'] = category[star_i - 1]
                df_titles = pd.concat([df_titles, df_section_titles], axis='rows', ignore_index=True)
                titles.clear()


    # 카테고리별 데이터프레임 정보 출력
    print(df_titles.head())
    df_titles.info()
    print(df_titles['category'].value_counts())

    # 제목 리스트 초기화
    titles.clear()

    # 카테고리별 데이터를 CSV 파일로 저장
    df_titles.to_csv('C:/PyCharm_workspace/Star_rating_review/test/깻잎_{}_star.csv'.format(category[star_i-1]), index=False)


    time.sleep(1)