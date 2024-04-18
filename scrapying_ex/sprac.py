import time
import requests
from time import sleep
import re, requests, csv
from bs4 import BeautifulSoup as bs
from selenium.webdriver.common.keys import Keys
import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By
import math

keys = Keys()

url = "https://brand.naver.com/espoir/products/10123826918"
driver = webdriver.Chrome()
driver.get(url)

page = requests.get(url)
count = 0
# 리뷰 갯수 알기 위해 페이지 로드 -> 리뷰탭이 활성화 되어 있어야하기 때문에 
time.sleep(3)

html = driver.page_source
soup = bs(html, "html.parser")
title = soup.find('title').text.split(',')[0]
stop = int(soup.find('strong', class_='_2pgHN-ntx6').text)
stop = math.ceil((stop) / 20)

# 기존 수기로 입력받던 코드
# stop = math.ceil(int(input("전체 리뷰 수를 입력해주세요")) / 20) + 1
next_btn = []
for i in range(2, stop + 2) :
    next_btn.append('a:nth-child(' + str(i) + ')')
    if stop >= 9 and i % 10 == 0 :
        next_btn.append('a.fAUKm1ewwo._2Ar8-aEUTq')

print(next_btn)


# for i in range(0, len(review)) :
#     temp = review[i].text
#     if len(temp) <= 10 :
#         continue
#     review_list.append(temp)

# for i in range(0, len(review_list) ) :
#     print(str(i) + " : " + review_list[i])
driver.find_element(By.CSS_SELECTOR, '#REVIEW').click()      #리뷰창 클릭
time.sleep(3)
review_list = []

for pagenum in next_btn[0 : stop] :
    print(pagenum)
    driver.find_element(By.CSS_SELECTOR, '#REVIEW > div > div._2LvIMaBiIO > div._2g7PKvqCKe > div > div >' +str(pagenum)).click()       #리뷰 클릭창
    html = driver.page_source
    soup = bs(html, "html.parser")
    review = soup.find_all('span', class_='_2L3vDiadT9')
    time.sleep(2)
    for i in range(0, len(review)) :
        temp = review[i].text
        if len(temp) <= 10 :
            continue
        review_list.append(temp)

df = pd.DataFrame({"리뷰" : review_list})
df.to_csv(title + ".csv", encoding='utf-8')
print("저장완료")
