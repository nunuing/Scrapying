import re
import urllib.request
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('seaborn-white')

import warnings
warnings.filterwarnings(action='ignore')


from konlpy.tag import Okt

okt = Okt()

train_df = pd.read_csv('에스쁘아 톤페어링 치크 9.6g + 톤페어링 하이라이터 9g .csv', encoding='utf-8')
# for i in range(0, len(train_df)) :
#     print(train_df['리뷰'][i])

#결측값 처리 -> 리뷰 있는 항목만 남기기
train_df = train_df[train_df['리뷰'].notnull()]
# for i in range(0, len(train_df)) :
#     print(train_df['리뷰'][i])

# 불가능한 문자 제거
# ‘ㄱ ~‘힣’까지의 문자를 제외한 나머지는 공백으로 치환, 영문: a-z| A-Z
train_df['리뷰'] = train_df['리뷰'].apply(lambda x : re.sub(r'[^ ㄱ-ㅣ가-힣]+', " ", x))

print(train_df.info())
review = train_df['리뷰']
for i in range(0, len(review)) :
    print(review[i])

# Train용 데이터셋과 Test용 데이터 셋 분리
# 1. 예측력을 높이기 위해 수집된 데이터를 학습용과 테스트 용으로 분리하여 진행
# 2. 보통 20~30%를 테스트용으로 분리해 두고 테스트

