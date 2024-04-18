import pandas as pd

# warning 표시 안함
import warnings
warnings.filterwarnings(action='ignore')
from konlpy.tag import Okt

okt = Okt()

train_df = pd.read_csv('myproject/에스쁘아 톤페어링 치크 9.6g + 톤페어링 하이라이터 9g .csv', encoding='utf-8')

print(train_df.info())

# 한글 외 문자 제거(옵션)
# ‘ㄱ ~‘힣’까지의 문자를 제외한 나머지는 공백으로 치환, 영문: a-z| A-Z
import re # 정규식을 사용하기 위해 re 모듈을 임포트
train_df['리뷰'] = train_df['리뷰'].apply(lambda x : re.sub(r'[^ ㄱ-ㅣ가-힣]+', " ", x))

review = train_df['리뷰']
print(review)

from sklearn.model_selection import train_test_split
train_set, test_set = train_test_split(review, test_size=0.2, random_state=0)
print(len(train_set), len(test_set))