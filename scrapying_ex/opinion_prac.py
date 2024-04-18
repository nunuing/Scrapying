import pandas as pd

# warning 표시 안함
import warnings
warnings.filterwarnings(action='ignore')
from konlpy.tag import Okt

okt = Okt()

train_df = pd.read_csv('myproject/에스쁘아 톤페어링 치크 9.6g + 톤페어링 하이라이터 9g .csv', encoding='utf-8')

print(train_df)