import requests
import urllib.request
from bs4 import BeautifulSoup

# # # 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36'
# # url = "https://brand.naver.com/espoir/products/10123826918"
# # req = Request(url,headers={'User-Agent':'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36'})
# # webpage = urlopen(req)
# # soup = BeautifulSoup(webpage, 'html5lib')

# # # objects = soup.select("._2L3vDiadT9")
# # # print(len(objects))

headers = {'User-Agent' : 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36'}
url = "https://brand.naver.com/espoir/products/10123826918"
response = requests.get(url, headers=headers)
response.raise_for_status()

soup = BeautifulSoup(response.text, 'html.parser')
print(soup.find('title').string)
tags = soup.select('div', class_='_2sO8BtgrbT')

# tags = soup.select(".review_cont")
# print(len(tags))
# # tags = soup.select(attrs = {'class':'_2L3vDiadT9'})
# # for t in tags :
# #     print(t.text)
# # tags = soup.select('span')

# tags = soup.select("#review")
# sections = soup.select('#review')
# #sections = soup.select('body > div > div > div > div > section')
# for section in sections:
#        section = section.name
#        print (section)
#REVIEW > div > div._2LvIMaBiIO > div._2g7PKvqCKe > ul > li:nth-child(2) > div > div > div > div._3-1uaKhzq4 > div > div > div._3z6gI4oI6l > div > span

for t in tags :
    print(t.text)

