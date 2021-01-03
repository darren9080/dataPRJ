import requests

req = requests.get('https://beomi.github.io/beomi.github.io_old/')

## HTML 소스 가져오기
html = req.text
## HTTP Header 가져오기
header = req.headers
## HTTP Status 가져오기 (200: 정상)
status = req.status_code
## HTTP가 정상적으로 되었는지 (True/False)
is_ok = req.ok

from bs4 import BeautifulSoup

req = requests.get('https://beomi.github.io/beomi.github.io_old/')
## HTML 소스 가져오기
html = req.text
## BeautifulSoup으로 html소스를 python객체로 변환하기
## 첫 인자는 html소스코드, 두 번째 인자는 어떤 parser를 이용할지 명시.
## 이 글에서는 Python 내장 html.parser를 이용했다.
soup = BeautifulSoup(html, 'html.parser')



'''
CES Exhibitor Directory Crawl
'''

targetLink = 'https://digital.ces.tech/exhibitors'



# paginate fucntion
