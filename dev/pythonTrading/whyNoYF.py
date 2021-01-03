'''
증권 데이터 분석을 위해서는 종목별로 OHLS 데이터를 구해야한다
OHLS :

'''

# yahoo finance 데이터의 문제점
from pandas_datareader import data as pdr
import yfinance as yf
yf.pdr_override()
import matplotlib.pyplot as plt

df =pdr.get_data_yahoo('005930.KS','2017-01-01')

plt.figure(figsize=(9,6))
plt.subplot(2,1,1)
plt.title('Samsung Electronics (Yahoo Finance)')
plt.plot(df.index,df['Close'],'c',label='Close')
plt.plot(df.index,df['Adj Close'],'b--',label='Adj Close')

plt.legend(loc="best")
plt.subplot(2,1,2)
plt.bar(df.index,df['Volume'],color ='g', label='Volume')
plt.legend(loc="best")
plt.show()

'''
1. missing data (2017년 10월 3주간 데이터가 비어있다) 
2. 액면 분할 후, 수정 종가가 정확하지 않다.
삼성전자 기준*
=> naver finance에서 데이터를 수집하여 데이터베이스에 구축한뒤 활용한다.
'''
