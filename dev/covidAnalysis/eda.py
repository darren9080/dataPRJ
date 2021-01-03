# https://www.kaggle.com/kimjihoo/coronavirusdataset

import pandas as pd
from matplotlib import pyplot as plt

# load data

train = pd.read_csv('/mnt/c/Users/darren_pc/PycharmProjects/testPRJ/data/titanic/train.csv',sep=',')

case = pd.read_csv('/mnt/c/Users/darren_pc/PycharmProjects/testPRJ/data/covid19kr/Case.csv',sep=',')
case.columns
patientInfo = pd.read_csv('/mnt/c/Users/darren_pc/PycharmProjects/testPRJ/data/covid19kr/PatientInfo.csv',sep=',')

# 정책
policy = pd.read_csv('/mnt/c/Users/darren_pc/PycharmProjects/testPRJ/data/covid19kr/Policy.csv',sep=',')
# 지역 정보
region = pd.read_csv('/mnt/c/Users/darren_pc/PycharmProjects/testPRJ/data/covid19kr/Region.csv',sep=',')

searchTrend = pd.read_csv('/mnt/c/Users/darren_pc/PycharmProjects/testPRJ/data/covid19kr/SearchTrend.csv',sep=',')
seoulFloating = pd.read_csv('/mnt/c/Users/darren_pc/PycharmProjects/testPRJ/data/covid19kr/SeoulFloating.csv',sep=',')


# time Series
time = pd.read_csv('/mnt/c/Users/darren_pc/PycharmProjects/testPRJ/data/covid19kr/Time.csv',sep=',')
timeAge = pd.read_csv('/mnt/c/Users/darren_pc/PycharmProjects/testPRJ/data/covid19kr/TimeAge.csv',sep=',')
timeGender = pd.read_csv('/mnt/c/Users/darren_pc/PycharmProjects/testPRJ/data/covid19kr/TimeGender.csv',sep=',')
timeProvince = pd.read_csv('/mnt/c/Users/darren_pc/PycharmProjects/testPRJ/data/covid19kr/TimeProvince.csv',sep=',')

weather = pd.read_csv('/mnt/c/Users/darren_pc/PycharmProjects/testPRJ/data/covid19kr/Weather.csv',sep=',')

## Feature Engineering

patientInfo.describe()
case.describe()
time.describe()