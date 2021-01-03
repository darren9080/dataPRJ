import pandas as pd
import re
import os


# 폴더안에 있는 국가별 파일 읽어와서

data = pd.read_csv(r'/mnt/c/Users/darren_pc/PycharmProjects/testPRJ/data/pr/PR_totalMentions_Thailand.rawdata.txt',sep = '\t')
data.columns



# find /comment and set ['Associated Cases'] = comment
# find /forum, forum.

forum_str = ['forum.','/forum']
answer_str = ['answer/']

dropTarget = ['forum.','/forum/','/community/','/comment/','blog.','/blog/','/comment/','answers.','/answer']


data['Permalink'].loc[data['Permalink'].str.contains('|'.join(dropTarget))]


