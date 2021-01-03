import requests
import pandas as pd
import io


nasdaq_url="https://pkgstore.datahub.io/core/nasdaq-listings/nasdaq-listed_csv/data/7665719fb51081ba0bd834fde71ce822/nasdaq-listed_csv.csv"
s = requests.get(nasdaq_url).content
companies = pd.read_csv(io.StringIO(s.decode('utf-8')))

companies['Market Category'].value_counts()

'''
q
s
g
'''


