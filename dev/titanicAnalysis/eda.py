# titanic
# https://www.kaggle.com/morihosseini/comprehensive-exploratory-data-analysis-of-titanic


import pandas as pd
import os
import re

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

train = pd.read_csv('/mnt/c/Users/darren_pc/PycharmProjects/testPRJ/data/titanic/train.csv',sep=',')
test = pd.read_csv('/mnt/c/Users/darren_pc/PycharmProjects/testPRJ/data/titanic/test.csv',sep=',')
gender = pd.read_csv('/mnt/c/Users/darren_pc/PycharmProjects/testPRJ/data/titanic/gender_submission.csv',sep=',')

train.shape
test.shape

gender.columns

datasets = [train,test]
train.info()
test.info()

# predict survived
train.columns


# feature engineering
## Title
train.groupby(['Pclass','Survived']).size()

def get_title(name):
    reg_exp = re.search(' ([A-Za-z]+)\.', name)
    return reg_exp.group(1) if reg_exp else ""

for ds in datasets:
    title = ds['Name'].apply(get_title)
    print(ds.groupby(title).size(),'\n')

for ds in datasets:
    ds['Title'] = ds['Name'].apply(get_title)
    ds['Title'] = ds['Title'].replace(['Mlle','Ms'], 'Miss').replace('Mme','Mrs')
    rare = ['Capt','Col','Countess','Don','Dona','Dr','Jonkheer','Lady','Major','Rev','Sir']
    ds['Title'] = ds['Title'].replace(rare,'Rare')


print('Train\n',train.groupby('Title').size(),'\n')
print('Test\n',test.groupby('Title').size(),'\n')

## Sex
train.groupby(['Sex','Survived']).size()


## Age
train.groupby(['Age','Survived']).size()


ds['Age'].isnull().sum()
SEED = 7

### replace Nan with Random ages between mean-std, mean+std
for ds in datasets:
    def rand_ages():
        np.random.seed(SEED)
        return np.random.randint(low = ds['Age'].mean()- ds['Age'].std(),
                                 high= ds['Age'].mean() + ds['Age'].std(),
                                 size= ds['Age'].isnull().sum())

    ds.loc[ds['Age'].isnull(),'Age'] = rand_ages()


    ### binning age groups
    ds['Age'] = pd.cut(ds['Age'],bins = [-np.inf, 14,24,64, np.inf], labels=range(4))
    ds.loc[:,'Age'] = ds['Age'].astype(int)

train.groupby(['Age','Survived']).size()

### sibSp = Siblngs/Spouses
'''
number of siblings/spouses aboard the Titanic and can be lconsidered along with "Parch"
'''

train.columns


### Parch = Parent + Children


for ds in datasets:
    ds['FamilySize']= ds['SibSp']+ds['Parch'] + 1


train.groupby(['FamilySize','Survived']).size()

for ds in datasets:
    ds['isAlone'] = 0
    ds.loc[ds['FamilySize'] ==1, 'isAlone'] = 1

ds['isAlone']

train.groupby(['isAlone','Survived']).size()

## Ticket Number - insignificant

## Fare

median = train['Fare'].median()

for ds in datasets:
    ds['Fare'] =ds['Fare'].fillna(median)

    ds['Fare'] = pd.qcut(ds['Fare'], q=4, labels=range(4))
    ds.loc[:,'Fare'] = ds['Fare'].astype(int)

train.groupby(['Fare','Survived']).size()


## Cabin

for ds in datasets:
    ds['hadCabin'] = ds['Cabin'].notnull().astype(int)

train.groupby(['hadCabin','Survived']).size()

print("Train\n", train.groupby('Embarked').size(),'\n')
print("Test\n", test.groupby('Embarked').size())

for ds in datasets:
    ds["Embarked"] = ds['Embarked'].fillna('S')

train.groupby(['Embarked','Survived']).size()



## Encoding
def encode_freq_sorted(feature):
    sorted_indices = feature.value_counts().index
    sorted_dict = dict(zip(sorted_indices,range(len(sorted_indices))))
    return feature.map(sorted_dict).astype(int)

for ds in datasets:
    ds['Sex'] = encode_freq_sorted(ds['Sex'])
    ds['Embarked'] = encode_freq_sorted(ds['Embarked'])
    ds['Title'] = encode_freq_sorted(ds['Title'])

train.head()


## Feature Selection

train.columns


drop_features = ['Name','SibSp','Parch','Ticket','Cabin']

train = train.drop(columns=drop_features)
test = test.drop(columns=drop_features)

X = train.drop(columns=['PassengerId','Survived'])
y = train['Survived']


X.head()



## Data Exploration

plt.figure(figsize=(11,9))
sns.heatmap(train.corr(),annot=True, cmap='Blues')

plt.show()
# Classification

# from sklearn.model_selection import train_test_split


from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(X, y, train_size=0.8,random_state=SEED)
X_test = test.drop(columns=['PassengerId']) # For consistency in naming variables

# random forest

from sklearn.ensemble import RandomForestClassifier

# By setting random_state, we get the same result every time we run the command
random_forest = RandomForestClassifier(random_state=SEED)

random_forest.get_params()


def random_search(X, y, estimator, params, score="accuracy", cv=3,
                  n_iter=100, random_state=SEED, n_jobs=-1):
    """
    Randomized search of parameters, using "cv" fold cross validation, search
    across "n_iter" different combinations, and use all available cores
    """
    print("# Tuning hyper-parameters for {} by randomized search".format(score))

    clf = RandomizedSearchCV(estimator=estimator, param_distributions=params,
                             scoring=score, cv=cv, n_iter=n_iter, n_jobs=n_jobs,
                             random_state=random_state)
    clf.fit(X, y)

    print("Best parameters by random search:\n", clf.best_params_)
    return clf

# Note: running this cell takes a few minutes

# Create the parameter grid
params_random = {
    # Number of trees in random forest
    'n_estimators': [int(x) for x in np.linspace(200, 2000, num = 10)],
    # Minimum number of samples required to split a node
    'min_samples_split': [2, 5, 10],
    # Minimum number of samples required at each leaf node
    'min_samples_leaf': [1, 2, 4],
    # Number of features to consider at every split
    'max_features': ['auto', 'sqrt'],
    # Maximum number of levels in tree
    'max_depth': [int(x) for x in np.linspace(10, 110, num = 11)],
    # Method of selecting samples for training each tree
    'bootstrap': [True, False]
}

random_forest_random = random_search(X_train, y_train, estimator=random_forest,
                                      params=params_random)