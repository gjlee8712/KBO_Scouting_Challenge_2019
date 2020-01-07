# -*- coding: utf-8 -*-

import pandas as pd
import pickle 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression
import numpy as np

# 모델 학습
with open('c:/study/project2/pickle/kbo_label.csv','rb') as f:
    df = pickle.load(f)
df.info()

with open('c:/study/project2/pickle/base_train.csv','rb') as f:
    train = pickle.load(f)
train.info()

train = pd.merge(df.loc[:,['pitcher_name','label','KBOFIP']],train, on='pitcher_name')
train.info()
train = train.drop(['TBF','H','HR','BB','HBP','SO'], axis=1)
train.info()

with open('c:/study/project2/pickle/base_test.csv','rb') as f:
    test = pickle.load(f)

pitcher = test['pitcher_name']
test = test.drop('pitcher_name', axis=1)
test = test.drop(['TBF','H','HR','BB','HBP','SO'], axis=1)
test.info()

train.info()
x = train.iloc[:,3:17]
x.info()
y = train['label']
y.value_counts()
y1 = train['KBOFIP']
'''
from sklearn.preprocessing import scale
x = scale(x)
test = scale(test)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
y_train.value_counts()
y_test.value_counts()
'''
# Decision Tree
model1 = DecisionTreeClassifier(max_depth=5)
model1.fit(x,y)
model1.score(x,y)

f = model1.feature_importances_
r = pd.Series(f, index=train.iloc[:,3:18].columns)
s = r.sort_values(ascending=False)



plt.figure(figsize=(10,10))
plt.title('feature importance')
sns.barplot(x=s, y=s.index)

# random forest
model2 = RandomForestClassifier(n_estimators=100, oob_score=True)
model2.fit(x,y)
model2.score(x,y)

f = model2.feature_importances_
r = pd.Series(f, index=train.iloc[:,3:18].columns)
s = r.sort_values(ascending=False)

plt.figure(figsize=(10,10))
plt.title('feature importance')
sns.barplot(x=s, y=s.index)

#로지스틱
model3 = LogisticRegression()
model3.fit(x, y)
model3.score(x, y)

#KNN
model4 = KNeighborsClassifier(n_neighbors=3)
model4.fit(x, y)
model4.score(x, y)

#회귀분석 
model5 = LinearRegression()
model5.fit(x, y1)
model5.score(x, y1)

print('절편 :',model5.intercept_)
print('기울기 :',model5.coef_)

# test set 예측
pre1 = model1.predict(test)
pre2 = model2.predict(test)
pre3 = model3.predict(test)
pre4 = model4.predict(test)
pre5 = model5.predict(test)

result = pd.DataFrame({'pitcher':pitcher, 'DT':pre1, 'RF':pre2, 'LG':pre3, 'KNN':pre4, 'pre_FIP':pre5})
result

# 예측 결과 scoring
score = pd.DataFrame({'pitcher':result['pitcher']})
score 

score['DT'] = [2 if i == 'best' else 1 if i == 'good' else 0 for i in result['DT']]
score['RF'] = [2 if i == 'best' else 1 if i == 'good' else 0 for i in result['RF']]
score['LG'] = [2 if i == 'best' else 1 if i == 'good' else 0 for i in result['LG']]
score['KNN'] = [2 if i == 'best' else 1 if i == 'good' else 0 for i in result['KNN']]
score['ling'] = [2 if i <= 3 else 1 if i <= 8 else 0 for i in result['pre_FIP'].rank()]

score
score['sum'] = score.iloc[:,1:6].sum(axis = 1)
score
print(score)

# 결과 data 저장
np.savetxt('c:/study/project2/p2_result.csv',result,header=','.join(list(result.columns)), delimiter=',', fmt='%s', comments='')
np.savetxt('c:/study/project2/p2_score.csv',score,header=','.join(list(score.columns)), delimiter=',', fmt='%s', comments='')
