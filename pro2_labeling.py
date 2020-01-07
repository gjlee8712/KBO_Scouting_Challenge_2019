# -*- coding: utf-8 -*-

import pandas as pd
import pickle
from sklearn.preprocessing import scale
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

pd.set_option('display.max_columns', 30)
pd.set_option('display.width', 1000)
pd.set_option('display.max_rows', 200)

# =============================================================================
# 기존 KBO 선수들의 labeling
# =============================================================================
# feature 선정 : ERA, FIP, 재계약 여부, 이닝별 타석수
kyf_11 = pd.read_csv('c:/study/project2/data/kbo_yearly_foreigners_2011_2018.csv')
kyf_11.info()

# 이닝수 데이터 추가 [KBO 기록실에서 검색]
IP = [187, 194, 118, 179+1/3, 90, 167+2/3, 179+2/3, 175+2/3, 134+1/3, 137+1/3,
      49, 8, 173, 78+1/3,179+1/3, 184+2/3, 187+1/3, 178+1/3, 75+ 2/3, 37+ 2/3, 
      83, 171+ 2/3, 164+ 2/3, 151+ 1/3, 202 +2/3, 210,177 +1/3, 72 +2/3, 168+ 2/3,
      62, 63, 68 +1/3, 112+ 2/3, 143 +2/3, 152+ 2/3,155, 168, 180, 87+ 1/3,
      87, 144, 199, 112, 161+ 2/3,41+ 1/3, 187 +1/3, 74, 62, 147+ 1/3, 164 +2/3, 
      125, 194 +1/3, 199, 185 +1/3, 181+ 1/3,92+ 1/3, 167, 91+ 2/3, 171, 151+ 1/3,      
      46+ 2/3, 56, 26+ 1/3, 110, 8, 38+ 2/3, 85+ 1/3, 118, 71, 172+ 1/3, 152, 84,
      62, 63+ 1/3, 159 +2/3, 138 +1/3, 156+ 1/3, 46 +2/3, 99 +2/3, 176, 129+ 1/3,
      68 +2/3, 165, 39, 177 +1/3, 182, 160, 163+ 1/3, 178 +1/3, 172+ 2/3, 204, 140+ 2/3, 160+ 1/3, 79+ 2/3,
      74+ 2/3, 124+ 2/3, 50+ 2/3, 206+ 2/3, 201+ 2/3, 174, 149 +1/3, 101+ 2/3, 145 +1/3, 170, 66 +1/3]
kyf_11['IP'] = IP
kyf_11

# FIP 계산

# FIP = (13*HR + 3*(BB+HBP-IBB) -2*K)/IP + C
# C = lgERA – (((13*lgHR)+(3*(lgBB+lgHBP))-(2*lgK))/lgIP) 일반적으로 3.2
# HR 피홈런 BB 피볼넷구 HBP 피사구수 K 삼진수 IP 이닝수 IBB 고의사구 
HR = kyf_11['HR']
BB = kyf_11['BB']
HBP = kyf_11['HBP']
K = kyf_11['SO']
IP = kyf_11['IP']
ERA = kyf_11['ERA']
FIP = (13*HR + 3*(BB+HBP) -2*K)/IP + 3.2

kyf_11['FIP'] = FIP
kyf_11.info()

# 필요한 data만 남기고 정리
kyf_11 = kyf_11.drop(['H','HR','BB','HBP','SO','year_born'], axis=1)
kyf_11.info()

# kbo 활동 연수
p_pitcher = pd.DataFrame({'pitcher_name':kyf_11.pitcher_name.unique()})
years = kyf_11.groupby('pitcher_name')['year'].count()
p_pitcher = pd.merge(p_pitcher, years, left_on='pitcher_name',right_index=True)
p_pitcher.info()
p_pitcher = p_pitcher.rename({'year':'years'}, axis=1)
p_pitcher.info()
# 데이터 merge
kyf_11 = kyf_11.sort_values(by=['pitcher_name','year'], ascending = True)
temp = pd.DataFrame(columns = kyf_11.columns)
for i in p_pitcher['pitcher_name']:
    temp = temp.append(kyf_11[kyf_11['pitcher_name'] == i].iloc[0,:], ignore_index=True)
temp.info()
p_pitcher = pd.merge(p_pitcher, temp, on='pitcher_name')
p_pitcher.info()
p_pitcher['TBF'] = p_pitcher['TBF'].astype('int32')
# 데뷔 연도 이후 재계약 여부 (앞의 인덱스의 이름, 구단가 동일하고 연도가 다음해이면 1)
kyf_11.info()
re = []
for i in range(len(kyf_11)):
    if i == len(kyf_11)-1:
        re.append(0)
    else:
        if (kyf_11.iloc[i,0] == kyf_11.iloc[i+1,0]) & (kyf_11.iloc[i,2] == kyf_11.iloc[i+1,2]) & (kyf_11.iloc[i,1] +1 == kyf_11.iloc[i+1,1]):
            re.append(1)
        else:
            re.append(0)
re
len(re)
kyf_11['re'] = re
re = kyf_11.groupby('pitcher_name')['re'].max()
p_pitcher = pd.merge(p_pitcher, re, left_on='pitcher_name', right_index=True)
p_pitcher.info()

# 이닝 별 타석수 
p_pitcher['TBF/IP'] = p_pitcher['TBF']/p_pitcher['IP']
p_pitcher.info()
p_pitcher = p_pitcher.drop('TBF', axis=1)

#데이터 저장
with open('c:/study/project2/pickle/p_pitcher.csv','wb') as f:
    pickle.dump(p_pitcher, f)

#데이터 군집
with open('c:/study/project2/pickle/p_pitcher.csv','rb') as f:
    df = pickle.load(f)

df.info()
df_train = df.loc[:,['ERA','TBF/IP','FIP','re']]

model = KMeans(n_clusters=8, random_state=1)
model.fit(scale(df_train)) 
model.cluster_centers_
df['label'] = model.labels_
df

df = df.rename({'FIP':'KBOFIP'}, axis=1)
df.info()

df[df['label'] == 0] # 
df[df['label'] == 1] # best
df[df['label'] == 2] # 
df[df['label'] == 3] # good
df[df['label'] == 4] # 
df[df['label'] == 5] # 
df[df['label'] == 6] # 
df[df['label'] == 7] # good

plt.scatter(df['label'], df['ERA'])
plt.scatter(df['label'], df['TBF/IP'])
plt.scatter(df['label'], df['KBOFIP'])
plt.yticks(range(2,18,2),range(2,18,2))
plt.scatter(df['label'], df['re'])

df['label'] = ['best' if i == 1 else 'good' if (i == 3) or (i == 7) else 'bad' for i in df['label']]
df

#Label 작업 완료
with open('c:/study/project2/pickle/kbo_label.csv','wb') as f:
    pickle.dump(df, f)




