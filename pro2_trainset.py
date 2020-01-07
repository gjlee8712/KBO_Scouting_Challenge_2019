# -*- coding: utf-8 -*-

import pandas as pd
import pickle

# 트레이닝 데이터 과거 메이저 데이터 
with open('c:/study/project2/pickle/kbo_label.csv','rb') as f:
    df = pickle.load(f)

ff11 = pd.read_csv('c:/study/project2/data/fangraphs_foreigners_2011_2018.csv')
df_set = set(df['pitcher_name']) #62명
ff_set = set(ff11['pitcher_name']) #60명
len(df_set)
len(ff_set)
df_set - ff_set  # {'마리몬', '밴덴헐크', '브리검'}
ff_set - df_set  # {'벨레스터'}

ff11.info()
ff11 = ff11.sort_values(by=['pitcher_name','year'])
df.info()

#kbo 데뷔년도 이전 데이터만 / 없는 투수 정리
remove_idx = []
for i in range(len(df)):
    for j in range(len(ff11)):
        if (df.iloc[i,0] == ff11.iloc[j,0]) & (df.iloc[i,2] <= ff11.iloc[j,1]):
            remove_idx.append(j)
remove_idx

ff11.iloc[remove_idx,:]
len(remove_idx)
ff11
ff11 = ff11.drop(remove_idx, axis=0)
train = ff11.groupby('pitcher_name').mean()
train.info()
train = train.drop('year',axis =1)
train.info()

# savant 분석 
bsf_11 = pd.read_csv('c:/study/project2/data/baseball_savant_foreigners_2011_2018.csv')
bsf_11.info()

#구종 수 뽑기
bsf_11['pitch_name'].unique()
# nan, Unknown, Intentional Ball, Pitch Out, Fastball etc로 변경 후 제거
bsf_11.loc[bsf_11['pitch_name'].isnull(),'pitch_name']
bsf_11.loc[bsf_11['pitch_name'].isnull(),'pitch_name'] = 'etc'
bsf_11['pitch_name'].unique()
bsf_11.loc[bsf_11['pitch_name'].isin(['Unknown', 'Intentional Ball', 'Pitch Out', 'Fastball']),'pitch_name']
bsf_11.loc[bsf_11['pitch_name'].isin(['Unknown', 'Intentional Ball', 'Pitch Out', 'Fastball']),'pitch_name'] = 'etc'
bsf_11['pitch_name'].unique()
bsf_11 = bsf_11[bsf_11['pitch_name'] != 'etc']

pitch_uni = bsf_11.groupby('pitcher_name')['pitch_name'].unique()

sav_11 = pd.DataFrame()
for i in range(len(pitch_uni)):
    temp = pd.DataFrame({'pitcher_name':[pitch_uni.index[i]],'pitch_cnt':[len(pitch_uni[i])]})
    sav_11 = sav_11.append(temp, ignore_index=True)
sav_11

#최고 구속 뽑기
bsf_11.groupby(['pitcher_name','pitch_name'])['release_speed'].max()
bsf_11.info()
r_Mspeed = bsf_11.groupby('pitcher_name')['release_speed'].max()
sav_11 = pd.merge(sav_11, r_Mspeed, left_on='pitcher_name', right_index=True)
sav_11.info()
sav_11 = sav_11.rename({'release_speed':'Max_speed'}, axis=1)
sav_11.info()
#최저 구속 뽑기
bsf_11.groupby(['pitcher_name','pitch_name'])['release_speed'].min()
r_mspeed = bsf_11.groupby('pitcher_name')['release_speed'].min()
sav_11 = pd.merge(sav_11, r_mspeed, left_on='pitcher_name', right_index=True)
sav_11.info()
sav_11 = sav_11.rename({'release_speed':'Min_speed'}, axis=1)
sav_11.info()
sav_11

#train set merge
train = pd.merge(train, sav_11, left_index=True, right_on='pitcher_name')
train.info()
train
with open('c:/study/project2/pickle/base_train.csv','wb') as f:
    pickle.dump(train, f)

