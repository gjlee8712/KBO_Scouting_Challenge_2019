# -*- coding: utf-8 -*-

import pandas as pd
import pickle

# test 셋 만들기
ff19 = pd.read_csv('c:/study/project2/data/fangraphs_foreigners_2019.csv')
ff19.info()
test = ff19.groupby('pitcher_name').mean()
test.info()
test = test.drop('year',axis =1)

#구종 수 뽑기
bsf_19 = pd.read_csv('c:/study/project2/data/baseball_savant_foreigners_2019.csv')
bsf_19.info()
bsf_19['pitch_name'].unique()
# nan, Unknown, Intentional Ball, Pitch Out, Fastball etc로 변경 후 제거
bsf_19.loc[bsf_19['pitch_name'].isnull(),'pitch_name']
bsf_19.loc[bsf_19['pitch_name'].isnull(),'pitch_name'] = 'etc'
bsf_19['pitch_name'].unique()
bsf_19.loc[bsf_19['pitch_name'].isin(['Unknown', 'Intentional Ball', 'Pitch Out', 'Fastball']),'pitch_name']
bsf_19.loc[bsf_19['pitch_name'].isin(['Unknown', 'Intentional Ball', 'Pitch Out', 'Fastball']),'pitch_name'] = 'etc'
bsf_19['pitch_name'].unique()
bsf_19 = bsf_19[bsf_19['pitch_name'] != 'etc']

pitch_uni = bsf_19.groupby('pitcher_name')['pitch_name'].unique()

sav_19 = pd.DataFrame()
for i in range(len(pitch_uni)):
    temp = pd.DataFrame({'pitcher_name':[pitch_uni.index[i]],'pitch_cnt':[len(pitch_uni[i])]})
    sav_19 = sav_19.append(temp, ignore_index=True)
sav_19

#최고 구속 뽑기
bsf_19.groupby(['pitcher_name','pitch_name'])['release_speed'].max()
bsf_19.info()
r_Mspeed = bsf_19.groupby('pitcher_name')['release_speed'].max()
sav_19 = pd.merge(sav_19, r_Mspeed, left_on='pitcher_name', right_index=True)
sav_19.info()
sav_19 = sav_19.rename({'release_speed':'Max_speed'}, axis=1)
sav_19.info()
#최저 구속 뽑기
bsf_19.groupby(['pitcher_name','pitch_name'])['release_speed'].min()
r_mspeed = bsf_19.groupby('pitcher_name')['release_speed'].min()
sav_19 = pd.merge(sav_19, r_mspeed, left_on='pitcher_name', right_index=True)
sav_19.info()
sav_19 = sav_19.rename({'release_speed':'Min_speed'}, axis=1)
sav_19.info()
sav_19

test = pd.merge(test, sav_19, left_index=True, right_on='pitcher_name')
test

with open('c:/study/project2/pickle/base_test.csv','wb') as f:
    pickle.dump(test, f)

