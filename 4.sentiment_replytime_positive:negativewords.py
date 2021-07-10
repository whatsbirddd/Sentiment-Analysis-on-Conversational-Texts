
import os
import sys
import re

import numpy as np
import pandas as pd

import json

import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
sns.set(style='white', context='notebook', palette='deep')

from collections import Counter
from pororo import Pororo 

data = pd.read_csv("전처리한 데이터 파일 넣어주세요!")

data.dropna(inplace=True)
data.reset_index(drop=True,inplace=True)

sa = Pororo(task="sentiment", model="brainbert.base.ko.nsmc", lang="ko") #Pororo 감성분석 모델
df = data[data["Date"].str[:7]=='2021-06'].reset_index(drop=True) #2021-6월 대화만

user = df.User.unique() 
df_me = df[df.User == user[0]].reset_index(drop=True) #유저1
df_you = df[df.User == user[1]].reset_index(drop=True) #유저2

#평균 답장시간
from datetime import datetime
def reply_time(df): #평균답장시간 구하는 함수
  diff = []
  for i in range(len(df)-1):
    before = datetime.strptime(df["Date"][i],"%Y-%m-%d %H:%M:%S")
    after = datetime.strptime(df["Date"][i+1],"%Y-%m-%d %H:%M:%S")

    diff_ = after-before
    diff.append(diff_.seconds) #시간차이를 'seconds'단위로 저장
  mean_diff = np.mean(diff) #시간 차이 평균값
  hour = mean_diff/3600
  min = (mean_diff %3600)/60
  return hour, min

hour, min = reply_time(df_me) #유저1의 평균 답장시간. 정수부분만 사용 -> ex. 0시간 7분 , 1시간 12분
hour_you, min_you = reply_time(df_you) #유저2의 평균 답장시간. 정수부분만 사용

def make_sentiment(data): #감성분석 결과를 새로운 데이터프레임으로 다시 만들어줌
    df_sentiment = pd.DataFrame(columns = ["positive","negative"]) #1100개 -> 1분27초
    for i in range(len(data)):
        temp = sa(data["preprocessed"][i],show_probs=True)
        df_sentiment=df_sentiment.append(temp,ignore_index=True) 
    df = pd.concat([data,df_sentiment],axis=1)
    return df

df_me = make_sentiment(df_me) #유저1의 감성분석 결과
df_you = make_sentiment(df_you) #유저2의 감성분석 결과

def pos_neg(df): #긍정, 부정 결과만 데이터프레임을 새로 만들어줌
  df_pos = df[df["positive"]>0.90]
  df_neg = df[df["negative"]>0.90]
  return df_pos, df_neg
df_me_pos, df_me_neg = pos_neg(df_me) #유저1의 긍정, 부정 데이터프레임
df_you_pos, df_you_neg = pos_neg(df_you) #유저2의 긍정, 부정 데이터프레임

#나와 상대방의 애정도 퍼센트
me_pos_prop=len(df_me_pos)/len(df_me) #유저1의 긍정텍스트 퍼센트 -> 이 값을 사용하면 됩니다!!
me_neg_prop=len(df_me_neg)/len(df_me) #유저1의 부정텍스트 퍼센트 -> 이 값을 사용하면 됩니다!!

you_pos_prop=len(df_you_pos)/len(df_you) #유저2의 긍정텍스트 퍼센트 -> 이 값을 사용하면 됩니다!!
you_neg_prop=len(df_you_neg)/len(df_you) #유저2의 부정텍스트 퍼센트 -> 이 값을 사용하면 됩니다!!


#기분좋을때/ 안좋을 때
pos = ["NNG","NNP","MAG","IC"]#pos : 명사, 동사, 형용사, 부사
verse = ["VV","VA"]#VV, VA -> 원형으로 복원

stopwords_verse = ["하다","같다","있다","되다"]

#동사,형용사 원형 복원, 명사, 부사 추출
from khaiii import KhaiiiApi 
api = KhaiiiApi() 
def Morphology_analysis(sentence):
  morphs =[]
  for word in api.analyze(sentence):
    for morph in word.morphs:

      #동사(VV,VA)는 원형으로 바꾸기
      if morph.tag in verse: 
        root = morph.lex + '다'
        if root not in stopwords_verse:#의미없는 동사원형(stopwords) 빼기
          morphs.append(root)

      #명사,부사,감탄사
      if morph.tag in pos and len(morph.lex) > 1 and morph.lex != "ㅋㅋ":
        morphs.append(morph.lex)
    result = ' '.join(morphs)
  return result

df_me_pos["morphs"] = df_me_pos["preprocessed"].apply(Morphology_analysis)
df_me_neg["morphs"] = df_me_neg["preprocessed"].apply(Morphology_analysis)

df_you_pos["morphs"] = df_you_pos["preprocessed"].apply(Morphology_analysis)
df_you_neg["morphs"] = df_you_neg["preprocessed"].apply(Morphology_analysis)

from collections import Counter

#유저1의 긍정적인 단어
me_p_text = list(df_me_pos.morphs) #형태소 분석한것들 가져오기
me_p_text = ' '.join(me_p_text) #counter에 넣기전 처리
count = Counter(me_p_text.split()) #단어 개수 세는 counter~~~
me_p_data = count.most_common(20).to_dict() #이거 딕셔너리 형태로 사용하면 됩니당!!!
me_pos_word = pd.DataFrame(me_p_data,columns=["word","count"])

#유저1의 부정적인 단어
me_n_text = list(df_me_neg.morphs)
me_n_text = ' '.join(me_n_text)
count = Counter(me_n_text.split())
me_n_data = count.most_common(20).to_dict() #이거 딕셔너리 형태로 사용하면 됩니당!!!
me_neg_word = pd.DataFrame(me_n_data,columns=["word","count"])

#유저2의 긍정적인 단어
you_p_text = list(df_you_pos.morphs)
you_p_text = ' '.join(you_p_text)
count = Counter(you_p_text.split())
you_p_data = count.most_common(20).to_dict() #이거 딕셔너리 형태로 사용하면 됩니당!!!
you_p_word = pd.DataFrame(you_p_data,columns=["word","count"])

#유저2의 부정적인 단어
you_n_text = list(df_you_neg.morphs)
you_n_text = ' '.join(you_n_text)
count = Counter(you_n_text.split())
you_n_data = count.most_common(20).to_dict() #이거 딕셔너리 형태로 사용하면 됩니당!!!
you_n_word = pd.DataFrame(you_n_data,columns=["word","count"])