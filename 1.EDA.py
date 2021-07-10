import numpy as np
import pandas as pd

import json

data = pd.read_csv("파일경로 넣어주세요")

#0. 전체 대화 개수 ->따로 화면에 띄워줄 예정
total_text = len(data) 

#0-1. 파일내 대화시작날짜, 대화 마지막 날짜 ->따로 화면에 띄워줄 예정
firstdate = data.Date[0]
lastdate = data.Date[total_text-1]

#0-2. 사용자별로 대화 개수 -> 따로 화면에 띄워줄 예정
df_user = dict(data.groupby("User")["Message"].count())

#1. 2021년 대화개수 비교
df_2021 = data[data.Date.str[:4] == "2021"] #2021년 대화만 남기기
year_month = df_2021.Date.str[:7] #year,month부분만 문자열 자르기
df_2021["year_month"]=year_month

df_month = df_2021.groupby(["year_month"]).count().reset_index()
df_month = df_month[["year_month","Message"]] #df_month를 시각화하면 됩니당!

#2. 우리는 24시간중 언제 대화를 많이 할까?

def time_24(t):
    ime = {"밤":[22,23,0,1],"새벽":[2,3,4,5],"오전":[6,7,8,9],"낮":[10,11,12,13], 
        "오후":[14,15,16,17],"저녁":[18,19,20,21]}
    for when, hour in time.items():
        if t in hour:
        break
    return when       

data["date"] = data.Date.str[:10]
data["hour"] = data.Date.str[11:13].astype(int)
data["timeslot"] = data["hour"].apply(time_24)
df_hour = data.groupby(["date","timeslot"]).count().reset_index() #하루 timeslot별로 count
df_hour = df_hour.groupby(["timeslot"])["Message"].mean().astype(int) #timeslot기준으로 평균값 -> 시각화하면됩니당!
