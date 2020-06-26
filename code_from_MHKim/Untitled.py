#!/usr/bin/env python
# coding: utf-8

# In[1]:


from pandas import Series, DataFrame


# In[2]:


import pandas as pd


# In[3]:


import matplotlib as mpl


# In[4]:


import re


# In[5]:


import csv


# In[6]:


import threading


# In[7]:


from bs4 import BeautifulSoup
import urllib.request as MyURL


# In[8]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

font_name = mpl.font_manager.FontProperties(fname='C:/Windows/Fonts/malgun.ttf').get_name()
mpl.rc('font', family=font_name)


# In[9]:


mpl.rcParams['axes.unicode_minus'] = False


# In[10]:


WT = 'http://www.kma.go.kr/wid/queryDFSRSS.jsp?zone=4711370000'
response = MyURL.urlopen(WT)
weather = BeautifulSoup(response, "html.parser")

for pubdate in weather.find('pubdate'):
    print("현재일시:",pubdate.string)
for title in weather.find('title'):
    print("위치:",title.string)


# In[11]:


#과거 데이터에 현재 및 미래 데이터 합산
def PastFuture():
    
    time = []
    temp = []
    wind = []
    day = []
    timeall=[]
    
    for data in weather.findAll('data'):
        time_split = data.hour.string.split("/")
        time = time + time_split
    
        temp_split = data.temp.string.split("/")
        temp = temp + temp_split
        
        wind_split = data.ws.string.split("/")
        wind = wind + wind_split
    
        day_split = data.day.string.split("/")
        day = day + day_split
    
    str2 = re.findall("\d+",pubdate.string)

    year = int(str2[0])
    month = int(str2[1])
    day_b = int(str2[2])
    day_I = map(int,day)

    Day_re = []
    for i in day_I:
        Day = str(i + day_b) + ' '
        Day_re = Day_re + Day.split()
        Day_I = map(int, Day_re)
        
        Day_em = []
        Day_om = []
        month_om = []
        month_em = []
        month_e = []
        year_om = []
        year_em = []

    for i in Day_I:
    
        if ((i < 31) and (month == 4 or month == 6 or month == 9 or month == 11)):
            Day = str(i) + ' '
            Day_om = Day_om + Day.split()
            month_om = month_om + list(str(month))
    
        elif ((i >= 31) and (month == 4 or month == 6 or month == 9 or month == 11)):
            Day = str(i - 30) + ' '
            Day_em = Day_em + Day.split()
            month_em = month_em + list(str(month+1))
        
        elif ((i < 29) and (month == 2) and (year % 4 == 0)):
            Day = str(i) + ' ' 
            Day_om = Day_om + Day.split()
            month_om = month_om + list(str(month))  
        
        elif ((i >= 29) and (month == 2) and (year % 4 == 0)):
            Day = str(i - 28) + ' ' 
            Day_em = Day_em + Day.split()
            month_em = month_em + list(str(month+1))
        
        elif ((i < 30) and (month == 2)):
            Day = str(i) + ' '
            Day_om = Day_om + Day.split()
            month_om = month_om + list(str(month))
        
        elif ((i >= 30) and (month == 2)):
            Day = str(i - 29) + ' '
            Day_em = Day_em + Day.split()
            month_em = month_em + list(str(month+1))
        
        elif (i < 32):
            Day = str(i) + ' '
            Day_om = Day_om + Day.split()
            month_oms = str(month) + ' '
            month_om = month_om + month_oms.split()
        
        elif (i >= 32):
            Day = str(i - 31) + ' '
            Day_em = Day_em + Day.split()
            month_ems = str(month+1) + ' '
            month_em = month_em + month_ems.split()

    month_pf = month_om + month_em

    month_pfI = map(int,month_pf) 
    for i in month_pfI:

        if i == 12:
            Year = str(year) + ' '
            year_om = year_om + Year.split() 
    
        elif i == 13:
            month_e = month_e + list(str(i - 12))
            month_pf = month_om + month_e
            Year = str(year+1) + ' '
            year_em = year_em + Year.split()
    
    Day_pf = Day_om + Day_em
    year_pf = year_om + year_em

    year_s = ''
    for i in range(len(Day_re)):
        year_s = str2[0] + ' '
        year_pf = year_pf + year_s.split()

    DATE_ALL = ''
    for i in range(len(day)):
        DATE_ALL = DATE_ALL + year_pf[i] + '-' + month_pf[i] + '-' + Day_pf[i] + ' ' + time[i] + '.'
        DATE_RE = DATE_ALL.split(".")
    
    del DATE_RE[DATE_RE.index("")]
    
    with open(r'C:/Users/MinHyun/Desktop/PAST+FUTURE.csv', 'a', newline='') as csvfile:
        fieldnames = ['시간','기온(°C)','풍속(m/s)']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        for i in range(len(day)):
            writer.writerow({'시간':DATE_RE[i],'기온(°C)':temp[i],'풍속(m/s)':wind[i]})
            
PastFuture()
#3시간 간격으로 함수 실행
threading.Timer(10800,PastFuture).start()


# In[12]:


#중복항 제거 및 제거된 값으로 그래프 그리는 함수
def drop_plot():

    df_forecast = pd.read_csv("C:/Users/MinHyun/Desktop/PAST+FUTURE.csv",encoding='cp949')
    df_re = df_forecast.drop_duplicates(['일시'])

    df_re1 = df_re.set_index(['일시'])
    df_re1.to_csv('C:/Users/MinHyun/Desktop/PAST+FUTURE.csv',encoding='cp949')

    ax3 = df_re1.plot(figsize=(12,4),legend=True, fontsize=12)
    ax3.set_title(['과거 데이터 및 '+ pubdate.string+'시 기준 예보 데이터', title.string])
    ax3.legend(['기온(°C)','풍속(m/s)'], fontsize=12)
    ax3.set_ylabel("풍속(m/s), 기온(°C)")
    ax3.set_xlabel("일시")
    ax3.grid(True)
    
    plt.tight_layout()
    for label in ax3.xaxis.get_ticklabels() :
        label.set_rotation(45)
    
drop_plot()

#3시간 간격으로 함수 실행
threading.Timer(10800,drop_plot).start()


#기울기 평균값 데이터 그래프
df=pd.read_csv("C:/Users/MinHyun/Desktop/2020-04-02_monitoring_data.csv", index_col="MTime", encoding='UTF-8')
Avg_df = df.loc[:,["MXSineAvg","MYSineAvg"]]

ax = Avg_df.plot(title='포항 크리스탈 원룸', figsize=(12,4),legend=True, fontsize=12)
ax.set_ylim([-0.1,0.1])
ax.set_xlabel('일시',fontsize=12)
ax.set_ylabel('기울기', fontsize=12)
ax.grid(True)

ax.legend(['x기울기','y기울기'], fontsize=12)
plt.tight_layout()
for label in ax.xaxis.get_ticklabels() :
    label.set_rotation(45)


# In[ ]:




