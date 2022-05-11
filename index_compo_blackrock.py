# -*- coding: utf-8 -*-
"""
Created on Mon Jan  3 15:49:11 2022

@author: bcui
"""
import os
import pickle
import urllib
import numpy as np
import pandas as pd
import requests

import datetime

def last_day_of_month(any_day):
    # this will never fail
    # get close to the end of the month for any day, and add 4 days 'over'
    next_month = any_day.replace(day=28) + datetime.timedelta(days=4)
    # subtract the number of remaining 'overage' days to get last day of current month, or said programattically said, the previous day of the first of next month
    return next_month - datetime.timedelta(days=next_month.day)

def read_backrock(date_i):
    # link = "https://www.blackrock.com/americas-offshore/en/products/251931/ishares-stoxx-europe-600-ucits-etf-de-fund/1474306011330.ajax?fileType=csv&fileName=EXSA_holdings&dataType=fund&asOfDate=%s"%date_i.strftime("%Y%m%d")
    link = "https://www.blackrock.com/americas-offshore/en/products/253740/ishares-msci-usa-b-ucits-etf/1474306011330.ajax?fileType=csv&fileName=CSUS_holdings&dataType=fund&asOfDate=%s"%date_i.strftime("%Y%m%d")
    f = urllib.request.urlopen(link)
    myfile = f.read()
    
    file_init = myfile.decode().split("\n")[2:]
    list_res = {};   
    
    for i, word_i in enumerate(file_init): 
        print(word_i)
        if len(word_i)>5:
            flag_head = word_i.find(r'Issuer Ticker')
            if flag_head>=0:
                word_use_i  = word_i.split(',')
            else:
                word_use_i  = word_i[1:-1].split('","')
            
            list_res[i] = pd.Series(word_use_i)
    
    df_res = pd.DataFrame(list_res).set_index(0)
    df_res.index.name = None
    df_res = df_res.T
    
    return df_res


def read_backrock_csv(date_i):
    # link = "https://www.blackrock.com/americas-offshore/en/products/251931/ishares-stoxx-europe-600-ucits-etf-de-fund/1474306011330.ajax?fileType=csv&fileName=EXSA_holdings&dataType=fund&asOfDate=%s"%date_i.strftime("%Y%m%d")
    link = "https://www.blackrock.com/americas-offshore/en/products/253740/ishares-msci-usa-b-ucits-etf/1474306011330.ajax?fileType=csv&fileName=CSUS_holdings&dataType=fund&asOfDate=%s"%date_i.strftime("%Y%m%d")
    resp = requests.get(link)
    filename = "%s_holding_msci_usa.csv"%date_i.strftime("%Y%m%d")
    with open(filename, 'wb') as output:
        output.write(resp.content)



date_list = [];
for year in range(2014, 2023):
    for month in range(1, 13):
        # date_i = last_day_of_month(datetime.date(2021, month, 1));
        date_i = pd.offsets.BMonthEnd().rollforward(datetime.date(year, month, 1))
        if date_i <= datetime.datetime.today():
            print(date_i)
            date_list.append(date_i)


# %% holding loading in dictionary
list_holding = {}

for date_i in date_list:
    print(date_i)
    hold_i = read_backrock(date_i)
    list_holding[str(date_i.date())] = hold_i
    # list_holding[str(date_i)] = hold_i

date_list_bad = [datetime.date(2013,3,28), datetime.date(2013,12,30), datetime.date(2014,12,30), datetime.date(2015,12,30)]

for date_i in date_list_bad:
# date_i = datetime.date(2013,12,30)
    hold_i = read_backrock(date_i)
    list_holding[str(date_i)] = hold_i


# %% holding downloading in Excel
cwd = r"C:\st_sim\Alpha_avantage\blackrock_csv_msci_USA"
os.chdir(cwd)    

for date_i in date_list:
    print(date_i)
    read_backrock_csv(date_i)
    
# date_list_bad = [datetime.date(2013,3,28), datetime.date(2013,12,30), datetime.date(2014,12,30), datetime.date(2015,12,30)]
date_list_bad = [datetime.date(2022,4,28)]

for date_i in date_list_bad:
    read_backrock_csv(date_i)


# %% Displaying files in the given folder.
cwd = r"C:\st_sim\Alpha_avantage\blackrock_csv_msci_USA"

files = os.listdir(cwd)

list_holding = {};
for f in files:
    print(f)
    df_init     = pd.read_csv(f, skiprows=[0, 1]);
    flag_clean  = (df_init['Asset Class']=='Equity') & (df_init['Market Currency']=='USD')
    df_clean    = df_init.loc[flag_clean, :] 
    
    date_use    = f.replace('_holding_msci_usa.csv', '')
    list_holding[date_use] = df_clean
    
    
# %%

cwd = r"C:\st_sim\Alpha_avantage\data"
os.chdir(cwd) 

# Save Data to local file
with open('raw_holding_msci_usa.pkl', 'wb') as f:
    pickle.dump(list_holding, f)


# %%














