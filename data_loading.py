# -*- coding: utf-8 -*-
"""
Created on Sun May  1 11:31:21 2022

@author: cui
"""

import os, re
import numpy as np
import pandas as pd

import csv
import datetime
import yfinance as yf
from progressbar import progressbar
import pickle


# %% Load Reference Data

cwd = r"C:\st_sim\Alpha_avantage\data"
os.chdir(cwd)    

list_holding    = pd.read_pickle('raw_holding_msci_usa.pkl')
df_ref          = list_holding['20220428']


# %% Yahoo Finance - price and volume

ISIN_list   = df_ref['ISIN'].to_list()
Ticker_list = df_ref['Issuer Ticker'].to_list()


df_close  = {};
df_volume = {};

# for ISIN_i in ISIN_list:
for i in progressbar(range(len(ISIN_list))):
    
    ISIN_i  = ISIN_list[i];
    Ticker_i  = Ticker_list[i];

    symbol_use = Ticker_i
    # symbol_use = 'HEI-A'   # Example to Test for 
    
    ticker  = yf.Ticker(symbol_use)
    df_i    = ticker.history(period="max")

    if len(df_i) == 0:
        ticker  = yf.Ticker(symbol_use[:-1]+"-"+symbol_use[-1])
        df_i    = ticker.history(period="max")


    df_close[symbol_use]  = df_i['Close']
    df_volume[symbol_use] = df_i['Volume']


cwd = r"C:\st_sim\Alpha_avantage\data"
os.chdir(cwd)    
with open('mkt_data_msci_usa.pkl', 'wb') as f:
    pickle.dump(df_close, f)
    pickle.dump(df_volume, f)
    pickle.dump(df_ref, f)
    
    
    
# %% Yahoo Finance - Fundamental Data
    
cwd = r"C:\st_sim\Alpha_avantage\data"
 # => It appears that 220 is missing ? check
 

ISIN_list   = df_ref['ISIN'].to_list()
Ticker_list = df_ref['Issuer Ticker'].to_list()

# for ISIN_i in ISIN_list:
for i in progressbar(range(221, len(ISIN_list))):
    
    ISIN_i  = ISIN_list[i];
    Ticker_i  = Ticker_list[i];

    symbol_use = Ticker_i
    # symbol_use = 'HEIA'   # Example to Test for 
    
    ticker  = yf.Ticker(symbol_use)
    
    df_info = ticker.info.copy()
    
    if len(df_info) == 0:
        ticker  = yf.Ticker(symbol_use[:-1]+"-"+symbol_use[-1])
        # df_i    = ticker.history(period="max")
    
    df_income_Y     = ticker.financials.copy()
    df_income_Q     = ticker.quarterly_financials.copy()
    flag_Y          = ~np.in1d(df_income_Y.columns, df_income_Q.columns)
    df_income_Y     = df_income_Y.loc[:, flag_Y]
    df_income_agg   = df_income_Q.merge(df_income_Y, left_index=True, right_index=True)


    df_bs_Y     = ticker.balance_sheet.copy()
    df_bs_Q     = ticker.quarterly_balance_sheet.copy()
    flag_Y      = ~np.in1d(df_bs_Y.columns, df_bs_Q.columns)
    df_bs_Y     = df_bs_Y.loc[:, flag_Y]
    df_bs_agg   = df_bs_Q.merge(df_bs_Y, left_index=True, right_index=True)

    str_path = '%s\\%s'%(cwd, symbol_use)
    if not os.path.exists(str_path):
        os.makedirs(str_path)

    os.chdir(str_path)    
    df_income_agg.to_csv('%s_income_statement.csv'%symbol_use)
    df_bs_agg.to_csv('%s_balance_sheet.csv'%symbol_use)







