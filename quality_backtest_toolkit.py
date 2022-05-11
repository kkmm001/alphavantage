# -*- coding: utf-8 -*-
"""
Created on Tue Mar 29 09:27:32 2022

@author: cui
"""
import os
import numpy as np
import pandas as pd
import datetime as dt
from pandas.tseries.offsets import BDay



# %% Step 1: data importation and adjustment

def data_import(cwd_init = r"C:\biwei_round3"):
    
    cwd = cwd_init+r'\data'
    os.chdir(cwd)
    
    #  Step 1.1: reference importation and adjustment

    # import reference csv file.
    ref = pd.read_csv('EURO_STOXX_50_Composition.csv', encoding='iso8859_2')
    
    # remove leading whiteespace of country_name
    ref['Registered'] = ref['Registered'].apply(lambda x: x.lstrip())
    
    
    
    #  Step 1.2: import fundamental data for each security
    
    report_type   = ['balance_sheet', 'income_statement', 'cash_flow', 'prices']
    
    folder_list = os.listdir(cwd)
    data_agg = {};
    
    for report_id, report_name in enumerate(report_type):
    
        df_report = [];    
        list_ticker_use = [];
        for i, folder_i in enumerate(folder_list):  
            if os.path.isdir(folder_i):
                print('No.%d - %s'%(i, folder_i))
            
                file_name = '%s_%s.csv'%(folder_i, report_name)
                file_name_agg = '%s/%s'%(folder_i, file_name)
    
                data_i = pd.read_csv(file_name_agg, encoding='iso8859_2', index_col=0)
                df_report.append(data_i)
                list_ticker_use.append(folder_i)
                
    
        df_report = pd.concat(df_report, axis=1, keys=list_ticker_use)
        data_agg[report_name] = df_report
    
    return data_agg, ref


# %% Step 2: data process

# Step 2.1: load data from "prices"
# it's quite straightforward for "prices"

def field_load_price(data_agg, start_date=None, end_date=None, field_name='Adj Close'):

    if start_date is not None:
        assert isinstance(start_date, dt.datetime), "'start_date' must both be datetime"
    
    if end_date is not None:
        assert isinstance(end_date, dt.datetime), "'end_date' must both be datetime"


    prc_adj = data_agg['prices'].loc[:, pd.IndexSlice[:, field_name]].sort_index()
    
    # flatten multi-index.
    prc_adj.columns = prc_adj.columns.get_level_values(0)
    # apply 5-step ffill in case of holiday or other situations.
    prc_adj = prc_adj.fillna(method='ffill', limit=5)

    # truncate the dates
    prc_adj = prc_adj.truncate(before=start_date, after=end_date)

    prc_adj.index.name = None;
    prc_adj.index = pd.to_datetime(prc_adj.index)


    return prc_adj


# Step 2.2: load data from "financial statement"
# create a function to load a target field and align date with prc_adj
# Attention: for any data tagged January 02, etc. it should be pushed back to end of last year.

def date_shift_weekend(target_date):

    assert isinstance(target_date, dt.datetime), "Error: 'target date' must both be datetime"

    if target_date.weekday() >= 5:  # For weekend, we search the last weekday
        target_date = target_date - BDay(1)

    return target_date


def date_shift_holiday(target_date):

    assert isinstance(target_date, dt.datetime), "Error: 'target date' must both be datetime"
    # For new year, search the last weekday.
    if (target_date.month == 1) & (target_date.day == 1):
        target_date = target_date - BDay(1)

    return target_date


def field_load_fs(data_agg, report_name='balance_sheet', field_name="Total Assets"):

    assert report_name in ['balance_sheet', 'income_statement',
                           'cash_flow'], "Error of report name: it must be one of 'balance_sheet', 'income_statement', 'cash_flow'"

    if report_name == 'balance_sheet':  # TODO: suffice for this assignment, but to improve later
        assert field_name in [
            'Total Assets', 'Total Stockholder Equity'], "Error of field name for BS: it must be one of 'Total Assets', 'Total Stockholder Equity', etc"
    if report_name == 'income_statement':  # TODO: suffice for this assignment, but to improve later
        assert field_name in [
            'Total Revenue', 'Net Income'], "Error of field name for Income: it must be one of 'Total Revenue', 'Net Income', etc"

    df_raw = data_agg[report_name].loc[field_name,
                                       :].swaplevel().unstack().copy()
    df_raw.index = pd.to_datetime(df_raw.index)

    # data cleaning for odd dates:
    # 2022-01-02, 2021-01-03, 2019-01-01, 2018-01-01 are not trading day.
    # Fund_data are supposed to be the last trading day prior to these dates

    df_raw.index.values[:] = df_raw.index.to_series().apply(
        date_shift_weekend).values
    df_raw.index.values[:] = df_raw.index.to_series().apply(
        date_shift_holiday).values

    # fundamental reporting is supposed to publish end-of-month.
    df_clean = df_raw.resample('BM').mean()
    # df_clean = df_clean.loc[df_clean.notna().mean(axis=1)>0, :] # remove months without publication.

    # forward fill 12M if missing
    df_clean = df_clean.fillna(method='ffill', limit=12)
    df_clean = df_clean.fillna(method='bfill', limit=12)

    return df_clean


# %% Step 3: Quality factor construction

def quality_factor_init(data_agg):

    assert isinstance(data_agg, dict), "Error: 'data_agg' must both be dict"
    

    df_asset = field_load_fs(
        data_agg, report_name='balance_sheet',  field_name="Total Assets")
    df_equity = field_load_fs(
        data_agg, report_name='balance_sheet',  field_name="Total Stockholder Equity")
    df_revenue = field_load_fs(
        data_agg, report_name='income_statement', field_name="Total Revenue")
    df_net_income = field_load_fs(
        data_agg, report_name='income_statement', field_name="Net Income")

    # df_net_margin   = df_net_income.div(df_revenue)
    # df_asset_turnover = df_revenue.div(df_asset)
    # df_leverage     = df_asset.div(df_equity)

    df_quality_factor = {}

    df_quality_factor['net_margin'] = df_net_income.div(df_revenue)
    df_quality_factor['asset_turnover'] = df_revenue.div(df_asset)
    df_quality_factor['leverage'] = -1 * df_asset.div(df_equity)

    return df_quality_factor


def winsorisation(df_score_raw, mode='quantile', ref=None, sector_neutral=False):

    assert isinstance(df_score_raw, pd.DataFrame), "Error: 'df_score_raw' must both be DataFrame"
    
    # two modes are ready to apply
    # 1) quantile, which winsorize them into Q5 and Q95, drop outlier about 10%
    # 2) 3 * sigma, which winsorize them into (mean-3*σ, mean + 3*σ), drop outliers 0.3%

    # Attention:
    # Banks and insurers exhibit high leverage. it's sector effect!!!
    
    
    if sector_neutral:

        assert isinstance(ref, pd.DataFrame), "Error: 'ref' must both be DataFrame of reference data"
        

        sector_info = ref.Industry.reindex(df_score_raw.columns)
        df_score_mean_sector = df_score_raw.groupby(
            sector_info, axis=1).mean().reindex(sector_info.values, axis=1)
        df_score_mean_sector.columns = df_score_raw.columns.copy()

        df_score = df_score_raw - df_score_mean_sector

    else:
        df_score = df_score_raw.copy()

    if mode == 'quantile':

        thres = df_score.quantile([0.05, 0.95], axis=1).T
        thres.columns = ['low_border', 'high_border']

        df_outlier_low = df_score.sub(thres.low_border, axis=0)
        df_outlier_low[df_outlier_low < 0] = 0
        df_score = df_outlier_low.add(thres.low_border, axis=0)

        df_outlier_high = df_score.sub(thres.high_border, axis=0)
        df_outlier_high[df_outlier_high > 0] = 0
        df_score = df_outlier_high.add(thres.high_border, axis=0)

    elif mode == 'three_sigma':

        var_mean = df_score.mean(axis=1)
        var_std = df_score.std(axis=1)

        thres = pd.concat([var_mean-var_std*3, var_mean+var_std*3], axis=1)
        thres.columns = ['low_border', 'high_border']

        df_outlier_low = df_score.sub(thres.low_border, axis=0)
        df_outlier_low[df_outlier_low < 0] = 0
        df_score = df_outlier_low.add(thres.low_border, axis=0)

        df_outlier_high = df_score.sub(thres.high_border, axis=0)
        df_outlier_high[df_outlier_high > 0] = 0
        df_score = df_outlier_high.add(thres.high_border, axis=0)

    if sector_neutral:
        df_score = df_score + df_score_mean_sector

    return df_score


def standardisation(df_score_raw, ref=None, sector_neutral=False):

    # It consists in computing "z-score"

    # 1) quantile, which winsorize them into Q5 and Q95, drop outlier about 10%
    # 2) 3 * sigma, which winsorize them into (mean-3*σ, mean + 3*σ), drop outliers 0.3%

    # Attention:
    # Banks and insurers exhibit high leverage. it's sector effect!!!

    if sector_neutral:
        
        assert isinstance(ref, pd.DataFrame), "Error: 'ref' must both be DataFrame of reference data"
        
        sector_info = ref.Industry.reindex(df_score_raw.columns)
        df_score_mean_sector = df_score_raw.groupby(sector_info, axis=1).mean().reindex(sector_info.values, axis=1)
        df_score_mean_sector.columns = df_score_raw.columns.copy()

        df_score = df_score_raw - df_score_mean_sector

    else:
        df_score = df_score_raw.copy()

    var_mean = df_score.mean(axis=1)
    var_std = df_score.std(axis=1)

    zscore = df_score.sub(var_mean, axis=0).div(var_std, axis=0)

    return zscore


def quality_factor_process(df_quality_factor, ref=None, sector_neutral=False):

    # "sector_neutral = True" enables to calculate xxx-neutral score before combination.    

    df_quality_factor_clean = {}

    for key, value in df_quality_factor.items():

        # print(key)

        df_tmp      = winsorisation(value, mode='quantile', ref=ref, sector_neutral = sector_neutral)
        df_zscore   = standardisation(df_tmp, ref=ref, sector_neutral = sector_neutral)

        df_quality_factor_clean[key] = df_zscore

    return df_quality_factor_clean


def quality_factor_combine(df_quality_factor_clean, ref=None, sector_neutral=False):

    df_quality_score_sum = [];
    fac_count = 0;
    
    for key, value in df_quality_factor_clean.items():
        
        # print(key)
        fac_count = fac_count + 1
        
        if len(df_quality_score_sum) == 0:
            df_quality_score_sum = value
        else:
            df_quality_score_sum = df_quality_score_sum.add(value)            

    df_compositive_quality_score = df_quality_score_sum/fac_count;


    if sector_neutral:
        
        assert isinstance(ref, pd.DataFrame), "Error: 'ref' must both be DataFrame of reference data"
        
        sector_info = ref.Industry.reindex(df_compositive_quality_score.columns)
        df_score_mean_sector = df_compositive_quality_score.groupby(sector_info, axis=1).mean().reindex(sector_info.values, axis=1)
        df_score_mean_sector.columns = df_compositive_quality_score.columns.copy()

        df_compositive_quality_score = df_compositive_quality_score - df_score_mean_sector


    return df_compositive_quality_score


# %% Step 4: Backtesting chain

def calc_weight(df_quality_factor_agg, Nb_stock_fixed = None, flag_ew = True, gross_expo=1, net_expo=0):
    
    # we assume the same number of stocks to include if nb_stock is fixed
    
    # firstly, I implement the equal-weight solution
    # secondly, I may attempt to weight based on quality score.
      # Worried with score-based weight because the quality score may be negative.
      # To address concern: convert z-score to negative  
          # => normal distribution function (Russell FSTE) or MSCI-like approach.
    
    
    rebal_date  = df_quality_factor_agg.index;
    df_weight   = {};
    
    expo_long   = (gross_expo + net_expo) / 2
    expo_short  = -1 * (gross_expo - net_expo) / 2

    for i, date_i in enumerate(rebal_date):
        # print(date_i)
        
        df_score_i      = df_quality_factor_agg.loc[date_i, :]
        
        if Nb_stock_fixed is not None:
            ranking_value   = df_score_i.reset_index().iloc[:, 1].argsort().sort_values().index
            flag_long       = ranking_value >= len(ranking_value) - Nb_stock_fixed
            flag_short      = ranking_value < Nb_stock_fixed
        else:
            flag_long       = df_score_i>=0
            flag_short      = df_score_i<0

        df_weight_i     = pd.Series(np.nan, index=df_score_i.index)
        
        if flag_ew:
            df_weight_i[flag_long]  = expo_long  / sum(flag_long)
            df_weight_i[flag_short] = expo_short / sum(flag_short)
        else:
            df_weight_i[flag_long]  = expo_long  / (df_score_i[flag_long].sum()) * (df_score_i[flag_long])
            df_weight_i[flag_short] = expo_short / (df_score_i[flag_short].sum()) * (df_score_i[flag_short])


        df_weight[date_i] = df_weight_i.reindex(df_score_i.index);

    df_weight = pd.DataFrame(df_weight).T
    
    return df_weight


# Function for strategy evaluation

def calc_ptf(df_price, input_weights, start_date  = None, 
                end_date = None, inception_value = 100):
   
    if start_date is not None:
        assert isinstance(start_date, dt.datetime), "'start_date' must both be datetime"
    
    if end_date is not None:
        assert isinstance(end_date, dt.datetime), "'end_date' must both be datetime"
    
    
    input_weights  = input_weights.truncate(before = start_date, after = end_date)
   
    df_price        = df_price.truncate(before = input_weights.index[0], after = input_weights.index[-1])
    rebal_date      = input_weights.index

    union_date      = df_price.index.append(rebal_date).unique().sort_values() #(ascending = False)
    df_price        = df_price.reindex(union_date, method='ffill')
    
    daily_return_next   = df_price.pct_change(1, fill_method = None).shift(-1)

    weight_close    = daily_return_next * np.nan
    weight_open     = daily_return_next * np.nan
    
    
    daily_return_ptf = pd.Series(0, index = daily_return_next.index)
    
    for pos, date in enumerate(daily_return_next.index[:-1]):  
        
        if date in rebal_date:
            open_expo_next = input_weights.loc[date, :]
        else:
            open_expo_next = weight_close.iloc[pos, :]
            
        close_expo_next             = (1 + daily_return_next.iloc[pos, :]) * open_expo_next
        daily_return_ptf.iloc[pos+1]= (daily_return_next.iloc[pos, :] * open_expo_next).sum()
        
        weight_close.iloc[pos+1, :] = close_expo_next;   
        weight_open.iloc[pos+1,  :] = open_expo_next;
    
    
    weight_close.loc[rebal_date[0], :]  = input_weights.loc[rebal_date[0], :].values
    daily_return_ptf.loc[rebal_date[0]] = np.nan;
    
    nav_ptf = np.cumprod(1+daily_return_ptf.fillna(0))*inception_value
    
    return {'nav_ptf':nav_ptf, 'daily_return_ptf':daily_return_ptf,
            'weight':weight_close, 'weight_open':weight_open, 'input_weights':input_weights}




# Function for strategy evaluation

def strategy_evaluation(strategy, start_date = None, end_date = None):

    if start_date is not None:
        assert isinstance(start_date, dt.datetime), "'start_date' must both be datetime"
    
    if end_date is not None:
        assert isinstance(end_date, dt.datetime), "'end_date' must both be datetime"


    result_dict = dict(); 
   
    # 1. NAV of factors
    df_nav = {};
    
    for key, value in strategy.items():
        df_nav[key] = value['nav_ptf'].truncate(before=start_date, after=end_date)
    df_nav = pd.DataFrame(df_nav) 
        
    nav_rebase = df_nav/df_nav.bfill().iloc[0,:]*100
    result_dict['NAV'] = nav_rebase;


    # 2. return of factors
    df_ret = {};
    
    for key, value in strategy.items():
        df_ret[key] = value['daily_return_ptf'].truncate(before=start_date, after=end_date)        
    df_ret = pd.DataFrame(df_ret) 
    result_dict['return'] = df_ret
        
    # 3. summmary of factors
    df_summary = summarise_ptf(df_ret, start_date = start_date, end_date=end_date)
    
    result_dict['summary'] = df_summary.T


    return result_dict


# Calculate the var and CVar given daily returns

def summarise_ptf(returns, start_date = None, end_date = None, 
                  annualization = 250, cutoff = 0.01):
    
    if start_date is not None:
        assert isinstance(start_date, dt.datetime), "'start_date' must both be datetime"
    
    if end_date is not None:
        assert isinstance(end_date, dt.datetime), "'end_date' must both be datetime"
    
    assert isinstance(returns, pd.DataFrame), "Error: 'returns' must both be DataFrame"

    
    import empyrical
        
    returns     = returns.truncate(before=start_date, after=end_date)
    
    ann_return  = empyrical.annual_return(returns, annualization = annualization)
    ann_vol     = empyrical.annual_volatility(returns, annualization = annualization)  
    MDD         = empyrical.max_drawdown(returns)
    sortino     = empyrical.sortino_ratio(returns, required_return = 0, annualization = annualization)  
    sharpe      = empyrical.sharpe_ratio(returns, risk_free = 0, annualization = annualization) 

    var, cvar   = var_cvar(returns, cutoff = cutoff);
    statistic   = {'ann_return' : ann_return.values,'ann_vol' :ann_vol, 
                   'max_drawdown':MDD.values, 'sortino' : sortino.values, 
                   'sharpe' : sharpe, 'var':var, 'cvar':cvar}
  
    df_out = pd.DataFrame(statistic)
    df_out.index = ann_return.index.to_list()
    
    return df_out


# Calculate the var and CVar given daily returns
def var_cvar(returns, cutoff = 0.01):
    
    assert isinstance(returns, pd.DataFrame), "Error: 'returns' must both be DataFrame"
    
    var         = returns.quantile(q=cutoff)
    cvar_ret    = returns.copy()
    cvar_ret[returns.gt(var, axis=1)] = np.nan;
    cvar        = cvar_ret.mean()
    return var, cvar
