# -*- coding: utf-8 -*-
"""
Created on Mon Mar 28 10:36:56 2022

@author: cui
"""

folder_path = r"C:\biwei_round3_official"

import sys 
sys.path.insert(0, folder_path)

import os
import numpy as np
import pandas as pd
import datetime as dt
import backtest_toolkit as bt
import matplotlib.pyplot as plt
plt.style.use("ggplot")


# %% Part 1: data importation

data_agg, ref = bt.data_import(folder_path)
ref = ref.set_index('Ticker')


# %% Part 2: factor generation

df_price = bt.field_load_price(data_agg, field_name='Adj Close')

df_quality_factor       = bt.quality_factor_init(data_agg)

df_quality_factor_clean = bt.quality_factor_process(df_quality_factor)


# %% Part 2.1: factor combination "without" sector neutralisation

df_quality_factor_agg_raw = bt.quality_factor_combine(df_quality_factor_clean, ref, sector_neutral = False)


strategy_raw = {};

calc_weights = bt.calc_weight(df_quality_factor_agg_raw, Nb_stock_fixed=25, flag_ew = True)
strategy_raw['Portfolio 1'] = bt.calc_ptf(df_price, calc_weights)


calc_weights = bt.calc_weight(df_quality_factor_agg_raw, Nb_stock_fixed=None, flag_ew = True)
strategy_raw['Portfolio 2'] = bt.calc_ptf(df_price, calc_weights)


calc_weights = bt.calc_weight(df_quality_factor_agg_raw, Nb_stock_fixed=None, flag_ew=False)
strategy_raw['Portfolio 3'] = bt.calc_ptf(df_price, calc_weights)


# %% Part 2.2: factor combination "with" sector neutralisation

df_quality_factor_agg_sector_neutral= bt.quality_factor_combine(df_quality_factor_clean, ref, sector_neutral = True)

strategy_sector_neutral  = {};

calc_weights = bt.calc_weight(df_quality_factor_agg_sector_neutral, Nb_stock_fixed=25, flag_ew = True)
strategy_sector_neutral['Portfolio 1'] = bt.calc_ptf(df_price, calc_weights)


calc_weights = bt.calc_weight(df_quality_factor_agg_sector_neutral, Nb_stock_fixed=None, flag_ew = True)
strategy_sector_neutral['Portfolio 2'] = bt.calc_ptf(df_price, calc_weights)


calc_weights = bt.calc_weight(df_quality_factor_agg_sector_neutral, Nb_stock_fixed=None, flag_ew=False)
strategy_sector_neutral['Portfolio 3'] = bt.calc_ptf(df_price, calc_weights)


# %% Part 3: factor portfolio computation

eval_ptf_raw = bt.strategy_evaluation(strategy_raw)

eval_ptf_sector_neutral = bt.strategy_evaluation(strategy_sector_neutral)


# %% Part 4: NAV plotting

df_plot_raw = eval_ptf_raw['NAV']

# NAV plotting
fig  = plt.figure(figsize=(9, 5))
ax_use = fig.gca();
df_plot_raw.plot(ax=ax_use)
plt.title('Quality factor - without sector neutralisation')
plt.ylabel('NAV (rebased to 100)')
plt.legend()


df_plot_sec_neutral = eval_ptf_sector_neutral['NAV']
df_plot = df_plot_raw.merge(df_plot_sec_neutral, left_index=True, right_index=True, suffixes=('- raw', '- sector neutral'))

# NAV plotting
fig  = plt.figure(figsize=(9, 5))
ax_use = fig.gca();
df_plot.plot(ax=ax_use)
plt.title('Quality factor')
plt.ylabel('NAV (rebased to 100)')
plt.legend()


# %% Part 5: table of summary

df_summary_raw = eval_ptf_raw['summary']

df_summary_sector_neutral  = eval_ptf_sector_neutral['summary']

















