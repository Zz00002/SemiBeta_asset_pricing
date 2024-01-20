# -*- coding: utf-8 -*-
"""
Created on Mon Jul 10 15:16:18 2023

@author: asus
"""

import numpy as np
import pandas as pd
import ToolBox as TB
from matplotlib import pyplot as plt
import re
from AssetPricing_test import AssetPricingTool
from multiprocessing import Pool, cpu_count
from Volatility_Study_Tool import Volatility_Tool
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
import os

os.chdir('..')


###############################################################################
## Create controled variables
###############################################################################

# continue and discontinue beta
# From: Bollerslev, T., Li, S.Z., Todorov, V., 2016. Roughing up beta: Continuous versus discontinuous betas and the cross section of expected stock returns. Journal of Financial Economics 120, 464–490. https://doi.org/10.1016/j.jfineco.2016.02.001

# Calculate continuse and discontinuse beta    
def Exec_ConDisconBeta(stkcd, hf_index_, idxcd=300, min_=5):
        
    data_dir = os.getcwd()

    VT = Volatility_Tool()
    hf_stock_ = TB.Fetch_Stock_HFdata_from_Resset(stkcd, minType=min_)
    
    print(data_dir)

    hf_stock_ = VT.Cpt_HF_LogReturn(hf_stock_).rename(columns={'log_close_diff':'ret_stk'})[['Trddt','time','ret_stk']]
    
    beta_day_df = VT.Cpt_ConDiscon_Beta(hf_stock_, hf_index_)
    beta_day_df['Stkcd'] = stkcd
    
    beta_day_df.to_csv(data_dir + '\\data_sample\\ConDisconBeta\\{}\\{}\\{}.csv'.format(min_, idxcd, stkcd), index=False)
    print('Finshed ConDisconBeta result calculation: {}_{}_{}'.format(min_, idxcd, stkcd))
    
    
# Mult Calculate continuse and discontinuse beta    
def Muti_Exec_ConDisconBeta(idxcd=300, min_=5, mult=True):
    
    data_dir = os.getcwd()

    VT = Volatility_Tool()
    stock_base_df = pd.read_csv(data_dir + '\\data_sample\\SAVIC_saveMV.csv')
    hf_index_ = TB.Fetch_Stock_HFdata_from_Resset(idxcd, asset_type='index', minType=min_)
    hf_index_ = VT.Cpt_HF_LogReturn(hf_index_).rename(columns={'log_close_diff':'ret_idx'})[['Trddt','time','ret_idx']]
    
    skip_dir1 = data_dir + '\\data_sample\\ConDisconBeta'
    skip_list = TB.Tag_FilePath_FromDir(skip_dir1)

    if mult:
        num_processes = 8
        # Create a Pool of processes
        with Pool(num_processes) as pool:
            for stkcd in stock_base_df.Stkcd.unique().tolist():
                file_name = skip_dir1 + '\\{}\\{}\\{}.csv'.format(min_, idxcd, stkcd)
                if file_name not in skip_list:
                    print(file_name)
                    pool.apply_async(Exec_ConDisconBeta, (stkcd, hf_index_))
        
            # Close the pool and wait for all processes to finish
            pool.close()
            pool.join()
    else:
        for stkcd in stock_base_df.Stkcd.unique().tolist():
            file_name = skip_dir1 + '\\{}\\{}\\{}.csv'.format(min_, idxcd, stkcd)
            if file_name not in skip_list:
                Exec_ConDisconBeta(stkcd, hf_index_)
        

# Merging continuse and discontinuse beta results
def Mrg_ConDisconBeta(index, min_=5):
    
    data_dir = os.getcwd()
    
    D_Ba_path = data_dir + '\\data_sample\\ConDisconBeta\\{}\\{}'.format(min_,index)
    file_list = TB.Tag_FilePath_FromDir(D_Ba_path)
    ConDisconBeta_list = []

    for file in file_list:
        try:
            stkcd = file.split('\\')[-1].split('.')[0]
            # print(stkcd)
            ConDisconBeta = pd.read_csv(file, index_col=0).reset_index()
            ConDisconBeta['Stkcd'] = stkcd
            ConDisconBeta_list.append(ConDisconBeta)
        except:
            pass
        
    ConDisconBeta_df = pd.concat(ConDisconBeta_list)
    ConDisconBeta_df['Trddt'] = ConDisconBeta_df.Trddt.astype(str)
    ConDisconBeta_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    ConDisconBeta_df.dropna(inplace=True)
    
    ConDisconBeta_df.to_csv(data_dir + '\\data_sample\\{}\\ConDisconBeta_{}_{}.csv'.format(min_, index, min_), index=False)
    print('finished ConDisconBeta_{}_{}.csv'.format(index, min_))


# realized beta and four different semi-beta
# From: Bollerslev, T., Patton, A.J., Quaedvlieg, R., 2022. Realized semibetas: Disentangling “good” and “bad” downside risks. Journal of Financial Economics 144, 227–246. https://doi.org/10.1016/j.jfineco.2021.05.056

# Calculate simple semi-beta  
def Exec_SemiBeta(index_dict, min_,idxcd,stkcd):
    
    data_dir = os.getcwd()
    VT = Volatility_Tool()    

    try:
        hf_stock_ = TB.Fetch_Stock_HFdata_from_Resset(stkcd, minType=min_)
        hf_stock_ = VT.Cpt_HF_LogReturn(hf_stock_).rename(columns={'log_close_diff':'ret_stk'})[['Trddt','time','ret_stk']]
    
        hf_index_ = VT.Cpt_HF_LogReturn(index_dict[idxcd])
        hf_index_ = VT.Cpt_HF_LogReturn(hf_index_).rename(columns={'log_close_diff':'ret_idx'})[['Trddt','time','ret_idx']]
    
        beta_day_df = VT.Cpt_SemiBeta(hf_stock_, hf_index_)
        beta_day_df.to_csv(data_dir + '\\data_sample\\SemiBeta\\Betas\\{}\\{}\\{}.csv'.format(min_, idxcd, stkcd), index=False)
        print('Finshed SemiBeta result calculation: {}_{}_{}'.format(min_, idxcd, stkcd))

    except:
        pd.DataFrame([stkcd,idxcd,min_]).to_csv(data_dir + '\\data_sample\\SemiBeta\\error\\error_{}_{}_{}.csv'.format(stkcd,min_,idxcd))
        print('error_{}_{}_{}'.format(stkcd,min_,idxcd))

    
# Mult Calculate simple semi-beta  
def Muti_Exec_SemiBeta(mult=True):
    
    data_dir = os.getcwd()
    skip_dir1 = data_dir + '\\data_sample\\SemiBeta\\Betas'
    
    stock_base_data = pd.read_csv(data_dir + '\\data_sample\\SAVIC_saveMV.csv')    
    skip_list = TB.Tag_FilePath_FromDir(skip_dir1)
    
    
    if mult:

        num_processes = 4
        # Create a Pool of processes
        with Pool(num_processes) as pool:
            for min_ in [5]:
                
                hf_index_300 = TB.Fetch_Stock_HFdata_from_Resset(300, asset_type='index', minType=min_) 
                hf_index_4000 = pd.read_csv(data_dir + '\\data_sample\\index\\4000_{}.csv'.format(min_))
                index_dict = {300:hf_index_300, 4000:hf_index_4000}
    
                for idxcd in [300,4000]:
                    for stkcd in stock_base_data.Stkcd.unique().tolist():
                        file_name = skip_dir1 + '\\{}\\{}\\{}.csv'.format(min_, idxcd, stkcd)
                        if file_name not in skip_list:
                            pool.apply_async(Exec_SemiBeta, (index_dict, min_, idxcd, stkcd))
        
            # Close the pool and wait for all processes to finish
            pool.close()
            pool.join()
        
    else:
        for min_ in [5]:
            
            hf_index_300 = TB.Fetch_Stock_HFdata_from_Resset(300, asset_type='index', minType=min_) 
            hf_index_4000 = pd.read_csv(data_dir + '\\data_sample\\index\\4000_{}.csv'.format(min_))
            index_dict = {300:hf_index_300, 4000:hf_index_4000}

            for idxcd in [300,4000]:
                for stkcd in stock_base_data.Stkcd.unique().tolist():
                    file_name = skip_dir1 + '\\{}\\{}\\{}.csv'.format(min_, idxcd, stkcd)
                    if file_name not in skip_list:
                        Exec_SemiBeta(index_dict, min_, idxcd, stkcd)
  
        
# Merging Calculate simple semi-beta results
def Mrg_SemiBeta(index, min_):
    
    data_dir = os.getcwd()
    
    D_Ba_path = data_dir + '\\data_sample\\SemiBeta\\Betas\\{}\\{}'.format(min_,index)
    file_list = TB.Tag_FilePath_FromDir(D_Ba_path)
    semibeta_list = []

    for file in file_list:
        try:
            stkcd = file.split('\\')[-1].split('.')[0]
            # print(stkcd)
            semibeta = pd.read_csv(file, index_col=0).reset_index()
            semibeta['Stkcd'] = stkcd
            semibeta_list.append(semibeta)
            print(file)
        except:
            pass
        
    semibeta_df = pd.concat(semibeta_list)
    semibeta_df['Trddt'] = semibeta_df.Trddt.astype(str)
    semibeta_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    semibeta_df.dropna(inplace=True)
    
    semibeta_df.to_csv(data_dir + '\\data_sample\\SemiBeta\\SemiBeta_{}_{}.csv'.format(min_,index), index=False)
    
    
# Mult Merging Calculate simple semi-beta results
def Mult_Mrg_SemiBeta(idx_lst, min_lst, mult=True):
    
    if mult:
    
        num_processes = 8
        # Create a Pool of processes
        with Pool(num_processes) as pool:
            for index in idx_lst:
                for min_ in min_lst:
                    pool.apply_async(Mrg_SemiBeta, (index, min_))
    
            # Close the pool and wait for all processes to finish
            pool.close()
            pool.join()
        
    else:
        for index in idx_lst:
            for min_ in min_lst:
                Mrg_SemiBeta(index, min_)



# Related sign jump that using RV to construct
# From: Bollerslev, T., Li, S.Z., Zhao, B., 2020. Good Volatility, Bad Volatility, and the Cross Section of Stock Returns. J. Financ. Quant. Anal. 55, 751–781. https://doi.org/10.1017/S0022109019000097

# Calculate RSJ base on RV  
def Exec_RSJ_on_RV(stkcd, min_):
    print('start {}'.format(stkcd))
    
    data_dir = os.getcwd()
    
    VT = Volatility_Tool()
    hf_stock_ = TB.Fetch_Stock_HFdata_from_Resset(stkcd, minType=min_)
    hf_stock_ = VT.Cpt_HF_LogReturn(hf_stock_,drop0=True).rename(columns={'log_close_diff':'ret_stk'})[['Trddt','time','ret_stk']]
    
    hf_data = VT.Cpt_DecomRV(hf_stock_, retTag='ret_stk')
    hf_data = VT.Cpt_RM(hf_data, retTag='ret_stk')
    data_day = hf_data[['Trddt','rvp','rvn','rv']].drop_duplicates().reset_index(drop=True)
    data_day['RSJ'] = (data_day.rvp - data_day.rvn)/data_day.rv
    data_day['Stkcd'] = stkcd
    data_day.replace([np.inf, -np.inf], np.nan, inplace=True)
    data_day.dropna(inplace=True)

    data_day.to_csv(data_dir + '\\data_sample\\RSJ_RV\\{}\\{}.csv'.format(min_, stkcd), index=False)
    print('Finshed RSJ_RV result calculation: {}_{}'.format(min_, stkcd))


# Mult Calculate RSJ base on RV  
def Muti_Exec_RSJ_RV(mult=True):
    
    data_dir = os.getcwd()
    
    stock_base_df = pd.read_csv(data_dir + '\\data_sample\\SAVIC_saveMV.csv')    
    use_stkcd = stock_base_df.Stkcd.unique().tolist()
    stock_base_df = 0
    
    skip_dir1 = data_dir + '\\data_sample\\RSJ_RV'
    skip_list = TB.Tag_FilePath_FromDir(skip_dir1)

    if mult:
        num_processes = 16
        # Create a Pool of processes
        with Pool(num_processes) as pool:
            for min_ in [5]:
                for stkcd in use_stkcd:
                    file_name = skip_dir1 + '\\{}.csv'.format(stkcd)
                    if file_name not in skip_list:
                        pool.apply_async(Exec_RSJ_on_RV, (stkcd, min_,))
        
            # Close the pool and wait for all processes to finish
            pool.close()
            pool.join()
    else:
        for min_ in [5]:
            for stkcd in use_stkcd:
                file_name = skip_dir1 + '\\{}.csv'.format(stkcd)
                if file_name not in skip_list:
                    Exec_RSJ_on_RV(stkcd, min_)


def Mrg_RSJ_RV(min_):
    
    data_dir = os.getcwd()
    
    D_Ba_path = data_dir + '\\data_sample\\RSJ_RV\\{}'.format(min_)
    file_list = TB.Tag_FilePath_FromDir(D_Ba_path)
    RSJ_RV_list = []

    for file in file_list:
        try:
            # print(stkcd)
            RSJ_RV = pd.read_csv(file, index_col=0).reset_index()
            RSJ_RV_list.append(RSJ_RV)
        except:
            pass
        
    RSJ_RV_df = pd.concat(RSJ_RV_list)
    RSJ_RV_df['Trddt'] = RSJ_RV_df.Trddt.astype(str)
    RSJ_RV_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    RSJ_RV_df.dropna(inplace=True)

    RSJ_RV_df.to_csv(data_dir + '\\data_sample\\RSJ_RV\\RSJ_RV_{}.csv'.format(min_), index=False)


def Cpt_IVOL(df, Stkcd, window=20):
    
    data_dir = os.getcwd()
    
    def run_regression(df):
        X = sm.add_constant(df[['mkt', 'smb', 'vmg']])
        model = sm.OLS(df['ex_ret'], X, missing='drop')
        results = model.fit()
        return results.resid.std()
    
    df = df.reset_index(drop=True)
    for i, df_ in enumerate(list(df.rolling(window))):
        if i >= window-1:
            df.loc[i,'IVOL'] = run_regression(df_)
    df.to_csv(data_dir + '\\data_sample\\IVOL\\CH3\\{}.csv'.format(Stkcd),index=False)
    print('{} IVOL finished'.format(Stkcd))


def Mult_Cpt_IVOL(mult=True):
    
    data_dir = os.getcwd()

    stock_day_trade_data = pd.read_csv(data_dir + '\\data_sample\\SAVIC_saveMV_day.csv', usecols=['Stkcd', 'Trddt', 'Dretwd'])
    
    SVIC_df = pd.read_csv(data_dir + '\\data_sample\\CH3_daily.csv')
    SVIC_df.columns = ['Trddt','rf','mkt','smb','vmg']

    IVOL_df = pd.merge(stock_day_trade_data, SVIC_df)
    IVOL_df = IVOL_df.sort_values(['Stkcd','Trddt']).reset_index(drop=True)
    IVOL_df['ex_ret'] = IVOL_df.Dretwd - IVOL_df.rf
            
    skip_dir1 = data_dir + '\\data_sample\\IVOL\\CH3'
    skip_list = TB.Tag_FilePath_FromDir(skip_dir1)

    if mult:
        num_processes = 16
        # Create a Pool of processes
        with Pool(num_processes) as pool:
            for Stkcd, df in IVOL_df.groupby('Stkcd'):
                file_name = skip_dir1 + '\\{}.csv'.format(Stkcd)
                if file_name not in skip_list:
                    pool.apply_async(Cpt_IVOL, (df, Stkcd,))
        
            # Close the pool and wait for all processes to finish
            pool.close()
            pool.join()
            
    else:
        for Stkcd, df in IVOL_df.groupby('Stkcd'):
            file_name = skip_dir1 + '\\{}.csv'.format(Stkcd)
            if file_name not in skip_list:
                Cpt_IVOL(df, Stkcd)
              
        
def Mrg_IVOL():
    
    data_dir = os.getcwd()
    
    D_Ba_path = data_dir + '\\data_sample\\IVOL\\CH3'
    file_list = TB.Tag_FilePath_FromDir(D_Ba_path)
    semibeta_list = []

    for file in file_list:
        try:
            # print(stkcd)
            semibeta = pd.read_csv(file, index_col=0).reset_index()
            semibeta_list.append(semibeta)
            print(file)
        except:
            pass
        
    semibeta_df = pd.concat(semibeta_list)
    semibeta_df['Trddt'] = semibeta_df.Trddt.astype(str)
    semibeta_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    semibeta_df.dropna(inplace=True)
    semibeta_df.to_csv(data_dir + '\\data_sample\\IVOL\\IVOL_CH3.csv', index=False)


###############################################################################
## Basic data construction
###############################################################################


# Create the stocks daily basic data for further empirical study
def Crt_Stock_base_df():
         
    RSJ_RV = pd.read_csv(r'F:\RSJ_RV\RSJ_RV_5.csv')
    ConDisconBeta = pd.read_csv(r'F:\ConDisconBeta\ConDisconBeta_300_5.csv')
    BM_df = pd.read_csv(r'F:\数据集合\学术研究_股票数据\CSMAR\Combined_Data\BM.csv')  
       
    stock_day_trade_data = pd.read_csv(r'F:\数据集合\学术研究_股票数据\CSMAR\Combined_Data\SAVIC_saveMV_day.csv', usecols=['Stkcd', 'Trddt', 'Dretwd', 'Dsmvtll', 'Dnshrtrd', 'Adjprcnd','Clsprc'])
    stock_day_trade_data = stock_day_trade_data.sort_values(['Stkcd','Trddt'])
    stock_day_trade_data['ME'] = stock_day_trade_data['Dsmvtll'].values
    stock_day_trade_data['MOM'] = stock_day_trade_data.groupby('Stkcd').Adjprcnd.shift(252)/stock_day_trade_data.groupby('Stkcd').Adjprcnd.shift(21) - 1
    stock_day_trade_data['ILLIQ'] = abs(stock_day_trade_data['Dretwd'])/stock_day_trade_data['Dnshrtrd']/stock_day_trade_data['Clsprc']
    
    stock_day_trade_data['Weighted_Ret'] = stock_day_trade_data['ME'] * stock_day_trade_data['Dretwd']
    grouped = stock_day_trade_data.groupby('Trddt')
    market_returns = grouped['Weighted_Ret'].sum() / grouped['ME'].sum()
    stock_day_trade_data = pd.merge(stock_day_trade_data, market_returns.to_frame().rename(columns={0:'MarketPort'}),left_on='Trddt',right_index=True)
    stock_day_trade_data = stock_day_trade_data.sort_values(['Stkcd','Trddt']).reset_index(drop=True)  
    
    def Cpt_CSK_CKT(data, window=20):

        index_ret_mean = data['MarketPort'].rolling(window).mean()
        stock_ret_mean = data['Dsmvtll'].rolling(window).mean()
        index_demean = data['MarketPort']-index_ret_mean
        stock_demean = data['Dsmvtll']-stock_ret_mean

        # Calculate the common denominator component
        deno1 = pow(pow(stock_demean, 2).rolling(window).mean(), 0.5)
        deno2 = pow(index_demean, 2).rolling(window).mean()

        mole_csk = stock_demean*pow(index_demean, 2)
        CSK = mole_csk.rolling(window).mean()/deno1/deno2

        mole_ckt = stock_demean*pow(index_demean, 3)
        CKT = mole_ckt.rolling(window).mean()/deno1/pow(deno2, 1.5)
        return (CSK, CKT)
    
    CSK, CKT = Cpt_CSK_CKT(stock_day_trade_data, window=20)
    stock_day_trade_data['CSK'] = CSK
    stock_day_trade_data['CKT'] = CKT
    stock_day_trade_data = stock_day_trade_data.drop(['Weighted_Ret','MarketPort'], axis=1)
    
    stock_day_trade_data = pd.merge(stock_day_trade_data, RSJ_RV)
    stock_day_trade_data = pd.merge(stock_day_trade_data, ConDisconBeta)
    
    stock_day_trade_data['Trddt'] = pd.to_datetime(stock_day_trade_data.Trddt)
    stock_day_trade_data['Trdmnt'] = stock_day_trade_data['Trddt'].dt.strftime('%Y-%m')
    stock_day_trade_data = pd.merge(stock_day_trade_data,BM_df)
    stock_day_trade_data = stock_day_trade_data.drop('Trdmnt', axis=1)
    
    SVIC_df = pd.read_csv(r'F:\数据集合\学术研究_股票数据\多因子模型收益率\CH3_daily.csv')
    SVIC_df.columns = ['Trddt','rf','mkt','smb','vmg']
    SVIC_df['Trddt'] = pd.to_datetime(SVIC_df.Trddt)

    stock_base_df = pd.merge(stock_day_trade_data, SVIC_df)
    stock_base_df['ex_ret'] = stock_base_df.Dretwd - stock_base_df.rf
    
    IVOL = pd.read_csv(r'F:\IVOL\IVOL.csv')
    IVOL = IVOL[['Stkcd','Trddt','IVOL']]
    IVOL['Trddt'] = pd.to_datetime(IVOL['Trddt'])
    stock_base_df = pd.merge(stock_base_df, IVOL)

    # stock_base_df = stock_base_df.sort_values(['Stkcd','Trddt']).reset_index(drop=True)
    return stock_base_df.dropna()


# Create all interested variables and resample it into the final analyse frequency
def Crt_SortTable(stock_base_df, min_, index_type, est_intervel, freq='W'):
    
            
    DS_df = pd.read_csv(r'F:\SemiBeta\Intraday_betas\{0}\{1}_{2}.csv'.format(min_, index_type, est_intervel),index_col=0)
        
    DS_day_df = DS_df.unstack().reset_index().rename(columns={'level_0':'Stkcd',0:'Beta_abs_intra','level_1':'Trddt'})
    DS_day_df = DS_day_df.dropna()
    try:
        DS_day_df = DS_day_df.rename(columns={'datetime':'Trddt'})
    except:
        pass
    
    DS_day_df = DS_day_df.set_index('Trddt')
    DS_exec_df = DS_day_df.copy()
    DS_exec_df['Stkcd'] = DS_exec_df.Stkcd.astype(int)
    DS_exec_df.index = DS_exec_df.reset_index()['Trddt'].apply(lambda x: x[:10])
    DS_exec_df = DS_exec_df.reset_index()
    
    # ABS_df = pd.read_csv(r'F:\SemiBeta\Other_measure\ABS_{}_{}_{}.csv'.format(min_,index_type, est_intervel))
    Square_df = pd.read_csv(r'F:\SemiBeta\Other_measure\Square_{}_{}_{}.csv'.format(min_,index_type, est_intervel))
    SemiBeta = pd.read_csv(r'F:\SemiBeta\SemiBeta_{}_5.csv'.format(index_type))    
    AC_df = pd.read_csv(r'F:\SemiBeta\Other_measure\AutoCorr_{}_{}_{}.csv'.format(min_,index_type, est_intervel))
    BQ100_df = pd.read_csv(r'F:\SemiBeta\Other_measure\BQ100_{}_{}_{}.csv'.format(min_,index_type, est_intervel))
    
    DS_exec_df = pd.merge(DS_exec_df, Square_df)
    DS_exec_df = pd.merge(DS_exec_df, AC_df)
    DS_exec_df = pd.merge(DS_exec_df, BQ100_df)
    DS_exec_df = pd.merge(DS_exec_df, SemiBeta)
    
    DS_exec_df['Trddt'] = pd.to_datetime(DS_exec_df['Trddt'])
    DS_exec_df = pd.merge(stock_base_df, DS_exec_df, right_on=['Trddt','Stkcd'], left_on=['Trddt','Stkcd'])
    DS_exec_df = DS_exec_df[['Stkcd', 'Trddt', 'ex_ret', 'Dretwd',
                              'rf', 'mkt', 'vmg', 'smb',
                              'Beta_abs_intra','BQ100','AutoCorr','Square',
                              'beta_n','beta_mn','beta_p','beta_mp','beta',
                              'BM','ME', 'MOM', 'ILLIQ', 'IVOL','CSK','CKT',
                              'RSJ','conBeta', 'disconBeta']]
    DS_exec_df['AutoCorr'] = -DS_exec_df['AutoCorr']

    
    DS_exec_df['Beta_neg'] = DS_exec_df.beta_n - DS_exec_df.beta_mn
    DS_exec_df['Beta_pos'] = DS_exec_df.beta_p - DS_exec_df.beta_mp
    DS_exec_df['Beta_abs'] = -DS_exec_df['Beta_neg'] + DS_exec_df['Beta_pos']
    DS_exec_df['semi_beta_vari'] = pow(DS_exec_df['Beta_abs'],2)
    DS_exec_df['abs_semi_beta_vari'] = abs(DS_exec_df['Beta_abs'])

    Factors = DS_exec_df.groupby([pd.Grouper(key='Trddt',freq=freq)])[['mkt', 'vmg', 'smb']].mean().reset_index().dropna()
    DS_exec_df = DS_exec_df.drop(['mkt', 'vmg', 'smb'], axis=1)
    
    DS_fin_df = DS_exec_df.groupby(['Stkcd', pd.Grouper(key='Trddt',freq=freq)]).mean().drop(['Dretwd','ex_ret'],axis=1)
    REV_df = DS_exec_df.groupby(['Stkcd', pd.Grouper(key='Trddt',freq=freq)])[['Dretwd','ex_ret']].sum().rename(columns={'Dretwd': 'REV'})
    MAX_df = DS_exec_df.groupby(['Stkcd', pd.Grouper(key='Trddt',freq=freq)])[['Dretwd']].max().rename(columns={'Dretwd': 'MAX'})
    MIN_df = DS_exec_df.groupby(['Stkcd', pd.Grouper(key='Trddt',freq=freq)])[['Dretwd']].min().rename(columns={'Dretwd': 'MIN'})
    DS_fin_df = pd.concat([DS_fin_df, MAX_df, MIN_df,REV_df], axis=1)  
    DS_fin_df = DS_fin_df.dropna()

    # DS_fin_df = DS_fin_df.drop('Stkcd',axis=1).reset_index()
    DS_fin_df = DS_fin_df.reset_index()
    DS_fin_df = pd.merge(DS_fin_df, Factors).sort_values(['Stkcd','Trddt'])
    DS_fin_df['retShit'] = DS_fin_df.groupby('Stkcd').ex_ret.shift(-1)
    DS_fin_df['retShit'] = DS_fin_df['retShit'] * 100
    DS_fin_df = DS_fin_df.dropna()

    return DS_fin_df.reset_index(drop=True)


# Do data cleaning as well as creating the basic data of semi-beta variation estimation
def Cpt_BV_and_LogClsDiff(stkcd, hf_index, stock_base_data=None, n=48, min_=5, asset_type='stock'):

    try:
        hf_data_df = TB.Fetch_Stock_HFdata_from_Resset(stkcd, asset_type=asset_type, minType=min_)
        hf_data_df = hf_data_df.rename(columns={'date':'Trddt'})  
        if stock_base_data is not None:
            hf_data_df = pd.merge(hf_data_df, stock_base_data,
                                  left_on=['stkcd','Trddt'] ,right_on=['Stkcd','Trddt'])
        hf_data_df = hf_data_df[['Trddt','time','close','open']]
        hf_data_df = pd.merge(hf_data_df, hf_index, left_on=['Trddt','time'], right_on=['Trddt','time'])
        hf_data_df = hf_data_df.sort_values(['Trddt','time'])
        hf_data_df = hf_data_df[(hf_data_df['close'] != 0) & (hf_data_df['open'] != 0)]
        hf_data_df = hf_data_df[(hf_data_df['index_close'] != 0) & (hf_data_df['index_open'] != 0)]
        
        # calculate diff log price
        hf_data_df['log_close'] = np.log(hf_data_df['close'])
        hf_data_df['log_close_s1'] = hf_data_df.groupby('Trddt').log_close.shift(1)
        hf_data_df.loc[np.isnan(hf_data_df.log_close_s1),'log_close_s1'] = np.log(hf_data_df.loc[np.isnan(hf_data_df.log_close_s1),'open'])
        hf_data_df['log_close_diff'] = hf_data_df.log_close - hf_data_df.log_close_s1
        hf_data_df['log_close_diff_abs'] = abs(hf_data_df.log_close_diff)
        
        hf_data_df['log_close_index'] = np.log(hf_data_df['index_close'])
        hf_data_df['log_close_index_s1'] = hf_data_df.groupby('Trddt').log_close_index.shift(1)
        hf_data_df.loc[np.isnan(hf_data_df.log_close_index_s1),'log_close_index_s1'] = \
            np.log(hf_data_df.loc[np.isnan(hf_data_df.log_close_index_s1),'index_open'])
        hf_data_df['log_close_diff_index'] = hf_data_df.log_close_index - hf_data_df.log_close_index_s1
        hf_data_df['log_close_diff_abs_index'] = abs(hf_data_df.log_close_diff_index)
        
        
        # if some day have nan in log_close_diff than drop that day
        drop_date1 = hf_data_df[hf_data_df.log_close_diff.isna()].Trddt   
        hf_data_df = hf_data_df[~hf_data_df.Trddt.isin(drop_date1)]
                
        # if the percentage of some day's intraday zero return's number is bigger than 1/3
        # that drop that day
        drop_date3 = hf_data_df.groupby('Trddt').apply(lambda x:((x.log_close_diff>0) | (x.log_close_diff<0)).sum()<int(n*2/3))
        drop_date3 = drop_date3[drop_date3].index
        hf_data_df = hf_data_df[~hf_data_df.Trddt.isin(drop_date3)]

        full_trading_dates = hf_index.Trddt.unique()
        all_dates = pd.DataFrame({'Trddt': sorted(full_trading_dates)})
        all_times = pd.DataFrame({'time': hf_index.time.unique().tolist()})
        merged_df = pd.merge(all_dates, all_times, how='cross')
        hf_data_df = pd.merge(merged_df, hf_data_df, on=['time', 'Trddt'], how='left')
        
        # Creat dX and bVm
        log_close_diff_pivot = hf_data_df.pivot(index='time',columns='Trddt',values='log_close_diff')
        log_close_diff_pivot = log_close_diff_pivot.fillna(method='ffill')
        log_close_diff_pivot = log_close_diff_pivot.fillna(method='bfill')
        
        dz = log_close_diff_pivot.values
                    
        log_close_diff_pivot_index = hf_data_df.pivot(index='time',columns='Trddt',values='log_close_diff_abs_index')
        log_close_diff_pivot_index = log_close_diff_pivot_index.fillna(method='ffill')
        log_close_diff_pivot_index = log_close_diff_pivot_index.fillna(method='bfill')
        #######################################################################
        dX = log_close_diff_pivot_index.values
        
        print('finished:',stkcd)
        return [(dz, dX),stkcd]
    
    except:
        pd.DataFrame([stkcd]).to_csv('F:\SemiBeta\error\erro_{}.csv'.format(stkcd))
        print('erro_{}'.format(stkcd))


def Cpt_betas(qq, dX, dZ, n=48, kn=25):
    
    idxI = np.arange(qq - kn + 1, qq + 1)  # Corrected index calculation

    Vm = TB.nb_mean(dX[idxI, :]**2, axis=0)  # Corrected axis

    betas = TB.nb_mean(dZ[idxI, :] * dX[idxI, :], axis=0) / Vm  # Corrected axis

    return betas


def Cpt_IntraBeta_and_Measures(stkcd, hf_index, stock_base_data, kn, min_=5, n=48):
    
    '''
    
    hf_index = TB.Fetch_Stock_HFdata_from_Resset(300, asset_type='index', minType=min_) 
    hf_index = hf_index[['stkcd','Trddt','time','open','close']]
    hf_index = hf_index.rename(columns={'close':'index_close','open':'index_open'})
    
    '''

    # for stkcd in all_stk_list:
    dZ, dX = Cpt_BV_and_LogClsDiff(stkcd, hf_index, stock_base_data, n=n, min_=min_)[0]
    n, T = dX.shape
    betas = np.full((n, T), np.nan)
    
    for qq in range(kn-1,n):
        betas[qq,:] = Cpt_betas(qq, dX, dZ, n=n, kn=kn)    

    index_name = hf_index['stkcd'].iloc[0]
    
    betas_df = pd.DataFrame(betas,columns=hf_index.Trddt.unique(),index=hf_index.time.unique()).T
    betas_df.to_csv(r'F:\SemiBeta\Intraday_betas\{}\{}\{}\{}.csv'.format(min_,index_name,kn,stkcd))

    betas2 = pow(betas_df,2).mean(axis=1).rename(stkcd)
    bq_100 = (betas_df.max(axis=1) - betas_df.min(axis=1)).rename(stkcd)
    autocorr_1 = betas_df.T.apply(lambda x: x.autocorr(lag=1)).rename(stkcd)

    betas2.to_csv(r'F:\SemiBeta\Other_measure\Square\{}\{}\{}\{}.csv'.format(min_,index_name,kn,stkcd))
    bq_100.to_csv(r'F:\SemiBeta\Other_measure\BQ100\{}\{}\{}\{}.csv'.format(min_,index_name,kn,stkcd))
    autocorr_1.to_csv(r'F:\SemiBeta\Other_measure\AutoCorr\{}\{}\{}\{}.csv'.format(min_,index_name,kn,stkcd))
    
    print('Betas finished:{}_{}_{}'.format(stkcd,kn,index_name))
    


def Cpt_All_Stock_DS(index_lst, kn_lst, n=48, min_=5):
    
    # get high frequency stock data
    hf_index_300 = TB.Fetch_Stock_HFdata_from_Resset(300, asset_type='index', minType=min_) 
    hf_index_4000 = pd.read_csv(r'F:\HF_MIN\Resset\A_Index_{}.csv'.format(min_))

    hf_index_300 = hf_index_300[['stkcd','Trddt','time','open','close']]
    hf_index_300 = hf_index_300.rename(columns={'close':'index_close','open':'index_open'})
    hf_index_4000 = hf_index_4000.rename(columns={'close':'index_close','open':'index_open'})
    hf_index_4000['stkcd'] = 4000

    index_dict = {300:hf_index_300, 4000:hf_index_4000}
    
    stock_base_data = pd.read_csv(r'F:\数据集合\学术研究_股票数据\CSMAR\Combined_Data\SAVIC_saveMV_day.csv',usecols=['Stkcd','Trddt'])

    skip_dir1 = r'F:\SemiBeta\Other_measure\BQ100\{}'.format(min_)
    skip_list = TB.Tag_FilePath_FromDir(skip_dir1)
    
    print('gogogogogogogogogo')
    num_processes = 16

    # Create a Pool of processes
    with Pool(num_processes) as pool:
        for index_type in index_lst:
            hf_index = index_dict[index_type]
            stock_base_data_ = stock_base_data[stock_base_data.Trddt.isin(hf_index.Trddt)]
            all_stk_list = stock_base_data_.Stkcd.unique().tolist()

            for kn in kn_lst:
                for stkcd in all_stk_list:
                    file_name2 = r'F:\SemiBeta\Other_measure\BQ100\{}\{}\{}\{}.csv'.format(min_,index_type,kn,stkcd)
                    if file_name2 not in skip_list:
                        pool.apply_async(Cpt_IntraBeta_and_Measures, (stkcd, hf_index, stock_base_data, kn, min_, n))

        # Close the pool and wait for all processes to finish
        pool.close()
        pool.join()   
      
        
      
  
def Mrg_Intraday_Beta(min_, index_type, est_intervel):
    
    
    print('Begin merging intraday data: {}_{}_{}'.format(min_,index_type,est_intervel))
    beta_dispersion_dir = r'F:\SemiBeta\Intraday_betas\{0}\{1}\{2}'.format(min_,index_type,est_intervel)            
    file_list = TB.Tag_FilePath_FromDir(beta_dispersion_dir)
    DS_df = pd.DataFrame()
    for stock in file_list:
        try:
            stkcd = re.findall('(\d+).csv',stock)[0]
            stock_data = pd.read_csv(stock, index_col=0).T
            stock_data = stock_data.unstack().reset_index().rename(columns={'level_0':'Trddt','level_1':'time',0:stkcd})
            stock_data['datetime'] = stock_data.Trddt + ' ' + stock_data.time
            stock_data.index = pd.to_datetime(stock_data.datetime)
            stock_data[stkcd] = stock_data[stkcd].astype(float)
            try:
                stock_day_data = stock_data.resample('D').mean(numeric_only=True)[stkcd]
            except:
                stock_day_data = stock_data.resample('D').mean()[stkcd]
            DS_df = pd.concat([DS_df,stock_day_data],axis=1)
            print(stock)
        except:
            pass
    
    DS_df.to_csv(r'F:\SemiBeta\Intraday_betas\{0}\{1}_{2}.csv'.format(min_,index_type,est_intervel))
    print('Finshed merging intraday data: {}_{}_{}'.format(min_,index_type,est_intervel))


def Mult_Mrg_Intraday_Beta_res():
    
    num_processes = 16
    # Create a Pool of processes
    with Pool(num_processes) as pool:
        for min_ in ['5']:
            # for index_type in ['300','500','1000']:
            for index_type in ['4000']:
                # for est_intervel in list(range(20,45,5)): 
                for est_intervel in [5,15,20,30,35,40]: 

                    pool.apply_async(Mrg_Intraday_Beta, (min_, index_type, est_intervel,))

        # Close the pool and wait for all processes to finish
        pool.close()
        pool.join()


def Mrg_Beta_measure(D_type, index, min_, est_intervel):
    
    D_Ba_path = r'F:\SemiBeta\Other_measure\{}\{}\{}\{}'.format(D_type, min_, index, est_intervel)
    file_list = TB.Tag_FilePath_FromDir(D_Ba_path)
    measure_df_list = []

    for file in file_list:
        try:
            stkcd = file.split('\\')[-1].split('.')[0]
            measure_df = pd.read_csv(file, index_col=0).reset_index().rename(columns={'index':'Trddt',stkcd:D_type}).dropna()
            measure_df['Stkcd'] = stkcd
            measure_df_list.append(measure_df)
            print(stkcd)
        except:
            pass
        
    Measure_df = pd.concat(measure_df_list)
    Measure_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    Measure_df.dropna(inplace=True)
    
    Measure_df.to_csv(r'F:\SemiBeta\Other_measure\{}_{}_{}_{}.csv'.format(D_type, min_, index, est_intervel), index=False)


def Mult_Mrg_Beta_measure(D_type_lst, index_lst, min__lst, est_intervel_lst):
    
    num_processes = 16
    # Create a Pool of processes
    with Pool(num_processes) as pool:
        for D_type in D_type_lst:
            for min_ in min__lst:
                for index_type in index_lst:
                    # for est_intervel in list(range(20,45,5)): 
                    for est_intervel in est_intervel_lst: 
    
                        pool.apply_async(Mrg_Beta_measure, (D_type, index_type, min_, est_intervel,))
    
        # Close the pool and wait for all processes to finish
        pool.close()
        pool.join()
        

def Merge_BQ_and_AutoCorr(index,kn):
    
    D_Ba_path = r'F:\Intrady Beta Pattern\Betasabs_5\{}\{}'.format(index,kn)
    file_list = TB.Tag_FilePath_FromDir(D_Ba_path)
    
    bq100_list = []
    bq90_list = []
    bq75_list = []
    auto_corr_list1 = []

    for file in file_list:
        try:
            print(file)
            stkcd = file.split('\\')[-1].split('.')[0]
            betas_df = pd.read_csv(file, index_col=0)
            
            bq_100 = (betas_df.max(axis=1) - betas_df.min(axis=1)).to_frame().reset_index().rename(columns={'index':'Trddt',0:'bq100'})
            bq_100['Stkcd'] = stkcd
            bq_90 = (betas_df.quantile(0.9, axis=1) - betas_df.quantile(0.1, axis=1)).to_frame().reset_index().rename(columns={'index':'Trddt',0:'bq90'})
            bq_90['Stkcd'] = stkcd
            bq_75 = (betas_df.quantile(0.75, axis=1) - betas_df.quantile(0.25, axis=1)).to_frame().reset_index().rename(columns={'index':'Trddt',0:'bq75'})
            bq_75['Stkcd'] = stkcd

            autocorr_1 = betas_df.T.apply(lambda x: x.autocorr(lag=1)).to_frame().reset_index().rename(columns={'index':'Trddt',0:'AC1'})
            autocorr_1['Stkcd'] = stkcd
            
            
            bq100_list.append(bq_100)
            bq90_list.append(bq_90)
            bq75_list.append(bq_75)
            auto_corr_list1.append(autocorr_1)

        except:
            print('error: {}'.format(file))
            pass
    
    measure_dict = {'BQ100':bq100_list,
                    'BQ90':bq90_list,
                    'BQ75':bq75_list,
                    'AC1':auto_corr_list1}
    
    for measure,measure_list in measure_dict.items():
        measure_df = pd.concat(measure_list)
        measure_df.replace([np.inf, -np.inf], np.nan, inplace=True)
        measure_df.dropna(inplace=True)
        measure_df.to_csv(r'F:\SemiBeta\Other_measure\{}_{}_{}.csv'.format(measure, index, kn), index=False)
    

def Mult_Merge_BQ_and_AutoCorr():
    
    num_processes = 16
    # Create a Pool of processes
    with Pool(num_processes) as pool:
        for index_type in ['300']:
            for est_intervel in [15,20,25,30,35,40]: 
                pool.apply_async(Merge_BQ_and_AutoCorr, (index_type, est_intervel,))

        # Close the pool and wait for all processes to finish
        pool.close()
        pool.join()





###############################################################################
## Do double and single sort
############################################################################### 
    
def Exec_TD_SSort(SSort_exec_df, min_, index_type, est_window, key, freq='W', weight_type='vw'):
    
    APT = AssetPricingTool()
    
    Factors = SSort_exec_df[['Trddt','mkt', 'vmg', 'smb']].set_index('Trddt').drop_duplicates()
    if weight_type.lower() == 'vw':
        SSort_res = APT.Rec_SingleSortRes(avgTag='retShit',sortTag=key,groupsNum=5,Factors=Factors, timeTag='Trddt',
                                          use_factors=['mkt', 'vmg', 'smb'],weightTag='ME',df=SSort_exec_df.copy())
    else:
        SSort_res = APT.Rec_SingleSortRes(avgTag='retShit',sortTag=key,groupsNum=5,Factors=Factors, timeTag='Trddt',
                                          use_factors=['mkt', 'vmg', 'smb'],df=SSort_exec_df.copy())
    
    SSort_res.to_csv(r'F:\SemiBeta\Sorted_res\Ssort\{}\{}\{}\Ssort_{}_{}_{}.csv'.format(weight_type, index_type, est_window, freq, min_,key))

    print('Finshed Single sort result calculation: {}_{}_{}_{}_{}'.format(freq, min_, index_type, weight_type, key))

    
    
def Muti_exec_Ssort(weight_lst, index_lst, est_lst, D_lst, min_=5, freq='W', mult=False):
    
    skip_dir1 = r'F:\SemiBeta\Sorted_res\Ssort'
    skip_list = TB.Tag_FilePath_FromDir(skip_dir1)
        
    print('Exec stock_base_df')
    stock_base_df = Crt_Stock_base_df()

    # SSort_exec_df = pd.read_csv(r'F:\SemiBeta\Basic_data\{}_{}_{}.csv'.format(5, 300, 'W'))

    if mult:
        num_processes = 4
        # Create a Pool of processes
        with Pool(num_processes) as pool:
            for index_type in index_lst:
                print('Exec SSort_exec_df')
                for est_window in est_lst:
                    SSort_exec_df = Crt_SortTable(stock_base_df, min_, index_type, est_window, freq=freq) 
                    for weight_type in weight_lst:
                        for key_tag in D_lst:
                            file_name = skip_dir1 + '\\{}\\{}\\{}\\Ssort_{}_{}_{}.csv'.format(weight_type,index_type,est_window, freq, min_, key_tag)
                            if file_name not in skip_list:
                                pool.apply_async(Exec_TD_SSort, (SSort_exec_df, min_, index_type, est_window,key_tag,
                                                                     freq, weight_type))

            # Close the pool and wait for all processes to finish
            pool.close()
            pool.join()



    
  
def Exec_TD_Dsort(DS_exec_df,
                  min_,
                  index_type, 
                  key_tag,
                  con_tag,
                  est_window,
                  freq='W',
                  weight_type='vw',
                  reverse=False):
    
    # ME MOM REV IVOL ILLIQ MAX MIN

    APT = AssetPricingTool()
    
    Factors = DS_exec_df[['Trddt','mkt', 'vmg', 'smb']].set_index('Trddt').drop_duplicates()
    if not reverse:
        if weight_type.lower() == 'vw':
            DSort_res = APT.Rec_DoubleSortRes(avgTag='retShit', sortTag=con_tag, sortTag_key=key_tag, groupsNum1=5, groupsNum2=5, 
                                              SortMethod='dependent', Factors=Factors, use_factors=['mkt', 'vmg', 'smb'],
                                              timeTag='Trddt', df=DS_exec_df.copy(), weightTag='ME')
        else:
            DSort_res = APT.Rec_DoubleSortRes(avgTag='retShit', sortTag=con_tag, sortTag_key=key_tag, groupsNum1=5, groupsNum2=5, 
                                              SortMethod='dependent', Factors=Factors, use_factors=['mkt', 'vmg', 'smb'],
                                              timeTag='Trddt', df=DS_exec_df.copy())
        
        DSort_res.to_csv(r'F:\SemiBeta\Sorted_res\Dsort\{}\{}\{}\{}\Dsort_{}_{}_{}_{}_{}.csv'.format(weight_type, index_type,est_window,reverse, freq, min_,  
                                                                                           weight_type, con_tag, key_tag))
        
    else:
        
        if weight_type.lower() == 'vw':
            DSort_res = APT.Rec_DoubleSortRes(avgTag='retShit', sortTag=key_tag, sortTag_key=con_tag, groupsNum1=5, groupsNum2=5, 
                                              SortMethod='dependent', Factors=Factors, use_factors=['mkt', 'vmg', 'smb'],
                                              timeTag='Trddt', df=DS_exec_df.copy(), weightTag='ME')
        else:
            DSort_res = APT.Rec_DoubleSortRes(avgTag='retShit', sortTag=key_tag, sortTag_key=con_tag, groupsNum1=5, groupsNum2=5, 
                                              SortMethod='dependent', Factors=Factors, use_factors=['mkt', 'vmg', 'smb'],
                                              timeTag='Trddt', df=DS_exec_df.copy())
        
        DSort_res.to_csv(r'F:\SemiBeta\Sorted_res\Dsort\{}\{}\{}\{}\Dsort_{}_{}_{}_{}_{}.csv'.format(weight_type, index_type,est_window,reverse, freq, min_,  
                                                                                           weight_type, con_tag, key_tag))

    print('Finshed Single sort result calculation: Dsort_{}_{}_{}_{}_{}_{}_{}.csv'.format(freq, min_,  index_type, 
                                                                                       weight_type, con_tag, key_tag,reverse))
     
    
def Muti_Exec_TD_Dsort(weight_lst, key_lst, index_lst, con_lst, est_lst, min_=5, freq='W',mult=False,reverse=False):
            
    skip_dir1 = r'F:\SemiBeta\Sorted_res\Dsort'
    skip_list = TB.Tag_FilePath_FromDir(skip_dir1)
    
    print('Exec stock_base_df')
    stock_base_df = Crt_Stock_base_df()

    # SSort_exec_df = pd.read_csv(r'F:\SemiBeta\Basic_data\{}_{}_{}.csv'.format(5, 300, 'W'))

    if mult:
        num_processes = 3
        # Create a Pool of processes
        with Pool(num_processes) as pool:
            for index_type in index_lst:
                print('Exec SSort_exec_df')
                for est_window in est_lst:
                    SSort_exec_df = Crt_SortTable(stock_base_df, min_, 5, index_type, est_window, freq=freq) 
                    for weight_type in weight_lst:
                        for key_tag in key_lst:
                            for con_tag in con_lst:
                                file_name = skip_dir1 + '\\{}\\{}\\{}\\{}\\Dsort_{}_{}_{}_{}_{}.csv'.format(weight_type,index_type,est_window,reverse, freq, min_, 
                                                                                                       weight_type, con_tag, key_tag)
                                if file_name not in skip_list:
                                    pool.apply_async(Exec_TD_Dsort, (SSort_exec_df, min_, index_type, key_tag,
                                                                      con_tag, est_window, freq, weight_type, reverse,))
    
            # Close the pool and wait for all processes to finish
            pool.close()
            pool.join()
    else:
        for index_type in index_lst:
            for est_window in est_lst:
                SSort_exec_df = Crt_SortTable(stock_base_df, min_, 5, index_type, est_window, freq=freq) 

                for weight_type in weight_lst:
                    for key_tag in key_lst:
                        for con_tag in con_lst:
                            file_name = skip_dir1 + '\\{}\\{}\\{}\\{}\\Dsort_{}_{}_{}_{}_{}.csv'.format(weight_type,index_type,est_window,reverse, freq, min_, 
                                                                                                   weight_type, con_tag, key_tag)
                            if file_name not in skip_list:
                                Exec_TD_Dsort(SSort_exec_df, min_, index_type, key_tag,
                                                                  con_tag, est_window, freq, weight_type, reverse)
    


def Mrg_Dsort_res_(freq, min_, control_list, est_lst, index_type, key_tag, weight_type, reverse=False):
    res_dir = r'F:\SemiBeta\Sorted_res\Dsort'
    
    for est_intervel in est_lst:
        Dsort_res = pd.DataFrame()
        for con_tag in control_list:
            file = res_dir + '\\{}\\{}\\{}\\{}\\Dsort_{}_{}_{}_{}_{}.csv'.format(weight_type, index_type, est_intervel,reverse, freq, min_, 
                                                                       weight_type, con_tag, key_tag)
            # df = pd.read_csv(file, usecols=['Group_DS', 'AvgRet','Alpha'],index_col=0)
            df = pd.read_csv(file,index_col=0)
            df.index = df.index.str.replace(r'.*_(G\d+)', r'\1', regex=True)
            df.index.name = 'Group'
            df = df[['AvgRet','Alpha']]
    
            avg_p = df.loc['p','AvgRet']
            alpha_p = df.loc['p','Alpha']
            
            if (avg_p <= 0.1) & (avg_p > 0.05):
                df.loc['HML','AvgRet'] = str(df.loc['HML','AvgRet']) + '*'
            elif (avg_p <= 0.05) & (avg_p > 0.01):
                df.loc['HML','AvgRet'] = str(df.loc['HML','AvgRet']) + '**'
            elif avg_p <= 0.01:
                df.loc['HML','AvgRet'] = str(df.loc['HML','AvgRet']) + '***'
                
            if (alpha_p <= 0.1) & (alpha_p > 0.05):
                df.loc['HML','Alpha'] = str(df.loc['HML','Alpha']) + '*'
            elif (alpha_p <= 0.05) & (alpha_p > 0.01):
                df.loc['HML','Alpha'] = str(df.loc['HML','Alpha']) + '**'
            elif alpha_p <= 0.01:
                df.loc['HML','Alpha'] = str(df.loc['HML','Alpha']) + '***'
    
                
            df = df.drop('p')
            df.loc['','AvgRet'] = ''
            df.loc['Alpha','AvgRet'] = df.loc['HML','Alpha']
            df.loc['Alpha_t','AvgRet'] = df.loc['t','Alpha']
            df = df.drop('Alpha', axis=1).rename(columns={'AvgRet':con_tag})
                
                        
            Dsort_res = pd.concat([Dsort_res, df],axis=1)
            
        Dsort_res.to_csv('F:\SemiBeta\Sorted_res\Merge_Dsort_{}_{}_{}_{}_{}_{}_{}.csv' \
                         .format(freq, min_, index_type, est_intervel, weight_type, key_tag,reverse))



def Exec_FamaMacbeth_Reg(SSort_exec_df, key_x_lst, index_type, min_=5, est_intervel=25, freq='W'):
    
    APT = AssetPricingTool()
        
    yTag = 'retShit'    
    # DS_exec_df = Crt_TD_SortTable(D_type, min_, index_type, est_intervel, stock_base_df, freq) 
    #  , 'beta','RSJ', 'conBeta', 'disconBeta' 'RSJ','beta', 'beta_n','disconBeta' 'Beta_abs',
    key_xTag_lst = key_x_lst + ['beta','Beta_neg']
    con_xTag_lst = ['RSJ', 'disconBeta', 'REV', 'BM', 'ME', 'MOM', 'ILLIQ', 
                    'IVOL', 'MAX', 'MIN','CSK','CKT']
    
    reg_lst = []
    for x in key_xTag_lst + con_xTag_lst:
        reg_lst.append([yTag] + [x])
    
    if len(key_xTag_lst) > 1:
        reg_lst.append([yTag] + key_xTag_lst)
                
        for key_xTag in key_xTag_lst:
            reg_lst.append([yTag] + [key_xTag] + con_xTag_lst)

        
    for con_xTag in con_xTag_lst:
        reg_lst.append([yTag] + key_xTag_lst + [con_xTag])
    reg_lst.append([yTag] + key_xTag_lst + con_xTag_lst)

    data_df = SSort_exec_df.drop(['Stkcd','retShit'], axis=1).copy()
    data_df = SSort_exec_df[['Trddt']+key_xTag_lst+con_xTag_lst]
    data_df = data_df.groupby('Trddt').apply(lambda x:TB.z_score(x))
    data_df = data_df.apply(lambda x: np.select([x.values<x.quantile(0.005),x.values>x.quantile(0.995)], [x.quantile(0.005),x.quantile(0.995)], default=x))

    data_df[['Stkcd','Trddt','retShit']] = SSort_exec_df[['Stkcd','Trddt','retShit']]
    data_df['Trddt'] = pd.to_datetime(data_df.Trddt)
    data_df = data_df.set_index(['Stkcd','Trddt'])
    data_df = data_df[[yTag] + key_xTag_lst + con_xTag_lst]
    
    regres = APT.FamaMacBeth_summary(data_df, reg_lst, key_xTag_lst + con_xTag_lst).astype(str)
    regres.to_csv(r'F:\SemiBeta\FMreg_res\{}\{}\{}\FMR_{}.csv'.format(min_, index_type, est_intervel, key_x_lst))
        
    print('Finshed Fama-Macbech regression result calculation: {}_{}_{}_{}'.format(key_x_lst, min_, index_type, est_intervel))
    return regres
    

def Muti_exec_FMR(min_, freq_lst, est_lst, index_lst=[300,4000], mult=True):
    
    stock_base_df = Crt_Stock_base_df()

    skip_dir1 = r'F:\SemiBeta\FMreg_res'
    skip_list = TB.Tag_FilePath_FromDir(skip_dir1)
    
    if mult:
        num_processes = 4
        # Create a Pool of processes
        with Pool(num_processes) as pool:
            for freq in freq_lst:
                for est_intervel in est_lst:
                    for index in index_lst: 
                        SSort_exec_df = Crt_SortTable(stock_base_df, min_, index, est_intervel, freq=freq) 
                        print('finished SSort_exec_df_{}'.format(index))
                        for key_x_lst in [['Beta_abs_intra','Square'],
                                          ['Beta_abs_intra','AutoCorr'],
                                          ['Beta_abs_intra','BQ100']]:
                            file_name = skip_dir1 + '\\{}\\{}\\{}\\FMR_{}.csv'.format(min_, index, est_intervel, key_x_lst)
                            if file_name not in skip_list:
                                pool.apply_async(Exec_FamaMacbeth_Reg, (SSort_exec_df,  key_x_lst, index, min_, est_intervel, freq))
        
                # Close the pool and wait for all processes to finish
                pool.close()
                pool.join()
            

def Bet_on_BetaDispersion(SSort_exec_df, min_, index_type, freq, rho=0.002):    
    
    APT = AssetPricingTool()
    sortTag_list = ['beta','Beta_neg','Beta_abs_intra', 'Square', 'BQ100', 'AutoCorr']
    # 'RSJ','beta','Beta_neg','disconBeta','Beta_abs','DS'
    
    Stragte_res = pd.DataFrame(index=['Avg ret', 'Avg ret_t', 'Std dev', 'SR', '',
                                      'beta_mkt', 'beta_mkt_t', 'beta_vmg', 'beta_vmg_t', 
                                      'beta_smb', 'beta_smb_t', 
                                      'Alpha', 'Alpha_t', '','Adj R2'])
    
    plt.figure(figsize=(10, 8),dpi=150)
    line_styles = ['-', '--', '-.', ':','dashdot', 'dotted']
    for j in range(len(sortTag_list)):
        
        sortTag = sortTag_list[j]
        Factors = SSort_exec_df[['Trddt','mkt', 'vmg', 'smb']].set_index('Trddt').drop_duplicates()
        Sort_table = APT.Exec_SingleSort(avgTag='retShit', sortTag=sortTag, groupsNum=5, timeTag='Trddt', df=SSort_exec_df.copy())[1]
        
        Sort_table['retShit'] = Sort_table['retShit']/100
        
        # value weighted
        # weight = Sort_table.groupby(['Trddt','Group']).ME.sum().rename('TTL_ME')
        # Sort_table = pd.merge(Sort_table, weight,left_on=['Trddt','Group'], right_index=True).copy()
        # Sort_table['weight'] = Sort_table['ME']/Sort_table['TTL_ME']
        
        # equal weighted
        weight = 1/Sort_table.groupby(['Trddt','Group']).ME.count().rename('weight')
        Sort_table = pd.merge(Sort_table, weight,left_on=['Trddt','Group'], right_index=True).copy()
        Sort_table['ret_'] = Sort_table['retShit'] * Sort_table['weight']
        
        Top_group = Sort_table[Sort_table.Group=='{}_G05'.format(sortTag)]
        Bot_group = Sort_table[Sort_table.Group=='{}_G01'.format(sortTag)]
        Top_group = Top_group.sort_values(['Stkcd','Trddt'])
        Bot_group = Bot_group.sort_values(['Stkcd','Trddt'])
        
        Top_group['weight_s1'] = Top_group.groupby('Stkcd').weight.shift(1).fillna(0)
        Top_group['weight_del'] = abs(Top_group['weight'] - Top_group['weight_s1'])
        Top_group['TC'] = Top_group['weight_del']*(1 + Top_group['retShit'])
        Top = Top_group.groupby('Trddt')[['TC','ret_']].sum()
        
        Bot_group['weight_s1'] = Bot_group.groupby('Stkcd').weight.shift(1).fillna(0)
        Bot_group['weight_del'] = abs(Bot_group['weight'] - Bot_group['weight_s1'])
        Bot_group['TC'] = Bot_group['weight_del']*(1 + Bot_group['retShit'])
        Bot = Bot_group.groupby('Trddt')[['TC','ret_']].sum()
        HML = (Top['ret_'] - Bot['ret_'])
        
        if HML.mean()>0:
            HML = HML
        else:
            HML = -HML
            
        HML_adj = HML - rho*(Bot['TC'] + Top['TC'])
        HML_cumpord = (HML_adj+1).cumprod()
        # HML_cumpord = HML_adj.cumsum() + 1
        
        plt.plot(HML_cumpord.index, HML_cumpord.values, label=sortTag, linestyle=line_styles[j])
        
        Stragte_res.loc['Avg ret', sortTag] = HML_adj.mean()
        Stragte_res.loc['Avg ret_t', sortTag] = APT.Exec_NWTest(HML_adj.values)['tstat']
        Stragte_res.loc['Std dev', sortTag] = HML_adj.std()
        Stragte_res.loc['SR', sortTag] = Stragte_res.loc['Avg ret', sortTag]/Stragte_res.loc['Std dev', sortTag]
        
        
        reg_df = pd.merge(HML_adj.rename('hml_ret').to_frame(), Factors.shift(-1).dropna(), right_index=True, left_index=True)
        
        ols = sm.OLS(reg_df['hml_ret'], sm.add_constant(reg_df[['mkt', 'vmg', 'smb']])).fit()
        Stragte_res.loc['Adj R2', sortTag] = ols.rsquared_adj
        
        regRes = APT.Cpt_ResAndBeta(['mkt', 'vmg', 'smb'], 
                                    'hml_ret', df=reg_df, NWtest=True)
        
        strategy_index = Stragte_res.index
        for i in range(regRes['beta'].shape[0]):
            Stragte_res.loc[strategy_index[i*2+5], sortTag] = regRes['beta'][i]
            Stragte_res.loc[strategy_index[i*2+6], sortTag] = regRes['beta'][i]/np.sqrt(regRes['NWse'][i,i])
    Stragte_res.index = ['Avg ret', '', 'Std dev', 'SR', '',
                         'beta_mkt', '', 'beta_vmg', '', 
                         'beta_smb', '', 
                         'Alpha', '', '','Adj R2']
    Stragte_res.astype(float).round(4).to_csv(r'F:\SemiBeta\Strategy_table_{}_{}_{}_{}.csv' \
        .format(min_, index_type, freq,  int(rho*1000)))
    plt.xlabel('Trading Date')
    plt.ylabel('Net Strategy Value')
    plt.title('Net value of betting on different beta')
    plt.legend()
    plt.grid()
    plt.xticks(range(0,len(HML_cumpord.index),50),HML_cumpord.index[range(0,len(HML_cumpord.index),50)],rotation=45)
    plt.savefig(r'F:\SemiBeta\Strategy_plot_{}_{}_{}_{}.png' \
       .format(min_, index_type, freq, int(rho*1000)))
                
    
def Muti_Bet_on_BetaDispersion(min_, freq_lst, index, est_lst, rho=0.002, mult=False):
        
    skip_dir1 = r'F:\Intrady Beta Pattern\Betting on Strategy res\Strategy_table'
    skip_list = TB.Tag_FilePath_FromDir(skip_dir1)
    stock_base_df = Crt_Stock_base_df(index)

    if mult:
        num_processes = 2
        # Create a Pool of processes
        with Pool(num_processes) as pool:
            for freq in freq_lst:
                for est_intervel in est_lst:
                    SSort_exec_df = Crt_SortTable(stock_base_df, min_, index, est_intervel, freq=freq) 
                    file_name = skip_dir1 + '\\Strategy_table_{}_{}_{}_{}_{}.csv' \
                        .format(min_, index, est_intervel, freq, int(rho*1000))
                    if file_name not in skip_list:
                        pool.apply_async(Bet_on_BetaDispersion, (SSort_exec_df, min_, index, freq, rho, ))

        # Close the pool and wait for all processes to finish
        pool.close()
        pool.join()
        
    else:
        for freq in freq_lst:
            for est_intervel in est_lst:
                SSort_exec_df = Crt_SortTable(stock_base_df, min_, index, est_intervel, freq=freq) 
                file_name = skip_dir1 + '\\Strategy_table_{}_{}_{}_{}_{}.csv' \
                    .format(min_, index, est_intervel, freq, int(rho*1000))
                if file_name not in skip_list:
                    Bet_on_BetaDispersion(SSort_exec_df, min_, index, freq, rho, )
    
    
if __name__ == '__main__':
    
    data_dir = os.getcwd()
    
    ###########################################################################
    ## 1. Create basic dir to store the all kind of data
    ###########################################################################
    # Create folders for sotring the intraday semi-beta variation estimating results
    folders_dict = {str(year):{type_1:{type_:'' for type_ in ['15','20','25','30','35','40']} for type_1 in ['4000','300'] } for year in [5]}
    TB.create_folders(data_dir + '\\data_sample\\Intraday_betas', folders_dict)
    
    # Create folders for sotring the dispersion of intraday semi-beta variation estimating results
    folders_dict = {measure:{str(year):{type_1:{type_:'' for type_ in ['15','20','25','30','35','40']} for type_1 in ['4000','300'] } for year in [5]} for measure in ['BQ100','AutoCorr','Square']}
    TB.create_folders(data_dir + '\\data_sample\\Other_measure', folders_dict)

    # Create folders to store stocks' continue and discontinue beta
    folders_dict = {str(year):{type_1:'' for type_1 in ['300']} for year in [5]}
    TB.create_folders(data_dir + '\\data_sample\\ConDisconBeta', folders_dict)

    # Create folders to store stocks' semi-betas
    folders_dict = {str(year):{type_1:'' for type_1 in ['300','4000']} for year in [5]}
    TB.create_folders(data_dir + '\\data_sample\\SemiBeta\\Betas', folders_dict)
    
    # Create folders to store stocks' relative sign jump that constructed by rv
    folders_dict = {str(year):'' for year in [5]}
    TB.create_folders(data_dir + '\\data_sample\\RSJ_RV', folders_dict)
    
    # Create folders to store stocks' IVOL
    folders_dict = {str(year):'' for year in ['CH3','CH4','FF3','FF5']}
    TB.create_folders(data_dir + '\\data_sample\\IVOL', folders_dict)


    ###########################################################################
    ## 2. Create all kind of basic data
    ###########################################################################
    
    # # Create the continue and discontinue beta
    # Muti_Exec_ConDisconBeta(idxcd=300, min_=5,mult=False)
    # Mrg_ConDisconBeta(index=300, min_=5)
    
    # # Create the semi-betas
    # Muti_Exec_SemiBeta(mult=False)
    # Mult_Mrg_SemiBeta([300, 4000], [5], mult=False)   
    
    # # Create the RSJ
    # Muti_Exec_RSJ_RV(mult=False)
    # Mrg_RSJ_RV(5)
    
    # # Create IVOL that based on CH3 factor model
    # Mult_Cpt_IVOL(mult=False)
    # Mrg_IVOL()
    
    
    

    # # 1. Calculate intraday semi-beta variation and its dispersion
    # Cpt_All_Stock_DS([300,4000], [15,20,25,30,35,40], n=48, min_=5)
    
    # # 2. Merge different stock's mean of intraday semi-beta varitions as well as its three different
    # Mult_Mrg_Intraday_Beta_res()
    # Mult_Mrg_Beta_measure(['AutoCorr'], [300,4000],[5],[15,20,25,30,35,40])

    # # 3. Mult generate single sort results
    # Muti_exec_Ssort(['ew'],[300,4000], ['15','20','25','30'], ['Square','Beta_abs_intra','BQ100','AutoCorr'], min_=5, mult=True)

    # # 4. Mult generate double sort results    
    # control_list = ['BM','ME','MOM','REV','IVOL','ILLIQ','MAX','MIN','CSK','CKT','RSJ','beta','Beta_neg','disconBeta']
    # for key_tag in ['Square','Beta_abs_intra','BQ100','AutoCorr']:
    #     for weight_type in ['ew']:
    #         Mrg_Dsort_res_('W', 5, control_list,[15,20,25,30], 300, key_tag, weight_type, reverse=False)
    #         Mrg_Dsort_res_('W', 5, control_list,[15,20,25,30], 4000, key_tag, weight_type, reverse=False)

    
    # # 5. Mult generate fama-macbeth regression results    
    # Muti_exec_FMR(5, ['W'], [15,20,25,30,35,40], index_lst=[300,4000], mult=True)    
    
    # # 6. Mult generate beting beta strategy results
    # Muti_Bet_on_BetaDispersion(5, ['W'], 300, [15,20,25,30,35,40])    
    # Muti_Bet_on_BetaDispersion(5, ['W'], 4000, [15,20,25,30,35,40])    



