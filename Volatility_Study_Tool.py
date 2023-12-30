# -*- coding: utf-8 -*-
"""
Created on Mon Jul 10 15:16:18 2023

@author: asus
"""

import numpy as np
import pandas as pd
import scipy
import ToolBox as TB
from matplotlib import pyplot as plt
import re
from AssetPricing_test import DataCleaner
from multiprocessing import Pool, cpu_count
from mat4py import loadmat
from AssetPricing_test import DataCombiner_CSMAR
from scipy.io import savemat


class Volatility_Tool():
    
    def Cpt_HF_LogReturn(self, hf_data_df_, closeTag='close', openTag='open', n=48, drop0=False):
        '''
    
        >>> hf_data_df_ = TB.Fetch_Stock_HFdata_from_Resset(1)
        >>> self = Volatility_Tool()
        '''
        

        hf_data_df = hf_data_df_.copy()     
        log_closeTag = 'log_' + closeTag
        log_closeTag_s1 = log_closeTag + '_s1'
        log_closeTag_diff = log_closeTag + '_diff'
        
        # variance covariance matrix estimation
        hf_data_df = hf_data_df[(hf_data_df[closeTag] > 0) & (hf_data_df[openTag] > 0)]
        hf_data_df[log_closeTag] = np.log(hf_data_df[closeTag])
        hf_data_df[log_closeTag_s1] = hf_data_df.groupby('Trddt')[log_closeTag].shift(1)
        hf_data_df.loc[np.isnan(hf_data_df[log_closeTag_s1]), log_closeTag_s1] = np.log(hf_data_df.loc[np.isnan(hf_data_df[log_closeTag_s1]), openTag])
        hf_data_df[log_closeTag_diff] = hf_data_df[log_closeTag] - hf_data_df[log_closeTag_s1]
        hf_data_df.replace([np.inf, -np.inf], np.nan, inplace=True)
        hf_data_df.dropna(inplace=True)
        
        
        if drop0:
            # if the percentage of some day's intraday zero return's number is bigger than 1/3
            # that drop that day
            drop_date3 = hf_data_df.groupby('Trddt').apply(lambda x:((x.log_close_diff>0) | (x.log_close_diff<0)).sum()<int(n*2/3))
            drop_date3 = drop_date3[drop_date3].index
            hf_data_df = hf_data_df[~hf_data_df.Trddt.isin(drop_date3)]


        return hf_data_df
    
    
    def Cpt_RM(self, hf_data_df_, retTag, dayTag='Trddt', minTag='time', rm=None):
       
        hf_data_df = hf_data_df_.copy()
        n = len(hf_data_df[minTag].unique())
        origin_columns = list(hf_data_df.columns)
        
        hf_data_df['ret2'] = pow(hf_data_df[retTag],2)
        hf_data_df['ret3'] = pow(hf_data_df[retTag],3)
        hf_data_df['ret4'] = pow(hf_data_df[retTag],4)
        hf_data_df['ret_s1'] = hf_data_df.groupby(dayTag)[retTag].shift(1).fillna(0)
        hf_data_df['retxret_s1'] = abs(hf_data_df.ret_s1 * hf_data_df[retTag])
        
        RV = hf_data_df.groupby(dayTag).apply(lambda df: df['ret2'].sum()).rename('rv') 
        BV = np.pi / 2 * hf_data_df.groupby(dayTag).apply(lambda df: df['retxret_s1'].sum()).rename('bv') 
        RSK = np.sqrt(n) * hf_data_df.groupby(dayTag).apply(lambda df: df['ret3'].sum()).rename('rsk') / pow(RV,3/2).values
        RKT = n * hf_data_df.groupby(dayTag).apply(lambda df: df['ret4'].sum()).rename('rkt') / pow(RV,2).values

        hf_data_df = pd.merge(hf_data_df, pd.concat([RV,BV,RSK,RKT], axis=1), left_on=dayTag, right_index=True)
        
        if rm is None:
            return hf_data_df.drop(['ret2','ret3','ret4','ret_s1','retxret_s1'],axis=1)
        else:
            return hf_data_df[origin_columns + rm]
        
    
    def Cpt_RM_OK(self, hf_data_df_, retTag, dayTag='Trddt', minTag='time', rm=None):
       
        hf_data_df = hf_data_df_.copy()
        n = len(hf_data_df[minTag].unique())
        origin_columns = list(hf_data_df.columns)
        
        hf_data_df['ret2'] = pow(hf_data_df[retTag],2)
        hf_data_df['ret3'] = pow(hf_data_df[retTag],3)
        hf_data_df['ret4'] = pow(hf_data_df[retTag],4)
        hf_data_df['ret_s1'] = hf_data_df.groupby(dayTag)[retTag].shift(1).fillna(0)
        hf_data_df['retxret_s1'] = abs(hf_data_df.ret_s1 * hf_data_df[retTag])
        hf_data_df['ok'] = (0.811*(np.log(hf_data_df.high) - np.log(hf_data_df.low)) - \
                           0.369*abs(np.log(hf_data_df.close) - np.log(hf_data_df.open)))/np.sqrt((1/n))
        hf_data_df['ok2'] = pow(hf_data_df['ok'],2)
        
        RV = hf_data_df.groupby(dayTag).apply(lambda df: df['ret2'].sum()).rename('rv') 
        BV = np.pi / 2 * hf_data_df.groupby(dayTag).apply(lambda df: df['retxret_s1'].sum()).rename('bv') 
        RSK = np.sqrt(n) * hf_data_df.groupby(dayTag).apply(lambda df: df['ret3'].sum()).rename('rsk') / pow(RV,3/2).values
        RKT = n * hf_data_df.groupby(dayTag).apply(lambda df: df['ret4'].sum()).rename('rkt') / pow(RV,2).values
        OK = hf_data_df.groupby(dayTag).ok.mean().rename('OK') 
        OK2 = hf_data_df.groupby(dayTag).ok2.mean().rename('OK2') 
        RQ = n / 3 * hf_data_df.groupby(dayTag).apply(lambda df: df['ret4'].sum()).rename('RQ')
        
        hf_data_df = pd.merge(hf_data_df, pd.concat([OK,OK2,RV,BV,RSK,RKT,RQ], axis=1), left_on=dayTag, right_index=True)
        hf_data_df['RVsqrt'] = np.sqrt(hf_data_df['rv'])
        hf_data_df['RQsqrt'] = np.sqrt(hf_data_df['RQ'])
        hf_data_df['RQsqrt2'] = np.sqrt(hf_data_df['RQsqrt'])

        
        if rm is None:
            return hf_data_df.drop(['ok','ok2','ret2','ret3','ret4','ret_s1','retxret_s1','ok'],axis=1)
        else:
            return hf_data_df[origin_columns + rm]
        
        
    def Cpt_DecomRV(self, hf_data_df_, retTag, dayTag='Trddt', minTag='time'):
        
        hf_data_df = hf_data_df_.copy()
        hf_data_df['ret2'] = pow(hf_data_df[retTag],2)
        hf_data_df['rv_tag'] = np.where(hf_data_df[retTag]>0, 1, -1)
        
        RV = hf_data_df.groupby([dayTag,'rv_tag']).apply(lambda df: df['ret2'].sum()).reset_index()   
        rvp = RV[RV.rv_tag==1].drop('rv_tag' ,axis=1).rename(columns={0:'rvp'}).set_index('Trddt')
        rvn = RV[RV.rv_tag==-1].drop('rv_tag' ,axis=1).rename(columns={0:'rvn'}).set_index('Trddt')
        
        return  pd.merge(hf_data_df_, pd.concat([rvp,rvn], axis=1), left_on=dayTag, right_index=True)
        
        
    
    def Cpt_SemiBeta(self, hf_stock_, hf_index_):
        '''
        VT = Volatility_Tool()
        hf_stock_ = TB.Fetch_Stock_HFdata_from_Resset(1,minType=5)
        hf_stock_ = VT.Cpt_HF_LogReturn(hf_stock_).rename(columns={'log_close_diff':'ret_stk'})[['Trddt','time','ret_stk']]
        hf_index_ = TB.Fetch_Stock_HFdata_from_Resset(300, asset_type='index',minType=5)
        hf_index_ = VT.Cpt_HF_LogReturn(hf_index_).rename(columns={'log_close_diff':'ret_idx'})[['Trddt','time','ret_idx']]

        '''
        
        hf_stock = hf_stock_.copy()
        hf_index = hf_index_.copy()
        hf_data = pd.merge(hf_stock, hf_index).sort_values(['Trddt','time'])
        
        hf_data['stk_tag'] = np.where(hf_data.ret_stk>0, 1, -1)
        hf_data['idx_tag'] = np.where(hf_data.ret_idx>0, 1, -1)
        hf_data['idx_ret2'] = pow(hf_data.ret_idx, 2)
        hf_data['stk_idx_cov'] = hf_data.ret_stk * hf_data.ret_idx
        
        
        '''
        hf_data['idx_ret_p'] = np.where(hf_data.ret_idx>0, hf_data.ret_idx, 0)
        hf_data['idx_ret_n'] = np.where(hf_data.ret_idx<0, hf_data.ret_idx, 0)
        
        hf_data['si_cov_p'] = hf_data.ret_stk * hf_data.idx_ret_p
        hf_data['si_cov_n'] = hf_data.ret_stk * hf_data.idx_ret_n
        
        df_roll = hf_data.groupby('Trddt').rolling(25)
        idx2sum = df.idx_ret2.sum()

        hf_data['mpos_beta'] = (df_roll.idx_ret_p.sum() / hf_data.idx_ret2.sum()).values
        hf_data['mneg_beta'] = (df_roll.idx_ret_n.sum() / hf_data.idx_ret2.sum()).values

        hf_data['semi_bata_vari'] = hf_data['mpos_beta'] - hf_data['mneg_beta']
        
        '''
        
        def crt_4semibeta(df):
            idx2sum = df.idx_ret2.sum()
            
            bool_stk_neg = df.stk_tag==-1
            bool_idx_neg = df.idx_tag==-1
            
            beta_n = df[bool_stk_neg & bool_idx_neg].stk_idx_cov.sum()/idx2sum
            beta_p = df[(~bool_stk_neg) & (~bool_idx_neg)].stk_idx_cov.sum()/idx2sum
            beta_mn = -df[(~bool_stk_neg) & bool_idx_neg].stk_idx_cov.sum()/idx2sum
            beta_mp = -df[(bool_stk_neg) & (~bool_idx_neg)].stk_idx_cov.sum()/idx2sum
            beta = beta_n + beta_p - beta_mn - beta_mp
            
            return pd.DataFrame([{'beta':beta,'beta_n':beta_n,'beta_p':beta_p,'beta_mn':beta_mn,'beta_mp':beta_mp,}])
        
        beta_day_df = hf_data.groupby('Trddt').apply(crt_4semibeta).reset_index().drop('level_1',axis=1)
        
        # beta_day_df['semi_bata_vari'] = pow((beta_day_df.beta_n - beta_day_df.beta_mn) - (beta_day_df.beta_p - beta_day_df.beta_mp),2)
        
        
        return beta_day_df    
    
    
    def Cpt_ContJump_Truncate(self, hf_data_df_, minTag='time', tau=2.5, omega=0.49):
        
        hf_data_df = hf_data_df_.copy()
        n = len(hf_data_df[minTag].unique())
        hf_data_df['minvol'] = np.where(hf_data_df.rv>hf_data_df.bv, hf_data_df.bv,hf_data_df.rv)
        hf_data_df['alpha_trun'] = tau * np.sqrt(hf_data_df.minvol) * pow(n,-omega)
        hf_data_df = hf_data_df.drop(['minvol'], axis=1)
        
        return hf_data_df

    
    def Cpt_TOD_VolPattern(self, hf_data_df_, retTag, minTag='time', crtRet=False):
        
        hf_data_df = hf_data_df_.copy()        

        if crtRet:
            hf_data_df = self.Cpt_HF_LogReturn(hf_data_df.copy())
            
        else:
            hf_data_df = self.Cpt_RM(hf_data_df, retTag, rm=['rv','bv'])
            hf_data_df = self.Cpt_ContJump_Truncate(hf_data_df)
            hf_data_df['C_tag'] = np.where(abs(hf_data_df[retTag])<=hf_data_df.alpha_trun, 1, 0)
            
            hf_data_df['C_ret2'] = pow(hf_data_df.C_tag * hf_data_df[retTag],2)
                
            
            TOD = hf_data_df.groupby(minTag).C_ret2.mean()/hf_data_df.C_ret2.mean()
            TOD = TOD.to_frame().reset_index().rename(columns={'C_ret2':'TOD'})
            hf_data_df = pd.merge(hf_data_df, TOD)
                    
        return hf_data_df
    
    
    def Cpt_ConDiscon_Beta(self, hf_stock_, hf_index_, L=60):
        '''
        >>> VT = Volatility_Tool()
        >>> hf_stock_ = TB.Fetch_Stock_HFdata_from_Resset(1)
        >>> hf_stock_ = VT.Cpt_HF_LogReturn(hf_stock_).rename(columns={'log_close_diff':'ret_stk','stkcd':'Stkcd'})[['stkcd','Trddt','time','ret_stk']]
        >>> hf_index_ = TB.Fetch_Stock_HFdata_from_Resset(300, asset_type='index')
        >>> hf_index_ = VT.Cpt_HF_LogReturn(hf_index_).rename(columns={'log_close_diff':'ret_idx','stkcd':'Idxcd'})[['idxcd','Trddt','time','ret_idx']]

        '''
        
        hf_stock = hf_stock_.copy()
        hf_index = hf_index_.copy()
        hf_data = pd.merge(hf_stock, hf_index).sort_values(['Trddt','time'])
        n = len(hf_data['time'].unique())

        
        hf_data = self.Cpt_TOD_VolPattern(hf_data, 'ret_stk')
        hf_data = hf_data.rename(columns={'rv':'rv_stk', 'bv':'bv_stk','alpha_trun':'alpha_trun_stk', 
                                          'C_tag':'C_tag_stk', 'C_ret2':'C_ret2_stk', 'TOD':'TOD_stk'})
        
        hf_data = self.Cpt_TOD_VolPattern(hf_data, 'ret_idx')
        hf_data = hf_data.rename(columns={'rv':'rv_idx', 'bv':'bv_idx','alpha_trun':'alpha_trun_idx', 
                                          'C_tag':'C_tag_idx', 'C_ret2':'C_ret2_idx', 'TOD':'TOD_idx'})

        hf_data['ret_sum'] = hf_data.ret_stk + hf_data.ret_idx
        hf_data = self.Cpt_TOD_VolPattern(hf_data, 'ret_sum')
        hf_data = hf_data.rename(columns={'rv':'rv_sum', 'bv':'bv_sum','alpha_trun':'alpha_trun_sum', 
                                          'C_tag':'C_tag_sum', 'C_ret2':'C_ret2_sum', 'TOD':'TOD_sum'})

        hf_data['ret_sub']= hf_data.ret_stk - hf_data.ret_idx
        hf_data = self.Cpt_TOD_VolPattern(hf_data, 'ret_sub')
        hf_data = hf_data.rename(columns={'rv':'rv_sub', 'bv':'bv_sub','alpha_trun':'alpha_trun_sub', 
                                          'C_tag':'C_tag_sub', 'C_ret2':'C_ret2_sub', 'TOD':'TOD_sub'})
        
        hf_data['k_sum'] = 3 * pow(n,-0.49) * np.sqrt(np.where(hf_data.rv_sum>hf_data.bv_sum, hf_data.bv_sum,hf_data.rv_sum) * hf_data.TOD_sum.values)
        hf_data['k_sub'] = 3 * pow(n,-0.49) * np.sqrt(np.where(hf_data.rv_sub>hf_data.bv_sub, hf_data.bv_sub,hf_data.rv_sub) * hf_data.TOD_sub.values)
        hf_data['k_idx'] = 3 * pow(n,-0.49) * np.sqrt(np.where(hf_data.rv_idx>hf_data.bv_idx, hf_data.bv_idx,hf_data.rv_idx) * hf_data.TOD_idx.values)
        
        hf_data['Csum_tag'] = np.where(abs(hf_data['ret_sum'])<=hf_data.k_sum, 1, 0)
        hf_data['Csub_tag'] = np.where(abs(hf_data['ret_sub'])<=hf_data.k_sub, 1, 0)
        hf_data['Cidx_tag'] = np.where(abs(hf_data['ret_idx'])<=hf_data.k_idx, 1, 0)

        hf_data['conBeta_up'] = pow(hf_data.ret_sum, 2) * hf_data['Csum_tag'] - pow(hf_data.ret_sub, 2) * hf_data['Csub_tag']
        hf_data['conBeta_bot'] = pow(hf_data.ret_idx, 2) * hf_data['Cidx_tag']
        hf_data['disconBeta_up'] = pow(hf_data.ret_stk * hf_data.ret_idx,2)
        hf_data['disconBeta_bot'] = pow(hf_data.ret_idx, 4)
    
        hf_data = hf_data[[ 'Trddt', 'time','conBeta_up','conBeta_bot','disconBeta_up','disconBeta_bot']]        
        hf_data = hf_data.groupby(['Trddt']).sum().rolling(L).sum().dropna()
        hf_data['conBeta'] = hf_data.conBeta_up/hf_data.conBeta_bot/4
        hf_data['disconBeta'] = np.sqrt(hf_data.disconBeta_up/hf_data.disconBeta_bot)
        
        return hf_data[['conBeta', 'disconBeta']].reset_index()
    
    
    def Cpt_BV_and_LogClsDiff(stkcd, hf_index, stock_base_data=None, n=240, min_=1, asset_type='stock'):
        """
        hf_stock_data_dir = r'D:\个人项目\数据集合\学术研究_股票数据\5minHFprice'
        file_list = TB.Tag_FilePath_FromDir(hf_stock_data_dir, suffix='mat')
        file = file_list[0]
        
        """
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
            
            def Cpt_BV(hf_data):
                hf_data_df = hf_data.copy()
                hf_data_df['log_close_diff_abs_s1'] = hf_data_df.groupby('Trddt').log_close_diff_abs.shift(1)
                hf_data_df['bv'] = hf_data_df.log_close_diff_abs * hf_data_df.log_close_diff_abs_s1
                BV = (np.pi/2 * hf_data_df.groupby('Trddt').bv.sum()).reset_index()
                
                return BV
            
            # Calculate bv of stocks and index
            BV = Cpt_BV(hf_data_df)
            hf_data_df = pd.merge(hf_data_df, BV, left_on=['Trddt'], right_on=['Trddt'])
            
            hf_data_df['log_close_diff_abs_index_s1'] = hf_data_df.groupby('Trddt').log_close_diff_abs_index.shift(1)
            hf_data_df['bv_index'] = hf_data_df.log_close_diff_abs_index * hf_data_df.log_close_diff_abs_index_s1
            BV_index = (np.pi/2 * hf_data_df.groupby('Trddt').bv_index.sum())
            hf_data_df = hf_data_df.drop('bv_index',axis=1)
            hf_data_df = pd.merge(hf_data_df, BV_index, left_on=['Trddt'], right_on=['Trddt'])
            
            # if some day have nan in log_close_diff than drop that day
            drop_date1 = hf_data_df[hf_data_df.log_close_diff.isna()].Trddt   
            hf_data_df = hf_data_df[~hf_data_df.Trddt.isin(drop_date1)]
                
            full_trading_dates = hf_index.Trddt.unique()
            all_dates = pd.DataFrame({'Trddt': sorted(full_trading_dates)})
            all_times = pd.DataFrame({'time': hf_index.time.unique().tolist()})
            merged_df = pd.merge(all_dates, all_times, how='cross')
            hf_data_df = pd.merge(merged_df, hf_data_df, on=['time', 'Trddt'], how='left')
            
            # Creat dX and bVm
            log_close_diff_pivot = hf_data_df.pivot(index='time',columns='Trddt',values='log_close_diff')
            log_close_diff_pivot = log_close_diff_pivot.fillna(log_close_diff_pivot.median())
            
            dz = log_close_diff_pivot.values
            bvz = np.sqrt(hf_data_df.groupby('Trddt').bv.mean().values).reshape(1,-1)
                        
            log_close_diff_pivot_index = hf_data_df.pivot(index='time',columns='Trddt',values='log_close_diff_abs_index')
            # have this or not may have some unexcpet influence
            log_close_diff_pivot_index = log_close_diff_pivot_index.fillna(log_close_diff_pivot.median())
            #######################################################################
            dX = log_close_diff_pivot_index.values
            bVm = np.sqrt(hf_data_df.groupby('Trddt').bv_index.mean().values).reshape(1,-1)
            
            print('finished:',stkcd)
            return [(dz, bvz, dX, bVm),stkcd]
    
        except:
            pd.DataFrame([stkcd]).to_csv('F:\Intrady Beta Pattern\erro_code\erro_{}.csv'.format(stkcd))
            print('erro_{}'.format(stkcd))    


    def Cpt_beta_quantiles(stkcd, hf_index, stock_base_data, kn, min_=5, n=48):
        
        # hf_index = TB.Fetch_Stock_HFdata_from_Resset(300, asset_type='index', minType=min_) 
        # hf_index = hf_index[['stkcd','Trddt','time','open','close']]
        # hf_index = hf_index.rename(columns={'close':'index_close','open':'index_open'})

        # stock_base_data = pd.read_csv(r'F:\数据集合\学术研究_股票数据\CSMAR\Combined_Data\SAVIC_saveMV_day.csv',usecols=['Stkcd','Trddt'])
        # stock_base_data = stock_base_data[stock_base_data.Trddt.isin(hf_index.Trddt)]
        # all_stk_list = stock_base_data.Stkcd.unique().tolist()

        # BQ90_df = pd.DataFrame()
        # BQ75_df = pd.DataFrame()
        
        def Cpt_betas(qq, dX, dZ, bVm, bVz, n=240, kn=24, b=1, omega=0.49, abar=4):
            
            idxI = np.arange(qq - kn + 1, qq + 1)  # Corrected index calculation

            trcx = ((n / b)**omega * np.abs(dX[idxI, :]) <= abar * bVm)  # Corrected broadcasting
            trcz = ((n / b)**omega * np.abs(dZ[idxI, :]) <= abar * bVz)  # Corrected broadcasting

            Vm = TB.nb_mean(dX[idxI, :]**2 * trcx * trcz, axis=0)  # Corrected axis

            betas = TB.nb_mean(dZ[idxI, :] * dX[idxI, :] * trcx * trcz, axis=0) / Vm  # Corrected axis

            return betas


        # for stkcd in all_stk_list:
        dZ, bVz, dX, bVm = Cpt_BV_and_LogClsDiff(stkcd, hf_index, stock_base_data, n=n, min_=min_)[0]
        n, T = dX.shape
        betas = np.full((n, T), np.nan)
        
        for qq in range(kn-1,n):
            betas[qq,:] = Cpt_betas(qq, dX, dZ, bVm, bVz, n=n, kn=kn, b=1, omega=0.49, abar=4)    

        betas_df = pd.DataFrame(betas,columns=hf_index.Trddt.unique(),index=hf_index.time.unique()).T
        bq_90 = (betas_df.quantile(0.9, axis=1) - betas_df.quantile(0.1, axis=1)).rename(stkcd)
        bq_75 = (betas_df.quantile(0.75, axis=1) - betas_df.quantile(0.25, axis=1)).rename(stkcd)
        
        # BQ90_df = pd.concat([BQ90_df, bq_90], axis=1)
        # BQ75_df = pd.concat([BQ75_df, bq_75], axis=1)
        
        bq_90.to_csv(r'F:\Intrady Beta Pattern\BQ90_{}\{}\{}\{}.csv'.format(min_,'300',kn,stkcd))
        bq_75.to_csv(r'F:\Intrady Beta Pattern\BQ75_{}\{}\{}\{}.csv'.format(min_,'300',kn,stkcd))
        
        print('BQ finished:{}_{}_{}'.format(stkcd,kn,'300'))
        # BQ90_df.to_csv(r'F:\Intrady Beta Pattern\Merge_res\BQ90_{}_300_{}.csv'.format(min_, kn))
        # BQ75_df.to_csv(r'F:\Intrady Beta Pattern\Merge_res\BQ75_{}_300_{}.csv'.format(min_, kn))

        
    def Cpt_OKVol():
        
        pass

    
    
def generate_time_list(start_time, end_time, minute_interval):
    time_list = []
    current_time = start_time

    while current_time <= end_time:
        time_list.append(current_time)
        hour = current_time // 100
        minute = current_time % 100

        if minute + minute_interval >= 60:
            hour += 1
            minute = (minute + minute_interval) % 60
        else:
            minute += minute_interval

        current_time = hour * 100 + minute

    return time_list


def Fetch_Index_Component_daily(beg_date='2005-01-03', end_date='2016-12-31', left_day_num=250):
    # get basic stock data
    DC = DataCleaner(main_dir_path=r'D:\个人项目\数据集合\学术研究_股票数据', init=False)
    DC._CSMAR.StockDayTradeData_df = DC._CSMAR.Comb_Stock_Day_TradeData()
    DC._CSMAR.StockMonTradeData_df = DC._CSMAR.Comb_Stock_Mon_TradeData()
    DC._CSMAR.StockCompBasicData_df = DC._CSMAR.Comb_Stock_CompBasicData()
    
    DC._CSMAR.Add_Stock_Listdt(timeType='D')
    DC._CSMAR.Add_Stock_Trdmnt_intoDay()
    
    # stock data filter
    stock_daily_data = DC._CSMAR.StockDayTradeData_df.copy()
    stock_daily_data = stock_daily_data.sort_values(['Stkcd','Trddt'])
    stock_daily_data['MV_s1'] = stock_daily_data.groupby(['Stkcd']).Dsmvtll.shift(1)

    stock_daily_data = DC.Select_Stock_ExchangeTradBoard(df=stock_daily_data, timeType='D')
    stock_daily_data = DC.Drop_Stock_InadeListdtData(df=stock_daily_data, timeType='D')
    stock_daily_data = DC.Drop_Stock_STData(df=stock_daily_data, timeType='D')
    stock_daily_data = DC.Drop_Stock_MVData(df=stock_daily_data, timeType='D')
    
    # stock_daily_data = DC.Drop_Stock_InadeTradData(df=stock_daily_data, timeType='D')
    stock_daily_data.to_csv(r'D:\个人项目\数据集合\学术研究_股票数据\CSMAR\Combined_Data\SAV_Stock_Daily_data.csv', index=False)
    stock_daily_data = DC.Select_Stock_GivenDayData(df=stock_daily_data, begDate=beg_date, endDate=end_date, timeType='D')
    stock_daily_data = stock_daily_data[['Stkcd','Trddt','MV_s1']]
    stock_daily_data = stock_daily_data[stock_daily_data.Stkcd.isin( \
                            stock_daily_data.groupby('Stkcd').Trddt.count()[stock_daily_data.groupby('Stkcd').Trddt.count()>=left_day_num].index)]
    
    # get stock code that exit in high frequency data and filtered daily data
    hf_stock_data_dir = r'D:\个人项目\数据集合\学术研究_股票数据\5minHFprice'
    file_list = TB.Tag_FilePath_FromDir(hf_stock_data_dir, suffix='mat')
    hf_stkcd_list = []
    for file in file_list:
        stkcd = int(re.findall('\d{6}', file)[0])
        hf_stkcd_list.append(stkcd)
    
    filtered_stkcd_list = stock_daily_data.Stkcd.unique().tolist()
    used_stkcd = list(set(hf_stkcd_list)&set(filtered_stkcd_list))
    stock_daily_data = stock_daily_data[stock_daily_data.Stkcd.isin(used_stkcd)]
    
    # calculate the weight of each stock to creat high frequency index
    MV_sum = stock_daily_data.groupby('Trddt').MV_s1.sum().reset_index().rename(columns={'MV_s1':'MV_sum'})
    stock_daily_data = pd.merge(stock_daily_data, MV_sum, left_on='Trddt',right_on='Trddt')
    stock_daily_data['Weight_mv'] = stock_daily_data['MV_s1']/stock_daily_data['MV_sum']
    stock_daily_data = stock_daily_data.drop(['MV_s1','MV_sum'], axis=1)
    stock_daily_data.to_csv(r'D:\个人项目\数据集合\学术研究_股票数据\CSMAR\Combined_Data\SAV_Stock_Daily_data_inner.csv', index=False)


def Fetch_Index_Component_intraday(stkcd, stock_base_data, threshold_intraday=3/4, threshold_day=250):
        
    # get high frequency stock data
    hf_stock_data_dir = 'D:\\个人项目\\数据集合\\学术研究_股票数据\\5minHFprice\\'
    try:
        file = hf_stock_data_dir + 'SH' + TB.Fill_StockCode(stkcd) + '.mat'
        mat = loadmat(file)
    except:
        file = hf_stock_data_dir + 'SZ' + TB.Fill_StockCode(stkcd) + '.mat'
        mat = loadmat(file)
    
    hf_data_df = pd.DataFrame(mat)
    hf_data_df['stkcd'] = stkcd
    tag_list = ['day','time','open','high','low','close','volume','turnover']
    for i in range(len(tag_list)):
        hf_data_df[tag_list[i]] = hf_data_df['datamat'].str[i]
    hf_data_df.drop('datamat',axis=1,inplace=True)
    hf_data_df['Trddt'] = hf_data_df.day.apply(int).apply(str).apply(lambda x:x[:4]+'-'+x[4:6]+'-'+x[6:])
    hf_data_df = hf_data_df[['stkcd','Trddt','time','close']]
    
    # drop data that not sampled in 5min
    start_time1 = 935
    end_time1 = 1130
    start_time2 = 1305
    end_time2 = 1500
    minute_interval = 5
    
    time_list1 = generate_time_list(start_time1, end_time1, minute_interval)
    time_list2 = generate_time_list(start_time2, end_time2, minute_interval)
    time_list = time_list1 + time_list2
    
    hf_data_df = hf_data_df[hf_data_df.time.isin(time_list)]
    hf_data_df = hf_data_df.sort_values(['Trddt','time'])
    
    # add log price and cpt log return
    # the other method is not fill and drop the first data point of each day
    # but due to the data is 5min sampled, intraday only hace 48 data points
    # drop the first data point may not the best choice
    hf_data_df['log_close'] = np.log(hf_data_df['close'])
    hf_data_df['log_close_diff'] = hf_data_df.log_close.diff()
    hf_data_df['log_close_diff_abs'] = abs(hf_data_df.log_close_diff)
    
    # if some day do not have 48 data points than drop that day
    drop_date2 = hf_data_df.groupby('Trddt').log_close_diff.count() != 48
    drop_date2 = drop_date2[drop_date2].index
    hf_data_df = hf_data_df[~hf_data_df.Trddt.isin(drop_date2)]
    
    # if the percentage of some day's intraday zero return's number is bigger than 1/3
    # that drop that day
    drop_date3 = hf_data_df.groupby('Trddt').apply(lambda x:(x.log_close_diff!=0).sum()<int(48*threshold_intraday))
    drop_date3 = drop_date3[drop_date3].index
    hf_data_df = hf_data_df[~hf_data_df.Trddt.isin(drop_date3)]
    
    # calculate high frequency data that after weight adjustment
    hf_data_df = pd.merge(hf_data_df, stock_base_data[stock_base_data.Stkcd==stkcd][['Trddt','Weight_mv']],
                          left_on='Trddt', right_on='Trddt')
    hf_data_df['close_adj'] = hf_data_df.close * hf_data_df.Weight_mv
    hf_data_df = hf_data_df.drop('Weight_mv', axis=1)
    
    if len(hf_data_df.Trddt.unique()) >= threshold_day:
        hf_data_df.to_csv(r'D:\个人项目\Intrady Beta Pattern\Index_conpoment\{}.csv'.format(stkcd))
        print('{} finfished'.format(stkcd))
        return hf_data_df[['stkcd','Trddt','time','close_adj']]

    else:
        print('{} do not have enough data'.format(stkcd))
        

def process_stock_data(stkcd, stock_base_data):
    
    hf_stock_data_dir = 'D:\\个人项目\\数据集合\\学术研究_股票数据\\5minHFprice\\'
    try:
        hf_data_path = hf_stock_data_dir + 'SH' + TB.Fill_StockCode(stkcd) + '.mat'
        mat = loadmat(hf_data_path)
    except:
        hf_data_path = hf_stock_data_dir + 'SZ' + TB.Fill_StockCode(stkcd) + '.mat'
        mat = loadmat(hf_data_path)

    hf_data_df = pd.DataFrame(mat)        
    hf_data_df['stkcd'] = stkcd
    tag_list = ['day','time','open','high','low','close','volume','turnover']
    
    for i in range(len(tag_list)):
        hf_data_df[tag_list[i]] = hf_data_df['datamat'].str[i]
    hf_data_df.drop('datamat',axis=1,inplace=True)
    hf_data_df['day'] = hf_data_df.day.apply(int)
    
    # drop data that not sampled in 5min
    start_time1 = 935
    end_time1 = 1130
    start_time2 = 1305
    end_time2 = 1500
    minute_interval = 5
    
    time_list1 = generate_time_list(start_time1, end_time1, minute_interval)
    time_list2 = generate_time_list(start_time2, end_time2, minute_interval)
    time_list = time_list1 + time_list2
    
    hf_data_df = hf_data_df[hf_data_df.time.isin(time_list)]

    hf_data_df['day'] = hf_data_df.day.apply(lambda x:str(x)[:4]+'-'+str(x)[4:6]+'-'+str(x)[6:])
    
    # calculate high frequency data that after weight adjustment
    hf_data_df = pd.merge(hf_data_df, stock_base_data[stock_base_data.Stkcd==stkcd][['Trddt','Weight_mv']],
                          left_on='day', right_on='Trddt')
    hf_data_df['close_adj'] = hf_data_df.close * hf_data_df.Weight_mv
    hf_data_df['open_adj'] = hf_data_df.open * hf_data_df.Weight_mv
    # hf_data_df['open_adj'] = hf_data_df.open * hf_data_df.Weight_mv
    hf_data_df = hf_data_df[['stkcd','Trddt','time','open_daj', 'close_adj']]
    
    hf_data_df.to_csv(r'D:\个人项目\Intrady Beta Pattern\Index_conpoment\{}.csv'.format(stkcd))
    print('{} finfished'.format(stkcd))
    return hf_data_df


def Mrg_HF_Index_data():
    
    hf_index_data = pd.DataFrame()
    stock_base_data = pd.read_csv(r'D:\个人项目\数据集合\学术研究_股票数据\CSMAR\Combined_Data\SAV_Stock_Daily_data_inner.csv')
    
    # Set the number of processes you want to use (e.g., 4)
    num_processes = 12

    # Create a Pool of processes
    with Pool(num_processes) as pool:
        # Use apply_async to process each stock chunk concurrently
        results = [pool.apply_async(process_stock_data, (stkcd, stock_base_data, )) for stkcd in stock_base_data.Stkcd.unique().tolist()]

        # Close the pool and wait for all processes to finish
        pool.close()
        pool.join()

    # Get the results from the processes
    processed_results = [result.get() for result in results if result.get() is not None]

    # Get the results from each process
    hf_index_data = pd.concat(processed_results, ignore_index=True)
    hf_index_data = hf_index_data.rename(columns={'open_adj':'hf_index_open', 'close_adj':'hf_index_close'}) 
    hf_index_data = hf_index_data.groupby(['Trddt','time'])[['hf_index_open','hf_index_close']].sum()     
    hf_index_data.to_csv(r'D:\个人项目\Intrady Beta Pattern\hf_index.csv')
    
    return hf_index_data


def Cpt_BV_and_LogClsDiff(stkcd, hf_index):
    """
    hf_stock_data_dir = r'D:\个人项目\数据集合\学术研究_股票数据\5minHFprice'
    file_list = TB.Tag_FilePath_FromDir(hf_stock_data_dir, suffix='mat')
    file = file_list[0]
    
    """
    
    # get high frequency stock data
    hf_stock_data_dir = 'D:\\个人项目\\数据集合\\学术研究_股票数据\\5minHFprice\\'
    try:
        file = hf_stock_data_dir + 'SH' + TB.Fill_StockCode(stkcd) + '.mat'
        mat = loadmat(file)
    except:
        file = hf_stock_data_dir + 'SZ' + TB.Fill_StockCode(stkcd) + '.mat'
        mat = loadmat(file)
    
    
    hf_data_df = pd.DataFrame(mat)
    
    hf_data_df['stkcd'] = int(re.findall('\d{6}', file)[0])
    tag_list = ['day','time','open','high','low','close','volume','turnover']
    
    for i in range(len(tag_list)):
        hf_data_df[tag_list[i]] = hf_data_df['datamat'].str[i]
    hf_data_df.drop('datamat',axis=1,inplace=True)
    hf_data_df['Trddt'] = hf_data_df.day.apply(int).apply(str).apply(lambda x:x[:4]+'-'+x[4:6]+'-'+x[6:])
    hf_data_df = pd.merge(hf_data_df, hf_index, left_on=['Trddt','time'], right_on=['Trddt','time'])
    hf_data_df = hf_data_df[['stkcd','Trddt','time','open','close','hf_index_open','hf_index_close']]

    # calculate diff log price
    hf_data_df = hf_data_df.sort_values(['Trddt','time'])
    hf_data_df['log_close'] = np.log(hf_data_df['close'])
    hf_data_df['log_close_s1'] = hf_data_df.groupby('Trddt').log_close.shift(1)
    hf_data_df.loc[np.isnan(hf_data_df.log_close_s1),'log_close_s1'] = np.log(hf_data_df.loc[np.isnan(hf_data_df.log_close_s1),'open'])
    hf_data_df['log_close_diff'] = hf_data_df.log_close - hf_data_df.log_close_s1
    hf_data_df['log_close_diff_abs'] = abs(hf_data_df.log_close_diff)
    
    hf_data_df['log_close_index'] = np.log(hf_data_df['hf_index_close'])
    hf_data_df['log_close_index_s1'] = hf_data_df.groupby('Trddt').log_close_index.shift(1)
    hf_data_df.loc[np.isnan(hf_data_df.log_close_index_s1),'log_close_index_s1'] = \
        np.log(hf_data_df.loc[np.isnan(hf_data_df.log_close_index_s1),'hf_index_open'])
    hf_data_df['log_close_diff_index'] = hf_data_df.log_close_index - hf_data_df.log_close_index_s1
    hf_data_df['log_close_diff_abs_index'] = abs(hf_data_df.log_close_diff_index)

    # drop data that not sampled in 5min
    start_time1 = 935
    end_time1 = 1130
    start_time2 = 1305
    end_time2 = 1500
    minute_interval = 5
    
    time_list1 = generate_time_list(start_time1, end_time1, minute_interval)
    time_list2 = generate_time_list(start_time2, end_time2, minute_interval)
    time_list = time_list1 + time_list2
    
    hf_data_df = hf_data_df[hf_data_df.time.isin(time_list)]
    hf_data_df = hf_data_df.sort_values(['Trddt','time'])
    
    def Cpt_BV(hf_data):
        hf_data_df = hf_data.copy()
        hf_data_df['log_close_diff_abs_s1'] = hf_data_df.groupby('Trddt').log_close_diff_abs.shift(1)
        hf_data_df['bv'] = hf_data_df.log_close_diff_abs * hf_data_df.log_close_diff_abs_s1
        BV = (np.pi/2 * hf_data_df.groupby('Trddt').bv.sum()).reset_index()
        
        return BV
    
    # Calculate bv of stocks and index
    BV = Cpt_BV(hf_data_df)
    hf_data_df = pd.merge(hf_data_df, BV, left_on=['Trddt'], right_on=['Trddt'])
    
    hf_data_df['log_close_diff_abs_index_s1'] = hf_data_df.groupby('Trddt').log_close_diff_abs_index.shift(1)
    hf_data_df['bv_index'] = hf_data_df.log_close_diff_abs_index * hf_data_df.log_close_diff_abs_index_s1
    BV_index = (np.pi/2 * hf_data_df.groupby('Trddt').bv_index.sum())
    hf_data_df = hf_data_df.drop('bv_index',axis=1)
    hf_data_df = pd.merge(hf_data_df, BV_index, left_on=['Trddt'], right_on=['Trddt'])
    
    # if some day have nan in log_close_diff than drop that day
    drop_date1 = hf_data_df[hf_data_df.log_close_diff.isna()].Trddt   
    hf_data_df = hf_data_df[~hf_data_df.Trddt.isin(drop_date1)]
    
    # if some day do not have 48 data points than drop that day
    drop_date2 = hf_data_df.groupby('Trddt').log_close_diff.count() != 48
    drop_date2 = drop_date2[drop_date2].index
    hf_data_df = hf_data_df[~hf_data_df.Trddt.isin(drop_date2)]
    
    # # if the percentage of some day's intraday zero return's number is bigger than 1/3
    # # that drop that day
    # drop_date3 = hf_data_df.groupby('Trddt').apply(lambda x:(x.log_close_diff!=0).sum()<int(48*3/4))
    # # drop_date3 = hf_data_df.groupby('Trddt').apply(lambda x:(x.log_close_diff!=0).sum()<int(42))

    # drop_date3 = drop_date3[drop_date3].index
    # hf_data_df = hf_data_df[~hf_data_df.Trddt.isin(drop_date3)]

    # if len(hf_data_df.Trddt.unique()) > 250:

    full_trading_dates = hf_index.index.unique()
    all_dates = pd.DataFrame({'Trddt': sorted(full_trading_dates)})
    all_times = pd.DataFrame({'time': time_list})
    merged_df = pd.merge(all_dates, all_times, how='cross')
    hf_data_df = pd.merge(merged_df, hf_data_df, on=['time', 'Trddt'], how='left')

    
    # Creat dX and bVm
    log_close_diff_pivot = hf_data_df.pivot(index='time',columns='Trddt',values='log_close_diff')
    dz = log_close_diff_pivot.values
    bvz = np.sqrt(hf_data_df.groupby('Trddt').bv.mean().values).reshape(1,-1)
        
    log_close_diff_pivot_index = hf_data_df.pivot(index='time',columns='Trddt',values='log_close_diff_abs_index')
    dX = log_close_diff_pivot_index.values
    bVm = np.sqrt(hf_data_df.groupby('Trddt').bv_index.mean().values).reshape(1,-1)
    
    print('finished:',file)
    
    return [(dz, bvz, dX, bVm),stkcd]

# else:
    # print('do not have enough data',file)
        

# def Cpt_Stock_DT(stkcd):
    
#     hf_data_dir = 'D:\\个人项目\\Intrady Beta Pattern\\Index_conpoment\\'
    

def Crt_DT():
    use_stock_df = pd.DataFrame(TB.Tag_FilePath_FromDir(r'D:\个人项目\Intrady Beta Pattern\Index_conpoment'))
    use_stock = use_stock_df[0].str.extract('(\d+)')[0]
    hf_index = pd.read_csv(r'D:\个人项目\Intrady Beta Pattern\hf_index.csv',index_col=0)

    num_processes = 16
    # Create a Pool of processes
    with Pool(num_processes) as pool:
        
        results = [pool.apply_async(Cpt_BV_and_LogClsDiff, (stkcd,hf_index,)) for stkcd in use_stock.tolist()]

        # Close the pool and wait for all processes to finish
        pool.close()
        pool.join()
        
    # Get the results from the processes
    processed_results = [result.get()[0] for result in results if result.get() is not None]
    use_stock_list = [result.get()[1] for result in results if result.get() is not None]
    DT1 = processed_results[:int(len(processed_results)/2)]
    DT2 = processed_results[int(len(processed_results)/2):]
    
    dtype_tuple = [('dZ', 'O'), ('bVz', 'O'),('dX', 'O'), ('bVm', 'O')]
    void_dtype = np.dtype(dtype_tuple)
    DT1 = np.array(DT1, dtype=void_dtype).reshape(1,-1)
    DT2 = np.array(DT2, dtype=void_dtype).reshape(1,-1)
    
    savemat(r'D:\个人项目\Intrady Beta Pattern\DT1.mat', {'DT1':DT1})
    savemat(r'D:\个人项目\Intrady Beta Pattern\DT2.mat', {'DT2':DT2})
    pd.DataFrame(use_stock_list).to_csv(r'D:\个人项目\Intrady Beta Pattern\use_stock.csv')
    print('successfully save DT')






# if __name__ == '__main__':
    # hf_index_data = Mrg_HF_Index_data()

    
    # # Crt index high frequency data
    # index_component_data_dir = r'D:\个人项目\Intrady Beta Pattern\Index_conpoment'
    # file_list = TB.Tag_FilePath_FromDir(index_component_data_dir, suffix='csv')
    # df_list = [pd.read_csv(file, index_col=0) for file in file_list]
    # hf_index_data = pd.concat(df_list, ignore_index=True)   
    # hf_index_data = hf_index_data.rename(columns={'open_daj':'hf_index_open', 'close_adj':'hf_index_close'}) 
    # hf_index_datas = hf_index_data.groupby(['Trddt','time'])[['hf_index_open','hf_index_close']].sum()     
    # hf_index_datas.to_csv(r'D:\个人项目\Intrady Beta Pattern\hf_index.csv')

        
    # start_time1 = 935
    # end_time1 = 1130
    # start_time2 = 1305
    # end_time2 = 1500
    # minute_interval = 5
    
    # time_list1 = generate_time_list(start_time1, end_time1, minute_interval)
    # time_list2 = generate_time_list(start_time2, end_time2, minute_interval)
    # time_list = time_list1 + time_list2
    
    # hf_index = hf_index_data[hf_index_data.time.isin(time_list)].groupby(['Trddt','time']).close_adj.sum()
    # hf_index = hf_index.rename(columns={'close_adj':'hf_index_close'})      
    # hf_index.to_csv(r'D:\个人项目\Intrady Beta Pattern\hf_index.csv')
    # hf_index.hf_index_close.plot()
    
    
    # hf_index = pd.read_csv(r'D:\个人项目\Intrady Beta Pattern\hf_index.csv')
    # index_day = hf_index.groupby('Trddt').mean().drop('time',axis=1)
    # index_day.hf_index_close.plot()
    
    
    # self = DataCombiner_CSMAR(main_dir_path=r'D:\个人项目\数据集合\学术研究_股票数据', init=False)
    # idx_data = self.Comb_Index_Day_TradeData()
    # hs300 = idx_data[idx_data.Idxcd=='000300'][['Trddt','close']].rename(columns={'close':'close_hs300'})
    # zz500 = idx_data[idx_data.Idxcd=='000905'][['Trddt','close']].rename(columns={'close':'close_zz500'})
    # zz1000 = idx_data[idx_data.Idxcd=='000852'][['Trddt','close']].rename(columns={'close':'close_zz1000'})
        
    # index_day = pd.merge(index_day, hs300, left_on='Trddt', right_on='Trddt', how='left')
    # index_day = pd.merge(index_day, zz500, left_on='Trddt', right_on='Trddt', how='left')
    # index_day = pd.merge(index_day, zz1000, left_on='Trddt', right_on='Trddt', how='left')
    # index_day = index_day.rename(columns={'hf_index_close':'close_my_index'})
    # index_day.set_index('Trddt',inplace=True)
    

    # fig, axs = plt.subplots(3, 1, figsize=(12, 10))

    # for i in range(len(index_day.columns)-1):
    #     column = index_day.columns[i+1]
    #     plot_data = index_day[['close_my_index',column]]
    #     plot_data = plot_data.dropna(subset=[column])
    #     plot_data = plot_data.apply(lambda x: x/x.max()) 
        
    #     axs[i].plot(plot_data, label=['close_my_index', column])
        
    #     # Calculate the stride based on the number of data points and the desired number of x-ticks
    #     desired_xticks = 10  # You can adjust this value as needed
    #     stride = max(len(plot_data.index) // desired_xticks, 1)
        
    #     xticks = range(0, len(plot_data.index), stride)
    #     axs[i].set_xticks(xticks)
        
    #     # Use the stride to display the corresponding x-tick labels
    #     axs[i].set_xticklabels(plot_data.index[xticks], rotation=45)
        
    #     # Add a legend to the subplot
    #     axs[i].legend()
        
    # plt.subplots_adjust(hspace=0.5) 
    # plt.suptitle('Normalized Time Series of Indexs Price')
    # plt.tight_layout()
    # plt.savefig(r'D:\个人项目\Latex file\article\Intraday Market Beta Pattern\pic\Normalized Time Series of Indexs Price')
    
    # corr_df = index_day.corr()
    # corr_df = corr_df.rename(columns={'close_my_index':'My_Index','close_hs300':'HS300','close_zz500':'ZZ500','close_zz1000':'ZZ1000'})
    # corr_df.index = corr_df.columns
    # corr_df.to_csv(r'D:\个人项目\Latex file\article\Intraday Market Beta Pattern\table\index_corr.csv')
    # index_day['close_zz1000_adj'] = index_day.close_zz1000/500
    # index_day.plot(y=['close_zz1000_adj','close_adj'])

    
    # # Calculate the log price diff and sqrt bv of index
    # hf_index = pd.read_csv(r'D:\个人项目\Intrady Beta Pattern\hf_index.csv',index_col=0)  
    # hf_index['log_close'] = np.log(hf_index['hf_index_close'])
    # hf_index['log_close_diff'] = hf_index.log_close.diff()
    # hf_index['log_close_diff_abs'] = abs(hf_index.log_close_diff)
    # log_close_diff_pivot = hf_index.pivot('time','Trddt','log_close_diff')
    # dX = log_close_diff_pivot.dropna(axis=1).values
    
    # hf_index = hf_index.sort_values(['Trddt','time'])
    # hf_index['log_close_diff_abs_s1'] = hf_index.groupby('Trddt').log_close_diff_abs.shift(1)
    # hf_index['bv'] = hf_index.log_close_diff_abs * hf_index.log_close_diff_abs_s1
    # BV = (np.pi/2 * hf_index.groupby('Trddt').bv.sum())
    # bVm = np.sqrt(BV.values).reshape(1,-1)
    # bVm = np.delete(bVm, 0, axis=1)

    
    # savemat(r'D:\个人项目\Intrady Beta Pattern\dX.mat', {'dX':dX})
    # savemat(r'D:\个人项目\Intrady Beta Pattern\bVm.mat', {'bVm':bVm})

    # Crt_DT()
    # Mrg_HF_Index_data()

        
    
# # 定义void数据类型
# dtype_str = 'i4,f8,a10'  # i4: 4字节整数，f8: 8字节浮点数，a10: 长度为10的字符串

# # 创建一个void数据
# data = np.array([(1, 2.5, b'hello'), (3, 4.7, b'world')], dtype=void_dtype) 

# dtype_tuple = [('dZ', 'O'), ('bVz', 'O')]
# void_dtype = np.dtype(dtype_tuple)
# DT1 = np.array([(dX,bVm),(dX,bVm)], dtype=void_dtype).reshape(1,-1)
    
# SH600296
    
    

# # Assuming your DataFrame is named 'df' with columns: 'stockname', 'tradingdate', and 'weight'

# # Step 1: Find the set of unique trading dates
# unique_trading_dates = stock_base_data['Trddt'].unique()

# # Step 2: Group the DataFrame by stock name
# grouped_by_stock = stock_base_data.groupby('Stkcd')

# # Step 3: Count the number of unique trading dates for each stock group
# stock_trading_date_counts = grouped_by_stock['Trddt'].nunique()

# # Step 4: Filter the stocks that exist in all trading dates
# stocks_exist_in_all_dates = stock_trading_date_counts[stock_trading_date_counts == len(unique_trading_dates)].index

# # Now, you can use 'stocks_exist_in_all_dates' to select the desired stocks from the original DataFrame
# filtered_df = stock_base_data[stock_base_data['Stkcd'].isin(stocks_exist_in_all_dates)]


# Tt = pd.Series(stock_base_data.Trddt.unique(), name='Trddt')
# Tt = Tt.sort_values().reset_index(drop=True).to_frame()
# Tt['year'] = Tt.Trddt.str.extract('(\d{4})')
# Tt.drop_duplicates('year',keep='last').reset_index()['index'] + 1

# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# from scipy.fft import fft

# # 生成模拟的股票收益率数据
# np.random.seed(0)
# num_days = 1000
# daily_returns = np.random.normal(loc=0, scale=0.02, size=num_days)  # 均值为0，标准差为0.02的正态分布随机数作为模拟收益率

# # 计算累积收益率
# cumulative_returns = np.cumsum(daily_returns)

# # 进行傅里叶变换
# fft_result = fft(cumulative_returns)

# # 计算频率轴
# sampling_rate = 1  # 假设每天采样一次
# freq_axis = np.fft.fftfreq(num_days, d=1/sampling_rate)

# # 绘制原始收益率和频域图像
# plt.figure(figsize=(12, 6))

# # 原始收益率图像
# plt.subplot(2, 1, 1)
# plt.plot(np.arange(num_days), cumulative_returns)
# plt.xlabel('Day')
# plt.ylabel('Cumulative Returns')
# plt.title('Cumulative Returns of Simulated Stock')

# # 频域图像
# plt.subplot(2, 1, 2)
# plt.plot(freq_axis, np.abs(fft_result))  # 使用np.abs()计算振幅谱
# plt.xlabel('Frequency (days)')
# plt.ylabel('Amplitude')
# plt.title('Frequency Domain')

# plt.tight_layout()
# plt.show()


###############################################################################
## intraday beta construction 
###############################################################################

# 1. beta 不同日期之间的变化太大，导致按时间取平均后相互抵消
# 2. beta 自身的计算有没有出问题
# 3. 使用rolling window计算日内beta时容易出现突变

# hf_index = pd.read_csv(r'D:\个人项目\Intrady Beta Pattern\hf_index.csv',index_col=0)
# use_stock = pd.read_csv(r'D:\个人项目\Intrady Beta Pattern\use_stock.csv',index_col=0)

# def intraday_beta(stkcd , hf_index):
#     dz, bvz, dX, bVm = Cpt_BV_and_LogClsDiff(stkcd, hf_index)[0]
#     abar = 4
#     delta = 1/48
    
#     dX = pd.DataFrame(dX,index=all_times.time.tolist(),columns=all_dates.Trddt.tolist())    
#     dz = pd.DataFrame(dz,index=all_times.time.tolist(),columns=all_dates.Trddt.tolist())    
#     bvz = pd.DataFrame(bvz, columns=all_dates.Trddt.tolist(),index=[stkcd])
#     bVm = pd.DataFrame(bVm, columns=all_dates.Trddt.tolist(),index=[stkcd])
    
#     bvz_ = abar*bvz*(delta**0.49)
#     bVm_ = abar*bVm*(delta**0.49)
    
#     # create trancate pivot table
#     def trancate(x, stkcd):
#        x[x>x[stkcd]] = 0
#        x[~(x>x[stkcd])] = 1
#        return x
#     z_bool = pd.concat([dz,bvz_]).apply(trancate,stkcd=stkcd ,axis=1)
#     z_bool = z_bool.drop(stkcd)
#     x_bool = pd.concat([dX,bVm]).apply(trancate,stkcd=stkcd ,axis=1)
#     x_bool = x_bool.drop(stkcd)
    
#     dX_ = dX * z_bool * x_bool
#     dz_ = dz * z_bool * x_bool
    
#     covXZ = dX_*dz_
#     varX = dX_*dX_

#     beta = covXZ.rolling(24).sum()/varX.rolling(24).sum()
#     beta_ = beta.dropna(how='all', axis=1)
#     beta_.max().max()
#     beta_.min().min()
    
#     beta_.describe()
    
#     beta_.mean().plot(kind='hist',  bins=200)
        
#     beta_mean_day = beta_.mean()
    
#     beta.mean(axis=1)
    
#     beta.dropna(how='all', axis=1).plot()



# # columns=['first_point','second_point','t_value','p_value']
# index_list = [6,12,24,36]
# column_list = [12,24,36,48]
# time_series = pd.DataFrame()
# for min_ in [30,60,120]:
#     test = pd.read_csv(r'D:\个人项目\Latex file\article\Intraday Market Beta Pattern\table\intraday_test_{}.csv'.format(min_),header=None)
#     time_series_test = test.copy()
#     time_series_test.index = index_list
#     time_series_test.columns = column_list
#     time_series = pd.concat([time_series,time_series_test])
# time_series.to_csv(r'D:\个人项目\Latex file\article\Intraday Market Beta Pattern\table\time_series_test.csv')


# data_dir = r'F:\HF_MIN\Resset\Csv_data\2019\stock'
# file_list = TB.Tag_FilePath_FromDir(data_dir)
# file_path = file_list[0]
# df = pd.read_csv(file_path)











