# -*- coding: utf-8 -*-
"""
Created on Mon Jul 10 15:16:18 2023

@author: asus
"""

import numpy as np
import pandas as pd
import ToolBox as TB


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
    
