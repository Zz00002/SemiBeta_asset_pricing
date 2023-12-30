# -*- coding: utf-8 -*-
"""
Created on Tue Jan 17 14:29:16 2023

@author: asus
"""

import numpy as np
import pandas as pd
import scipy
import ToolBox as TB
from matplotlib import pyplot as plt
import re
import math
import statsmodels.api as sm
from linearmodels import FamaMacBeth


# 前期处理工作，将从数据库中下载的数据表合并成大表或拆分成不同股票各自的信息表
class DataCombiner_CSMAR:
    def __init__(self, main_dir_path, init=True):
        self.main_dir_path = main_dir_path + '\\CSMAR'
        self.saved_dir_path = self.main_dir_path + '\\Combined_Data'
        self.FS_dict_sheetname = {'资产负债表': 'FS_Combas', '利润表': 'FS_Comins',
                                  '现金流量表_直接法': 'FS_Comscfd', '现金流量表_间接法': 'FS_Comscfi'}
        self.FS_dict_subject = {'资产负债表': 'A0', '利润表': 'B0',
                                '现金流量表_直接法': 'C0', '现金流量表_间接法': 'D0'}

        if init:
            # 生成合并汇总后的数据表
            self.Comb_InterRate_RiskFreeRateData(save_res=False)
            self.Comb_Stock_CompBasicData(save_res=False)
            self.Comb_Stock_SuspResumData(save_res=False)
            self.Comb_Stock_Day_TradeData(save_res=False)
            self.Comb_Stock_Mon_TradeData(save_res=False)
            for statementType in self.FS_dict_sheetname.keys():
                self.Comb_Stock_FinaStatementData(
                    statement=statementType, save_res=False)

            # 对数据进行简单的合并处理
            self.Add_Stock_Status_intoMon()
            self.Add_Stock_Listdt()

    def Comb_Index_Day_TradeData(self, dir_name='指数日交易数据', save_res=True):
        file_dir = self.main_dir_path + '\\{}'.format(dir_name)

        IndexDayTradeData_df = TB.Concat_RawData(file_dir)
        IndexDayTradeData_df.drop_duplicates(inplace=True)
        IndexDayTradeData_df.columns = [
            'Idxcd', 'Trddt', 'open', 'high', 'low', 'close', 'return']
        self.IndexDayTradeData_df = IndexDayTradeData_df

        if save_res:
            save_path = self.saved_dir_path + '\\IndexDayTradeData.csv'
            IndexDayTradeData_df.to_csv(save_path, index=False)

        print('-----Index daily trading data successfully generated-----')
        return IndexDayTradeData_df

    def Comb_Index_Compoment_Change(self, dir_name='指数成分变更', save_res=True):
        file_dir = self.main_dir_path + '\\{}'.format(dir_name)

        IndexCompChg_df = TB.Concat_RawData(file_dir)
        IndexCompChg_df.drop_duplicates(inplace=True)
        IndexCompChg_df.rename(columns={'Chgsmp01': 'ChgDate', 'Chgsmp02': 'Compcd',
                                        'Chgsmp03': 'CompName', 'Chgsmp04': 'ChgType',
                                        'Chgsmp05': 'StkType', 'Chgsmp06': 'AnnoDate'}, inplace=True)
        self.IndexCompChg_df = IndexCompChg_df

        if save_res:
            save_path = self.saved_dir_path + '\\IndexCompChg.csv'
            IndexCompChg_df.to_csv(save_path, index=False)

        print('-----Index Compoment Change data successfully generated-----')
        return IndexCompChg_df

    def Comb_InterRate_RiskFreeRateData(self, dir_name='无风险利率数据', save_res=True):
        file_dir = self.main_dir_path + '\\{}'.format(dir_name)

        riskfree_rate_df = TB.Concat_RawData(file_dir)
        riskfree_rate_df.drop_duplicates(inplace=True)
        riskfree_rate_df.rename(columns={'Clsdt': 'Trddt'}, inplace=True)
        riskfree_rate_df['Trdmnt'] = riskfree_rate_df.Trddt.str[:-3]
        self.riskfree_rate_df = riskfree_rate_df

        if save_res:
            save_path = self.saved_dir_path + '\\RiskFreeRate.csv'
            riskfree_rate_df.to_csv(save_path, index=False)

        print('-----RiskFree rate data successfully generated-----')
        return riskfree_rate_df

    def Comb_Stock_CompBasicData(self, dir_name='上市公司基本信息数据', save_res=True):
        file_dir = self.main_dir_path + '\\{}'.format(dir_name)

        StockCompBasicData_df = TB.Concat_RawData(file_dir)
        StockCompBasicData_df.drop_duplicates(inplace=True)
        self.StockCompBasicData_df = StockCompBasicData_df

        if save_res:
            save_path = self.saved_dir_path + '\\CompanyBasicData.csv'
            StockCompBasicData_df.to_csv(save_path, index=False)

        print('-----Stock company basic data successfully generated-----')
        return StockCompBasicData_df

    def Comb_Stock_SuspResumData(self, dir_name='股票停复牌数据', save_res=True):
        file_dir = self.main_dir_path + '\\{}'.format(dir_name)

        StockSuspResum_df = TB.Concat_RawData(file_dir, skip_cols=('Reason'))
        StockSuspResum_df.drop_duplicates(inplace=True)
        self.StockSuspResum_df = StockSuspResum_df

        if save_res:
            save_path = self.saved_dir_path + '\\StockSuspResumData.csv'
            StockSuspResum_df.to_csv(save_path, index=False)

        print('-----Stock suspension and resumption data successfully generated-----')
        return StockSuspResum_df

    def Comb_Stock_Day_TradeData(self, dir_name='股票日交易数据', save_res=True):
        file_dir = self.main_dir_path + '\\{}'.format(dir_name)

        StockDayTradeData_df = TB.Concat_RawData(file_dir)
        StockDayTradeData_df.drop_duplicates(inplace=True)
        self.StockDayTradeData_df = StockDayTradeData_df

        if save_res:
            save_path = self.saved_dir_path + '\\StockDayTradeData.csv'
            StockDayTradeData_df.to_csv(save_path, index=False)

        print('-----Stock daily trading data successfully generated-----')
        return StockDayTradeData_df

    # 将下载的月度数据合并成一个df，注意下载的数据格式需为csv

    def Comb_Stock_Mon_TradeData(self, dir_name='股票月交易数据', save_res=True):
        file_dir = self.main_dir_path + '\\{}'.format(dir_name)

        StockMonTradeData_df = TB.Concat_RawData(file_dir)
        StockMonTradeData_df.drop_duplicates(inplace=True)
        self.StockMonTradeData_df = StockMonTradeData_df

        if save_res:
            save_path = self.saved_dir_path + '\\StockMonTradeData.csv'
            StockMonTradeData_df.to_csv(save_path, index=False)

        print('-----Stock monthly trading data successfully generated-----')
        return StockMonTradeData_df

    def Comb_Stock_FinaStatementData(self, dir_name='财务报表数据', statement='资产负债表', save_res=True):
        file_dir = self.main_dir_path + \
            '\\{}'.format(dir_name) + '\\{}'.format(statement)

        # 只保留合并报表数据
        StockFSData_df = TB.Concat_RawData(file_dir)
        StockFSData_df = StockFSData_df[StockFSData_df.Typrep == 'A']

        # 将财务报表的科目名替换为可以识别的名称
        column_df = pd.read_csv(file_dir + '\\{}[DES][csv].txt'.format(self.FS_dict_sheetname[statement]),
                                sep=('\t'), header=None, na_values=['NaN'], comment='#')
        column_df['col_raw'] = column_df[0].str.findall('(.*) \[').str[0]
        column_df['col_translated'] = column_df[0].str.findall(
            '\[(.*)\]').str[0]
        column_df['is_subject'] = column_df[0].str[0:2]
        column_df = column_df[column_df.is_subject ==
                              self.FS_dict_subject[statement]]

        dict_df = column_df.set_index('col_raw')['col_translated']
        col_dict = dict_df.to_dict()
        StockFSData_df.rename(columns=col_dict, inplace=True)
        StockFSData_df.drop_duplicates(inplace=True)
        exec('self.StockFSData_{}_df = StockFSData_df'.format(
            self.FS_dict_sheetname[statement]))

        if save_res:
            save_path = self.saved_dir_path + \
                '\\StockFSData_{}.csv'.format(
                    self.FS_dict_sheetname[statement])
            StockFSData_df.to_csv(save_path, index=False)

        print(
            '-----Stock {} data successfully generated-----'.format(self.FS_dict_sheetname[statement]))
        return StockFSData_df

    def Add_Stock_Listdt(self, timeType=None):
        if timeType is None:
            self.StockDayTradeData_df = pd.merge(self.StockDayTradeData_df,
                                                 self.StockCompBasicData_df[[
                                                     'Stkcd', 'Listdt']],
                                                 left_on='Stkcd', right_on='Stkcd', how='left')

            self.StockMonTradeData_df = pd.merge(self.StockMonTradeData_df,
                                                 self.StockCompBasicData_df[[
                                                     'Stkcd', 'Listdt']],
                                                 left_on='Stkcd', right_on='Stkcd', how='left')
        elif timeType.upper() == 'M':
            self.StockMonTradeData_df = pd.merge(self.StockMonTradeData_df,
                                                 self.StockCompBasicData_df[[
                                                     'Stkcd', 'Listdt']],
                                                 left_on='Stkcd', right_on='Stkcd', how='left')

        elif timeType.upper() == 'D':
            self.StockDayTradeData_df = pd.merge(self.StockDayTradeData_df,
                                                 self.StockCompBasicData_df[[
                                                     'Stkcd', 'Listdt']],
                                                 left_on='Stkcd', right_on='Stkcd', how='left')

        print('-----Stock listdt data successfully added into trade data-----')

    def Add_Stock_Trdmnt_intoDay(self):
        self.StockDayTradeData_df['Trdmnt'] = self.StockDayTradeData_df['Trddt'] \
                                                  .apply(lambda date: '-'.join(date.split('-')[:-1]))

        print('-----Stock Trdmnt data successfully added into daily trade data-----')

    # 以股票在当月最后一天的状态作为股票在当月的状态（有待商榷）

    def Add_Stock_Status_intoMon(self):
        self.Add_Stock_Trdmnt_intoDay()
        Trdsta_df = self.StockDayTradeData_df[['Stkcd', 'Trdmnt', 'Trdsta']]
        Trdsta_df = Trdsta_df.drop_duplicates(['Stkcd', 'Trdmnt'], keep='last')

        self.StockMonTradeData_df = pd.merge(self.StockMonTradeData_df, Trdsta_df,
                                             left_on=['Stkcd', 'Trdmnt'],
                                             right_on=['Stkcd', 'Trdmnt'], how='left')

        print('-----Stock status data successfully added into monthly trade data-----')


class DataCombiner_RESSET:
    def __init__(self, main_dir_path, init=True):
        self.main_dir_path = main_dir_path + '\\RESSET'
        self.saved_dir_path = self.main_dir_path + '\\Combined_Data'
    
    
    def Comb_Stock_FinReportInfo(self, dir_name='财务报表数据\\财务报表发布数据', save_res=True):
        
        file_dir = self.main_dir_path + '\\{}'.format(dir_name)

        ReportInfo_df = TB.Concat_RawData(file_dir)
        ReportInfo_df.drop_duplicates(inplace=True)
        ReportInfo_df = ReportInfo_df.dropna(axis=1, how='all')
        
        self.ReportInfo_df = ReportInfo_df

        if save_res:
            save_path = self.saved_dir_path + '\\ReportInfo.csv'
            ReportInfo_df.to_csv(save_path, index=False)

        print('-----Financial Report Information data successfully generated-----')
        return ReportInfo_df

    

# 后期如果有不同数据来源的数据，可以针对性分别设计不同的数据合并类后汇总在这个类中
# 感觉不要这个页没有关系
class DataCombiner:

    def __init__(self, main_dir_path, init=True):
        self._CSMAR = DataCombiner_CSMAR(main_dir_path, init)


# 构建进行前期数据预处理的数据处理类
# 分为日和月两种情况
class DataCleaner(DataCombiner):
    """
    >>> main_dir_path = 'F:\\数据集合\\学术研究_股票数据'
    """

    def Select_Stock_GivenDayData(self, begDate=None, endDate=None,
                                  df=None, timeType='M', inplace=False):

        begMon = begDate[:-3] if begDate else None
        endMon = endDate[:-3] if endDate else None

        if df is None:
            if begDate is None and endDate is None:
                StockDayTradeData_df = self._CSMAR.StockDayTradeData_df
                StockMonTradeData_df = self._CSMAR.StockMonTradeData_df
            elif begDate is None and endDate is not None:
                StockDayTradeData_df = self._CSMAR.StockDayTradeData_df[
                    self._CSMAR.StockDayTradeData_df.Trddt <= endDate]
                StockMonTradeData_df = self._CSMAR.StockMonTradeData_df[
                    self._CSMAR.StockMonTradeData_df.Trdmnt <= endMon]
            elif begDate is not None and endDate is None:
                StockDayTradeData_df = self._CSMAR.StockDayTradeData_df[
                    self._CSMAR.StockDayTradeData_df.Trddt >= begDate]
                StockMonTradeData_df = self._CSMAR.StockMonTradeData_df[
                    self._CSMAR.StockMonTradeData_df.Trdmnt >= begMon]
            elif begDate is not None and endDate is not None:
                StockDayTradeData_df = self._CSMAR.StockDayTradeData_df[(
                    self._CSMAR.StockDayTradeData_df.Trddt >= begDate) & (self._CSMAR.StockDayTradeData_df.Trddt <= endDate)]
                StockMonTradeData_df = self._CSMAR.StockMonTradeData_df[(
                    self._CSMAR.StockMonTradeData_df.Trdmnt >= begMon) & (self._CSMAR.StockMonTradeData_df.Trdmnt <= endMon)]

            if inplace:
                self._CSMAR.StockDayTradeData_df = StockDayTradeData_df
                self._CSMAR.StockMonTradeData_df = StockMonTradeData_df
            return (StockDayTradeData_df, StockMonTradeData_df)

        else:
            if timeType.upper() == 'D':
                if begDate is None and endDate is None:
                    StockDayTradeData_df = df
                elif begDate is None and endDate is not None:
                    StockDayTradeData_df = df[df.Trddt <= endDate]
                elif begDate is not None and endDate is None:
                    StockDayTradeData_df = df[df.Trddt >= begDate]
                elif begDate is not None and endDate is not None:
                    StockDayTradeData_df = df[(
                        df.Trddt >= begDate) & (df.Trddt <= endDate)]

                if inplace:
                    self._CSMAR.StockDayTradeData_df = StockDayTradeData_df
                return StockDayTradeData_df

            elif timeType.upper() == 'M':
                if begDate is None and endDate is None:
                    StockMonTradeData_df = df
                elif begDate is None and endDate is not None:
                    StockMonTradeData_df = df[df.Trdmnt <= endMon]
                elif begDate is not None and endDate is None:
                    StockMonTradeData_df = df[df.Trdmnt >= begMon]
                elif begDate is not None and endDate is not None:
                    StockMonTradeData_df = df[(
                        df.Trdmnt >= begMon) & (df.Trdmnt <= endMon)]

                if inplace:
                    self._CSMAR.StockMonTradeData_df = StockMonTradeData_df
                return StockMonTradeData_df

        print('-----Successfully select stock data {} to {}-----'.format(begDate, endDate))

    def Select_Stock_ExchangeTradBoard(self, board_list=[1, 4, 16, 32],
                                       df=None, timeType='M', inplace=False):

        if df is None:
            StockDayTradeData_df = self._CSMAR.StockDayTradeData_df[self._CSMAR.StockDayTradeData_df.Markettype.isin(
                board_list)]
            StockMonTradeData_df = self._CSMAR.StockMonTradeData_df[self._CSMAR.StockMonTradeData_df.Markettype.isin(
                board_list)]

            if inplace:
                self._CSMAR.StockDayTradeData_df = StockDayTradeData_df
                self._CSMAR.StockMonTradeData_df = StockMonTradeData_df
            return (StockDayTradeData_df, StockMonTradeData_df)

        else:
            if timeType.upper() == 'D':
                StockDayTradeData_df = df[df.Markettype.isin(board_list)]
                if inplace:
                    self._CSMAR.StockDayTradeData_df = StockDayTradeData_df
                return StockDayTradeData_df

            elif timeType.upper() == 'M':
                StockMonTradeData_df = df[df.Markettype.isin(board_list)]
                if inplace:
                    self._CSMAR.StockMonTradeData_df = StockMonTradeData_df
                return StockMonTradeData_df

        print('-----Successfully select stock data that belong to {} board-----'
              .format(','.join(list(map(str, board_list)))))

    def Drop_Stock_SuspOverRet(self, df=None, timeType='M', inplace=False):

        StockDayTradeData_df = self._CSMAR.StockDayTradeData_df.copy()

        drop_1 = (StockDayTradeData_df.Markettype.isin([1, 4])) & \
                 (StockDayTradeData_df.Trddt >= '1996-12-15') & \
                 (abs(StockDayTradeData_df.Dretwd) >= 0.11)
        drop_2 = (StockDayTradeData_df.Markettype.isin([32])) & \
                 (abs(StockDayTradeData_df.Dretwd) >= 0.21)
        drop_3 = (StockDayTradeData_df.Markettype.isin([16])) & \
                 (abs(StockDayTradeData_df.Dretwd) >= 0.21) & \
                 (StockDayTradeData_df.Trddt >= '2020-06-20')
        drop_4 = (StockDayTradeData_df.Markettype.isin([16])) & \
                 (abs(StockDayTradeData_df.Dretwd) >= 0.11) & \
                 (StockDayTradeData_df.Trddt < '2020-06-20')
        drop_data = StockDayTradeData_df[drop_1 | drop_2 | drop_3 | drop_4]
        drop_data['sign'] = 1
        drop_data = drop_data[['Stkcd', 'Trddt', 'Trdmnt', 'sign']]

        if df is None:
            StockDayTradeData_df = self._CSMAR.StockDayTradeData_df.drop(
                drop_data.index).copy()
            StockDayTradeData_df = StockDayTradeData_df.drop(['sign'], axis=1)

            StockMonTradeData_df = pd.merge(self._CSMAR.StockMonTradeData_df, drop_data,
                                            left_on=['Stkcd', 'Trdmnt'], right_on=['Stkcd', 'Trdmnt'], how='left')
            StockMonTradeData_df = StockMonTradeData_df[~(
                StockMonTradeData_df.sign == 1)]
            StockMonTradeData_df = StockMonTradeData_df.drop(
                ['sign', 'Trddt'], axis=1)

            if inplace:
                self._CSMAR.StockDayTradeData_df = StockDayTradeData_df
                self._CSMAR.StockMonTradeData_df = StockMonTradeData_df

            return (StockDayTradeData_df, StockMonTradeData_df)

        else:
            if timeType.upper() == 'D':
                StockDayTradeData_df = pd.merge(df, drop_data, left_on=['Stkcd', 'Trdmnt'],
                                                right_on=['Stkcd', 'Trdmnt'], how='left')
                StockDayTradeData_df = StockDayTradeData_df[~(
                    StockDayTradeData_df.sign == 1)]
                StockDayTradeData_df = StockDayTradeData_df.drop(
                    ['sign'], axis=1)
                if inplace:
                    self._CSMAR.StockDayTradeData_df = StockDayTradeData_df
                return StockDayTradeData_df

            elif timeType.upper() == 'M':
                StockMonTradeData_df = pd.merge(df, drop_data, left_on=['Stkcd', 'Trdmnt'],
                                                right_on=['Stkcd', 'Trdmnt'], how='left')
                StockMonTradeData_df = StockMonTradeData_df[~(
                    StockMonTradeData_df.sign == 1)]
                StockMonTradeData_df = StockMonTradeData_df.drop(
                    ['sign', 'Trddt'], axis=1)

                if inplace:
                    self._CSMAR.StockMonTradeData_df = StockMonTradeData_df
                return StockMonTradeData_df

        print('-----long suspension stock that in resumption day had abnormal ret successfully droped-----')

    def Drop_Stock_STData(self, state_list=[1, 4, 7, 10], df=None, timeType='M', inplace=False):

        if df is None:
            StockDayTradeData_df = self._CSMAR.StockDayTradeData_df[self._CSMAR.StockDayTradeData_df.Trdsta.isin(
                state_list)]
            StockMonTradeData_df = self._CSMAR.StockMonTradeData_df[self._CSMAR.StockMonTradeData_df.Trdsta.isin(
                state_list)]

            if inplace:
                self._CSMAR.StockDayTradeData_df = StockDayTradeData_df
                self._CSMAR.StockMonTradeData_df = StockMonTradeData_df
            return (StockDayTradeData_df, StockMonTradeData_df)

        else:
            if timeType.upper() == 'D':
                StockDayTradeData_df = df[df.Trdsta.isin(state_list)]
                if inplace:
                    self._CSMAR.StockDayTradeData_df = StockDayTradeData_df
                return StockDayTradeData_df

            elif timeType.upper() == 'M':
                StockMonTradeData_df = df[df.Trdsta.isin(state_list)]
                if inplace:
                    self._CSMAR.StockMonTradeData_df = StockMonTradeData_df
                return StockMonTradeData_df

        print('-----ST stock successfully droped-----')

    def Drop_Stock_InadeListdtData(self, drop_Mon_num=6, df=None, timeType='M', inplace=False):

        def filter_listMon(df, col1, col2, X):
            df_use = df.copy()
            df_use['timeTag1'] = pd.to_datetime(df_use[col1], format='%Y-%m')
            df_use['timeTag2'] = pd.to_datetime(
                df_use[col2], format='%Y-%m-%d')

            df_use['month_diff'] = (df_use['timeTag1'].dt.year - df_use['timeTag2'].dt.year) * 12 + \
                (df_use['timeTag1'].dt.month - df_use['timeTag2'].dt.month)

            df_return = df_use[df_use['month_diff'] > X]
            df_return = df_return.drop(
                ['month_diff', 'timeTag1', 'timeTag2'], axis=1)
            return df_return

        if df is None:
            StockDayTradeData_df = filter_listMon(self._CSMAR.StockDayTradeData_df,
                                                  'Trdmnt', 'Listdt', drop_Mon_num)
            StockMonTradeData_df = filter_listMon(self._CSMAR.StockMonTradeData_df,
                                                  'Trdmnt', 'Listdt', drop_Mon_num)

            if inplace:
                self._CSMAR.StockDayTradeData_df = StockDayTradeData_df
                self._CSMAR.StockMonTradeData_df = StockMonTradeData_df
            return (StockDayTradeData_df, StockMonTradeData_df)

        else:
            if timeType.upper() == 'D':
                StockDayTradeData_df = filter_listMon(
                    df, 'Trdmnt', 'Listdt', drop_Mon_num)
                if inplace:
                    self._CSMAR.StockDayTradeData_df = StockDayTradeData_df
                return StockDayTradeData_df

            elif timeType.upper() == 'M':
                StockMonTradeData_df = filter_listMon(
                    df, 'Trdmnt', 'Listdt', drop_Mon_num)
                if inplace:
                    self._CSMAR.StockMonTradeData_df = StockMonTradeData_df
                return StockMonTradeData_df

        print('-----Successfully drop stock data that List less than {} month-----'.format(drop_Mon_num))

    # 由于停牌超过一天的股票可能会出现异常收益率，因此把其剔除

    def Drop_Stock_ResumFirstDayData(self, df=None, inplace=False):

        Resum_info_df = self._CSMAR.StockSuspResum_df.dropna().copy()
        Resum_info_df = Resum_info_df[Resum_info_df.Suspdate !=
                                      Resum_info_df.Resmdate]
        Resum_info_df = Resum_info_df[['Stkcd', 'Resmdate']]
        Resum_info_df['Resum_tag'] = 1
        Resum_info_df.rename(columns={'Resmdate': 'Trddt'}, inplace=True)

        if df is None:
            StockDayTradeData_df = pd.merge(self._CSMAR.StockDayTradeData_df, Resum_info_df,
                                            left_on=['Stkcd', 'Trddt'], right_on=['Stkcd', 'Trddt'], how='left')
            StockDayTradeData_df = StockDayTradeData_df[~(
                StockDayTradeData_df.Resum_tag == 1)]
            StockDayTradeData_df = StockDayTradeData_df.drop(
                'Resum_tag', axis=1)

            if inplace:
                self._CSMAR.StockDayTradeData_df = StockDayTradeData_df
            return StockDayTradeData_df

        else:
            StockDayTradeData_df = pd.merge(df, Resum_info_df,
                                            left_on=['Stkcd', 'Trddt'], right_on=['Stkcd', 'Trddt'], how='left')
            StockDayTradeData_df = StockDayTradeData_df[~(
                StockDayTradeData_df.Resum_tag == 1)]
            StockDayTradeData_df = StockDayTradeData_df.drop(
                'Resum_tag', axis=1)

            if inplace:
                self._CSMAR.StockDayTradeData_df = StockDayTradeData_df
            return StockDayTradeData_df

        print("-----Successfully drop stock's first resumption day data-----")

    def Drop_Stock_MVData(self, mv_quantile=0.3, df=None, timeType='M', inplace=False):

        def filter_MVQuanti(df, mv_quantile):
            StockMonMV = self._CSMAR.StockMonTradeData_df.copy()
            MonMV = StockMonMV.pivot('Trdmnt', 'Stkcd', 'Msmvttl')
            MV_sign = MonMV.sub(MonMV.quantile(mv_quantile, axis=1), axis=0)

            MV_sign = MV_sign.apply(lambda x: np.where(x.values > 0, 1, 0)) \
                .stack().reset_index().rename(columns={0: 'sign'})
            MV_sign = MV_sign[MV_sign.sign == 1]

            df_return = pd.merge(df, MV_sign)
            df_return.drop('sign', axis=1, inplace=True)
            return df_return

        if df is None:
            StockDayTradeData_df = filter_MVQuanti(self._CSMAR.StockDayTradeData_df,
                                                   mv_quantile)
            StockMonTradeData_df = filter_MVQuanti(self._CSMAR.StockMonTradeData_df,
                                                   mv_quantile)

            if inplace:
                self._CSMAR.StockDayTradeData_df = StockDayTradeData_df
                self._CSMAR.StockMonTradeData_df = StockMonTradeData_df
            return (StockDayTradeData_df, StockMonTradeData_df)

        else:
            if timeType.upper() == 'D':
                StockDayTradeData_df = filter_MVQuanti(df, mv_quantile)
                if inplace:
                    self._CSMAR.StockDayTradeData_df = StockDayTradeData_df
                return StockDayTradeData_df

            elif timeType.upper() == 'M':
                StockMonTradeData_df = filter_MVQuanti(df, mv_quantile)
                if inplace:
                    self._CSMAR.StockMonTradeData_df = StockMonTradeData_df
                return StockMonTradeData_df

        print('-----Successfully drop stock data that MV less than {} quantile-----'.format(mv_quantile))

    def Drop_Stock_InadeTradData_(self, threshold=2/3, df=None, inplace=False):

        if df is None:
            StockMonTradeData = self._CSMAR.StockMonTradeData_df.copy()
        else:
            StockMonTradeData = df.copy()

        legal_trade_day = StockMonTradeData.groupby(
            'Trdmnt').Ndaytrd.max().rename('legal_trade_day')
        StockMonTradeData = pd.merge(
            StockMonTradeData, legal_trade_day, left_on='Trdmnt', right_index=True)
        StockMonTradeData = StockMonTradeData[StockMonTradeData.Ndaytrd >=
                                              StockMonTradeData.legal_trade_day * threshold]

        if inplace:
            self._CSMAR.StockMonTradeData_df = StockMonTradeData

        return StockMonTradeData

        print('-----Successfully drop stock data that can not need the required trade day number-----')

    def Drop_Stock_InadeTradData(self, Montreshold=15, Yeartreshold=120,
                                 df=None, timeType='M', inplace=False):

        StockDayTag_df = self._CSMAR.StockDayTradeData_df.copy()
        StockDayTag_df['Trdsign'] = 1

        StockDayTag_pivot = StockDayTag_df.pivot('Trddt', 'Stkcd', 'Trdsign')
        stock_list = StockDayTag_pivot.columns

        StockDayTag_pivot = StockDayTag_pivot.reset_index()
        StockDayTag_pivot['Trdmnt'] = StockDayTag_pivot.Trddt.apply(
            lambda date: '-'.join(date.split('-')[:-1]))

        StockMonTag_pivot = StockDayTag_pivot.drop_duplicates(
            'Trdmnt', keep='last')
        month_lastday_list = StockMonTag_pivot.index

        # 这里太慢了，可以优化
        for stock in stock_list:
            for month_lastday in month_lastday_list:
                StockMonData = StockDayTag_pivot.loc[month_lastday -
                                                     19:month_lastday, stock]
                StockYearData = StockDayTag_pivot.loc[month_lastday -
                                                      249:month_lastday, stock]
                if StockMonData.sum() < Montreshold or StockYearData.sum() < Yeartreshold:
                    StockMonTag_pivot.loc[month_lastday, stock] = np.nan

        StockMonTag_pivot.index = StockMonTag_pivot.Trdmnt
        StockMonTag_pivot = StockMonTag_pivot.drop(['Trddt', 'Trdmnt'], axis=1)
        StockMonTag_df = StockMonTag_pivot.unstack(
        ).reset_index().rename(columns={0: 'sign'})
        StockMonTag_df = StockMonTag_df[StockMonTag_df.sign == 1]
        StockMonTag_df = StockMonTag_df.drop('sign', axis=1)

        if df is None:
            StockDayTradeData_df = pd.merge(self._CSMAR.StockDayTradeData_df, StockMonTag_df,
                                            left_on=['Stkcd', 'Trdmnt'], right_on=['Stkcd', 'Trdmnt'])
            StockMonTradeData_df = pd.merge(self._CSMAR.StockMonTradeData_df, StockMonTag_df,
                                            left_on=['Stkcd', 'Trdmnt'], right_on=['Stkcd', 'Trdmnt'])

            if inplace:
                self._CSMAR.StockDayTradeData_df = StockDayTradeData_df
                self._CSMAR.StockMonTradeData_df = StockMonTradeData_df
            return (StockDayTradeData_df, StockMonTradeData_df)

        else:
            if timeType.upper() == 'D':
                StockDayTradeData_df = pd.merge(df, StockMonTag_df,
                                                left_on=['Stkcd', 'Trdmnt'], right_on=['Stkcd', 'Trdmnt'])
                if inplace:
                    self._CSMAR.StockDayTradeData_df = StockDayTradeData_df
                return StockDayTradeData_df

            elif timeType.upper() == 'M':
                StockMonTradeData_df = pd.merge(df, StockMonTag_df,
                                                left_on=['Stkcd', 'Trdmnt'], right_on=['Stkcd', 'Trdmnt'])
                if inplace:
                    self._CSMAR.StockMonTradeData_df = StockMonTradeData_df
                return StockMonTradeData_df

        print('-----Successfully drop stock data that can not need the required trade day number-----')

    def Recon_Data_SVIC(self,):
        """
        >>> Data_Source.Comb_Stock_Mon_TradeData()
        >>> Data_Source.Comb_Stock_Day_TradeData()
        >>> Data_Source.Add_Stock_Status_intoMon()
        >>> Data_Source.Comb_Stock_CompBasicData()
        >>> Data_Source.Add_Stock_Listdt()

        """
        # I. Identify Stock's belong to which factor's group
        # get stocks' month trade data
        Data_Source = self._CSMAR
        stock_mon_trade_data = Data_Source.StockMonTradeData_df.copy()

        # get stocks' last month MV data of each year
        stock_mon_trade_data['Trdyr'] = stock_mon_trade_data['Trdmnt'].str.extract(
            '(\d{4})')
        # mv_lastmon_data = stock_mon_trade_data.groupby(['Stkcd', 'Trdyr']).tail(1)[['Stkcd','Trdyr','Msmvttl']].copy()

        # Do the data cleaning process
        # 1. drop the data that don't meet the trade day number requirement
        # 2. select specific ExchangeTradBoard data
        # 3. drop stocks that have been tagged ST
        # 4. drop stocks whose listdt less than 6 months
        # 5. drop stocks whose have abnormal ret(mostly due to suspension)
        stock_mon_trade_data = self.Select_Stock_ExchangeTradBoard(
            df=stock_mon_trade_data, board_list=[1, 4, 16, 32])
        stock_mon_trade_data = self.Drop_Stock_STData(df=stock_mon_trade_data)
        stock_mon_trade_data = self.Drop_Stock_InadeListdtData(
            df=stock_mon_trade_data, drop_Mon_num=6)
        # stock_mon_trade_data = self.Drop_Stock_MVData(df=stock_mon_trade_data)
        stock_mon_trade_data = self.Drop_Stock_InadeTradData_(
            df=stock_mon_trade_data)
        stock_mon_trade_data = self.Drop_Stock_SuspOverRet(
            df=stock_mon_trade_data)

        self.StockData_SVIC = stock_mon_trade_data.copy()
        # stock_mon_trade_data.to_csv(r'F:\数据集合\学术研究_股票数据\CSMAR\Combined_Data\SAVIC_saveMV.csv')
        # stock_mon_trade_data.to_csv(r'F:\数据集合\学术研究_股票数据\CSMAR\Combined_Data\SAVIC_deMV.csv')
        # Data_Source.StockDayTradeData_df
        pd.merge(Data_Source.StockDayTradeData_df, stock_mon_trade_data).to_csv(
            r'F:\数据集合\学术研究_股票数据\CSMAR\Combined_Data\SAVIC_saveMV_day.csv', index=False)
        return stock_mon_trade_data

# 构建用于进行资产定价研究时所需要用到的各种工具类


class AssetPricingTool:

    def __init__(self,
                 df=None):

        if type(df) == pd.core.frame.DataFrame:
            self.data_set = df.copy()
        else:
            self.data_set = pd.DataFrame(df)

    # 进行Newey-West Test

    def Exec_NWTest(self,
                    test_array=None,
                    L=None):

        if test_array is None:
            test_array = self.data_set

        if L == None:
            L = int(4*pow(len(test_array)/100, 2/9))

        Y = np.array(test_array)
        mean = Y.mean()
        res = Y - mean
        T = len(Y)

        S = 0
        for l in range(1, L + 1):
            w_l = 1 - l / (L + 1)
            for t in range(l + 1, T + 1):
                S += w_l * res[t - 1] * res[t - 1 - l] * 2
        S = S + (res * res).sum()
        S = S / (T - 1)

        se = np.sqrt(S / T)
        tstat = mean / se
        pval = scipy.stats.t.sf(np.abs(tstat), T - 1) * 2

        test_result = {'mean': round(mean, 6),
                       'se': round(se, 6),
                       'tstat': round(tstat, 6),
                       'pval': round(pval, 6)}

        return test_result

    # 计算OLS残差和beta

    def Cpt_ResAndBeta(self,
                       x_tag,
                       y_tag,
                       df=None,
                       intercept=True,
                       NWtest=False):
        '''
        给定一个数据集，指定自变量和因变量的数据标签后，计算OLS回归的残差

        Parameters
        ----------
        data_set : dataframe
            包含因变量和自变量数据的数据集.
        x_tag : list
            自变量在data_set中所对应的列标签，列表内为字符串格式的标签名称.
        y_tag : str
            因变量在data_set中对应的列标签.  
        intercept : bool
            设定回归是否包含常数项，默认为True，即包含常数项

        Returns
        -------
        res : ndarray
            OLS回归残差序列.
        '''
        # 判断是否传入了df参数，如果没有，则使用类自带的数据集
        if df is None:
            df = self.data_set

        X = pd.DataFrame(columns=x_tag)
        # 矩阵法求解残差
        X[x_tag] = df[x_tag]
        if intercept == True:
            X['intercept'] = 1

        X = np.mat(X)
        Y = np.mat(df[[y_tag]])
        try:
            beta = ((X.T.dot(X)).I).dot(X.T).dot(Y)
            res = Y - X.dot(beta)

        except:
            # 如果出现矩阵不可逆的情况，则给主对角线上的元素均加上一个很小的数
            # 从而使矩阵可逆
            mat = X.T.dot(X)
            zeros = np.zeros((mat.shape))

            for i in range(mat.shape[0]):
                zeros[i][i] = zeros[i][i] + 0.00001
            beta = ((X.T.dot(X) + zeros).I).dot(X.T).dot(Y)
            res = Y - X.dot(beta)

        if NWtest is True:
            long = df.shape[0]
            L = int(4*pow(long/100, 2/9))

            S = 0
            for l in range(1, L+1):
                w_l = 1-l/(L+1)
                for t in range(l+1, long+1):
                    S += float((w_l * res[t-1]*res[t-1-l])) * \
                        (X[t-1].T.dot(X[t-1-l])+X[t-1-l].T.dot(X[t-1]))

            for t in range(1, long+1):
                S += float(res[t-1]*res[t-1]) * (X[t-1].T.dot(X[t-1]))
            S = S/long
            V = long*(X.T.dot(X)).I * S * (X.T.dot(X)).I

            return {'res': res, 'beta': beta, 'NWse': V}
        else:
            return {'res': res, 'beta': beta}

    def FamaMacBeth_self(self,
                         DF,
                         reg_lst,
                         reg_order,
                         reg_names=None,
                         params_format='{:.3f}',
                         tvalues_format='{:.2f}'):
        '''
        A function for Fama-MacBeth regression and results summary.

        Parameters
        ----------
        DF: DataFrame
            A panel date of which multi-index is stock and month (datetime64[ns]),
            containing all the dependent and independent variables.
        reg_lst: list
            A list containing multiple lists of dependent variable and independent
            variables, e.g., [['Y', 'X1', ...],..., ['Y', 'X1', ...,]].
        reg_order: list
            The order of independent variables in result table.
        reg_names: list
            The names for each regression.
        params_format: str
            The number of decimal places for parameters, e.g., '{:.3f}'.
        tvalues_format: str
            The number of decimal places for t-values, e.g., '{:.2f}'.
        '''

        def getOLS(group, x_var, y_var):

            X = group[x_var]
            y = group[y_var]

            X = X.assign(const=1)
            res = sm.OLS(y, X).fit()

            return res.params
            # return res

        # Create a DataFrame
        rows = sum([[var, f'{var}_t'] for var in ['const'] + reg_order], [])
        if reg_names is None:
            reg_names = [f'({i+1})' for i in range(len(reg_lst))]
        show = pd.DataFrame(index=rows, columns=reg_names)

        res_list = []
        for reg, reg_name in zip(reg_lst, reg_names):

            df = DF.loc[:, reg].copy().dropna()

            T = len(df.index.get_level_values(df.index.names[1]).unique())
            lag = math.floor(4*(T/100)**(2/9))

            res = df.groupby(['Trddt']).apply(getOLS, reg[1:], reg[0])
            res_list.append(res)
            print(res)

            df.groupby('Trddt').count()

            # params, tvalues(tstats) and pvalues
            params = fmb.params
            tvalues = fmb.tstats
            pvalues = fmb.pvalues

            # Obs.
            total_obs = df.shape[0]
            # mean_obs = fmb.time_info['mean']

            # average rsquared_adj
            dft = df.reset_index(level=df.index.names[0], drop=True).copy()
            rsquared_adj = []
            for month in dft.index.unique():
                dftm = dft.loc[month].copy()
                ols = sm.OLS(dftm[reg[0]], sm.add_constant(
                    dftm[reg[1:]])).fit()
                rsquared_adj.append(ols.rsquared_adj)
            ar2a = np.mean(rsquared_adj)

            # params and significance
            ps_lst = []
            for param, pvalue in zip(params, pvalues):
                param = params_format.format(param)
                if (pvalue <= 0.1) & (pvalue > 0.05):
                    param = param + '*'
                elif (pvalue <= 0.05) & (pvalue > 0.01):
                    param = param + '**'
                elif pvalue <= 0.01:
                    param = param + '***'
                ps_lst.append(param)

            # params and tvalues
            tvalues = [tvalues_format.format(t) for t in tvalues]
            t_lst = [f'{t}' for t in tvalues]
            pt_lst = [[i, j] for i, j in zip(ps_lst, t_lst)]

            # put them in place
            for var, pt in zip(['const'] + reg[1:], pt_lst):
                show.loc[var, reg_name] = pt[0]
                show.loc[f'{var}_t', reg_name] = pt[1]
            show.loc['No. Obs.', reg_name] = str(total_obs)
            show.loc['Adj. R2', reg_name] = '{:.2f}%'.format(ar2a * 100)

        rename_index = sum([[var, '']
                           for var in ['Intercept'] + reg_order], [])
        show.index = rename_index + ['No. Obs.', 'Adj. R2']

        return show.dropna(axis=0, how='all').fillna('')

    def Shw_Reg_Result(reg_res):
        pass

    ###########################################################################
    ######################### 与股票sorting有关 ###############################
    ###########################################################################

    ########################### 中介函数，必须传入df#############################
    # 计算加权平均值
    def Cpt_ValueWeight(self,
                        df,
                        avgTag,
                        weightTag):

        data = df[avgTag]
        weight = df[weightTag]

        try:
            return (data * weight).sum() / weight.sum()
        except ZeroDivisionError:
            return np.nan

    # 计算DataFrame中每一个Group的截面均值

    def Cpt_CrossWeightMean(self,
                            df,
                            avgTag,
                            timeTag,
                            weightTag=None):

        # 如果没有要用于计算权重的列，则默认为等权重
        if weightTag == None:
            # 将df按时间和组别分组后，对每组需要进行展示的变量进行截面维度上的聚合取mean
            ew = df.groupby([timeTag, 'Group'], as_index=False)[avgTag].mean()
            ew = ew.set_index(timeTag)
            return ew

        else:
            vw = df.groupby([timeTag, 'Group'], as_index=False) \
                .apply(self.Cpt_ValueWeight, avgTag, weightTag) \
                .rename(columns={None: avgTag})
            vw = vw.set_index(timeTag)
            return vw

    # 计算股票分组

    def Creat_StockGroups(self,
                          data,
                          sortTag,
                          groupsNum,
                          labels=None):

        df = data.copy()
        # 生成记录分组的标签
        if labels is None:
            if type(groupsNum) is int:
                labels = ['{}_G0'.format(sortTag) + str(i)
                          for i in range(1, groupsNum + 1)]
            else:
                labels = ['{}_G0'.format(sortTag) + str(i)
                          for i in range(1, len(groupsNum))]
        try:
            # 使用qcut将股票按照分位数划分为给定数量的分组
            groups = pd.DataFrame(pd.qcut(df[sortTag], groupsNum, labels=labels).astype(str)) \
                .rename(columns={sortTag: 'Group'})
        except:
            groups = pd.DataFrame(pd.qcut(df[sortTag].rank(method='first'), groupsNum, labels=labels).astype(str)) \
                .rename(columns={sortTag: 'Group'})
        return groups

    # 对DataFrame执行计算分组操作

    def Exec_GroupStockDf(self,
                          df,
                          groupby_list,
                          sortTag,
                          groupsNum,
                          labels=None):

        # 按照需要进行分组的标签设置多重索引
        df.index = pd.MultiIndex.from_frame(df[groupby_list])
        df = df.sort_index()

        # 将需要分组指标进行分组
        df['Group'] = ''
        for temp in df.groupby(df.index):
            df_temp = temp[1]
            # 获取分组
            group_temp = self.Creat_StockGroups(
                df_temp, sortTag, groupsNum, labels)
            index_temp = temp[0]
            # 写入分组信息
            df.loc[index_temp, 'Group'] = group_temp

        df.reset_index(drop=True, inplace=True)
        return df

    ###########################################################################

    # 获取单分组结果
    def Exec_SingleSort(self,
                        avgTag,
                        sortTag,
                        groupsNum,
                        timeTag='Trdmnt',
                        df=None,
                        weightTag=None,
                        labels=None):

        if df is None:
            df = self.data_set
        df = self.Exec_GroupStockDf(df, [timeTag], sortTag, groupsNum, labels)
        result = self.Cpt_CrossWeightMean(df, avgTag, timeTag, weightTag)
        result[avgTag] = result[avgTag]
        result = result.sort_values([timeTag, 'Group'])

        return (result, df)

    # 获取双重分组结果
    def Exec_DoubleSort(self,
                        avgTag,
                        sortTag1,
                        sortTag2,
                        groupsNum1,
                        groupsNum2,
                        labels1=None,
                        labels2=None,
                        timeTag='Trdmnt',
                        df=None,
                        weightTag=None,
                        SortMethod='Independent'):

        if df is None:
            df = self.data_set.copy()
        # 单分组计算
        df = self.Exec_GroupStockDf(
            df, [timeTag], sortTag1, groupsNum1, labels1)
        df.rename(columns={'Group': 'G1'}, inplace=True)

        # 在上一次单分组的结果上计算第二次分组
        if SortMethod == 'Independent':
            df = self.Exec_GroupStockDf(
                df, [timeTag], sortTag2, groupsNum2, labels2)
        else:
            df = self.Exec_GroupStockDf(
                df, [timeTag, 'G1'], sortTag2, groupsNum2, labels2)
        df.rename(columns={'Group': 'G2'}, inplace=True)

        # 将两个变量的分组用@来联系起来，便于聚合后再将分组还原
        df['Group'] = df.G1 + '@' + df.G2
        # 计算每个Group的截面均值
        result = self.Cpt_CrossWeightMean(df, avgTag, timeTag, weightTag)
        result[avgTag] = result[avgTag]
        result['Group_' +
               sortTag1] = result.Group.apply(lambda x: x.split('@')[0])
        result['Group_' +
               sortTag2] = result.Group.apply(lambda x: x.split('@')[1])
        result = result.sort_values([timeTag, 'Group'])

        return (result, df)

    # 将截面分组聚合的结果再时序上再进行聚合，并生成二维的数据表

    def Creat_SortTimePolymer(self,
                              df=None,
                              avgTag=None,
                              sortTag1=None,
                              sortTag2=None,
                              sortType='D'):

        if df is None:
            df = self.data_set

        df = df.groupby('Group', as_index=False).mean()
        if sortType.upper() == 'D':
            # 将加总的分组信息列再分解为两列后，再转化为二维格式
            df['Group_' +
                sortTag1] = df['Group'].apply(lambda x: x.split('@')[0])
            df['Group_' +
                sortTag2] = df['Group'].apply(lambda x: x.split('@')[1])
            df.drop('Group', inplace=True, axis=1)
            df = df.pivot('Group_'+sortTag1, 'Group_'+sortTag2, avgTag)
        else:
            df.set_index('Group', inplace=True)
        return df

    def Rec_SingleSortRes(self,
                          avgTag,
                          sortTag,
                          groupsNum,
                          Factors,
                          use_factors,
                          timeTag='Trdmnt',
                          df=None,
                          weightTag=None):

        # 计算给定指标单变量排序结果
        Ssort = self.Exec_SingleSort(avgTag,
                                     sortTag,
                                     df=df,
                                     timeTag=timeTag,
                                     groupsNum=groupsNum,
                                     weightTag=weightTag)[0] \
            .reset_index() \
            .pivot(timeTag,
                   'Group',
                   avgTag)

        # 做多值低的组，做空值高的组，构建因子值
        Ssort['HML'] = -Ssort['{}_G01'.format(
            sortTag)] + Ssort['{}_G05'.format(sortTag)]
        Ssort = Ssort.shift(1).dropna()
        SSortRes = pd.DataFrame(Ssort.mean(), columns=['AvgRet'])

        # 循环计算每一个投资组合的回归结果
        portName = Ssort.columns
        sortTable = pd.merge(Ssort, Factors,
                             left_index=True, right_index=True)
        for port in portName:
            regRes = self.Cpt_ResAndBeta(
                use_factors, port, df=sortTable, NWtest=True)
            alpha = float(regRes['beta'][regRes['beta'].shape[0]-1])
            SSortRes.loc[port, 'Alpha'] = alpha

            if port == 'HML':
                NWse = float(
                    regRes['NWse'][regRes['NWse'].shape[0]-1, regRes['NWse'].shape[1]-1])
                p_Alpha = scipy.stats.t.sf(
                    np.abs(alpha/np.sqrt(NWse)), Factors.shape[0] - 1) * 2
                SSortRes.loc['t', 'Alpha'] = alpha/np.sqrt(NWse)
                SSortRes.loc['p', 'Alpha'] = p_Alpha

        SSortRes.loc['t', 'AvgRet'] = self.Exec_NWTest(Ssort['HML'])['tstat']
        SSortRes.loc['p', 'AvgRet'] = self.Exec_NWTest(Ssort['HML'])['pval']

        return SSortRes.astype(float).round(4)

    def Rec_DoubleSortRes(self,
                          avgTag,
                          sortTag,
                          sortTag_key,
                          groupsNum1,
                          groupsNum2,
                          SortMethod,
                          Factors,
                          use_factors,
                          df=None,
                          timeTag='TradingDate',
                          weightTag=None):

        double_sort = self.Exec_DoubleSort(avgTag,
                                           sortTag,
                                           sortTag_key,
                                           groupsNum1,
                                           groupsNum2,
                                           df=df,
                                           timeTag=timeTag,
                                           SortMethod=SortMethod,
                                           weightTag=weightTag)[0]
        double_sort['retShit'] = double_sort.groupby('Group').retShit.shift(1)
        double_sort = double_sort.dropna()

        # 展示5*5分组的二维图表
        sort_table = self.Creat_SortTimePolymer(df=double_sort,
                                                avgTag=avgTag,
                                                sortTag1=sortTag,
                                                sortTag2=sortTag_key).T
        contName = sort_table.columns
        portName = sort_table.index
        sortRes = sort_table.copy()
        sortRes.loc['HML', :] = ''
        for cont in contName:
            df = double_sort[double_sort['Group_{}'.format(sortTag)] == cont]
            # 计算HML
            bigTag = sort_table.index[-1]
            smallTag = sort_table.index[0]
            bigGroup = df[df['Group_{}'.format(sortTag_key)] == bigTag]
            smallGroup = df[df['Group_{}'.format(sortTag_key)] == smallTag]
            HML = bigGroup[avgTag] - smallGroup[avgTag]
            HML.dropna(inplace=True)

            sortRes.loc['HML', cont] = HML.mean()
            sortRes.loc['t', cont] = self.Exec_NWTest(test_array=HML)['tstat']
            sortRes.loc['p', cont] = self.Exec_NWTest(test_array=HML)['pval']

        # 计算每一个key变量分组的alpha和平均收益率
        for port in portName:
            df = double_sort[double_sort['Group_{}'.format(
                sortTag_key)] == port]
            port_df = df.groupby(
                ['Group_{}'.format(sortTag_key), timeTag]).mean().reset_index()
            port_df = pd.merge(port_df, Factors, left_on=timeTag,
                               right_index=True, how='left')
            regRes = self.Cpt_ResAndBeta(
                use_factors, avgTag, df=port_df, NWtest=True)
            alpha = float(regRes['beta'][regRes['beta'].shape[0]-1])

            sortRes.loc[port, 'AvgRet'] = port_df[avgTag].mean()
            sortRes.loc[port, 'Alpha'] = alpha

        # 计算HML组平均收益率、以及该平均收益率的alpha和对应t值
        bigTag = sort_table.index[-1]
        bigGroup = double_sort[double_sort['Group_{}'.format(
            sortTag_key)] == bigTag]
        bigGroup = bigGroup.groupby(
            [timeTag, 'Group_{}'.format(sortTag_key)]).mean().reset_index()
        smallTag = sort_table.index[0]
        smallGroup = double_sort[double_sort['Group_{}'.format(
            sortTag_key)] == smallTag]
        smallGroup = smallGroup.groupby(
            [timeTag, 'Group_{}'.format(sortTag_key)]).mean().reset_index()

        HML_df = bigGroup[[avgTag]] - smallGroup[[avgTag]]
        HML.dropna(inplace=True)
        HML_df.index = bigGroup[timeTag]
        HML_df = pd.merge(HML_df, Factors, left_index=True,
                          right_index=True, how='left')
        regRes = self.Cpt_ResAndBeta(
            use_factors, avgTag, df=HML_df, NWtest=True)
        alpha = float(regRes['beta'][regRes['beta'].shape[0]-1])
        NWse = float(
            regRes['NWse'][regRes['NWse'].shape[0]-1, regRes['NWse'].shape[1]-1])
        p_Alpha = scipy.stats.t.sf(
            np.abs(alpha/np.sqrt(NWse)), Factors.shape[0] - 1) * 2

        sortRes.loc['HML', 'AvgRet'] = HML_df[avgTag].mean()
        sortRes.loc['HML', 'Alpha'] = alpha
        sortRes.loc['t', 'Alpha'] = alpha/np.sqrt(NWse)
        sortRes.loc['t', 'AvgRet'] = self.Exec_NWTest(
            test_array=HML_df[avgTag])['tstat']
        sortRes.loc['p', 'Alpha'] = p_Alpha
        sortRes.loc['p', 'AvgRet'] = self.Exec_NWTest(
            test_array=HML_df[avgTag])['pval']

        return sortRes.astype(float).round(4)

    # 绘制用于进行double—sort的两个变量最低组、最高组、hml组的时序图，并计算相关性

    def Show_SortGroupChart(self,
                            avgTag,
                            sortTag,
                            sortTag_key,
                            groupsNum1,
                            groupsNum2,
                            SortMethod,
                            dirPath,
                            weightTag=None):

        double_sort = self.Exec_DoubleSort(avgTag,
                                           sortTag,
                                           sortTag_key,
                                           groupsNum1,
                                           groupsNum2,
                                           SortMethod=SortMethod,
                                           weightTag=weightTag)[0]

        groupName1 = double_sort['Group_{}'.format(
            sortTag)].sort_values().unique()
        groupName2 = double_sort['Group_{}'.format(
            sortTag_key)].sort_values().unique()

        firstGroup1 = groupName1[0]
        firstGroup2 = groupName2[0]
        lastGroup1 = groupName1[-1]
        lastGroup2 = groupName2[-1]

        firstDf1 = double_sort[double_sort['Group_{}'.format(sortTag)] == firstGroup1] \
            .groupby(double_sort.index.name).mean().rename(columns={avgTag: firstGroup1})
        firstDf2 = double_sort[double_sort['Group_{}'.format(sortTag_key)] == firstGroup2] \
            .groupby(double_sort.index.name).mean().rename(columns={avgTag: firstGroup2})
        firstDf = pd.merge(firstDf1, firstDf2,
                           left_index=True, right_index=True)
        ax = firstDf.plot(figsize=(20, 10))
        fig = ax.get_figure()
        fig.savefig(
            r'{}\First Group Time-Series Return Chart.png'.format(dirPath))
        firstCorr = round(firstDf[firstGroup1].corr(firstDf[firstGroup2]), 4)

        # plt.plot(firstDf.index,firstDf.values)
        # # plt.xlabel()
        # plt.text('','','corr:{}'.format(firstCorr))
        # plt.legend([firstGroup1,firstGroup2])
        # plt.show()

        lastDf1 = double_sort[double_sort['Group_{}'.format(sortTag)] == lastGroup1] \
            .groupby(double_sort.index.name).mean().rename(columns={avgTag: lastGroup1})
        lastDf2 = double_sort[double_sort['Group_{}'.format(sortTag_key)] == lastGroup2] \
            .groupby(double_sort.index.name).mean().rename(columns={avgTag: lastGroup2})
        lastDf = pd.merge(lastDf1, lastDf2, left_index=True, right_index=True)
        ax = lastDf.plot(figsize=(20, 10))
        fig = ax.get_figure()
        fig.savefig(
            r'{}\Last Group Time-Series Return Chart.png'.format(dirPath))
        lastCorr = round(lastDf[lastGroup1].corr(lastDf[lastGroup2]), 4)

        hmlDf1 = pd.DataFrame(
            lastDf1[lastGroup1] - firstDf1[firstGroup1]).rename(columns={0: '{}_hml'.format(sortTag)})
        hmlDf2 = pd.DataFrame(lastDf2[lastGroup2] - firstDf2[firstGroup2]
                              ).rename(columns={0: '{}_hml'.format(sortTag_key)})
        hmlDf = pd.merge(hmlDf1, hmlDf2, left_index=True, right_index=True)
        ax = hmlDf.plot(figsize=(20, 10))
        fig = ax.get_figure()
        fig.savefig(
            r'{}\Last Minus First Group Time-Series Return Chart.png'.format(dirPath))
        hmlCorr = round(hmlDf['{}_hml'.format(sortTag)].corr(
            hmlDf['{}_hml'.format(sortTag_key)]), 4)

        return {'最低组相关性': firstCorr, '最高组相关性': lastCorr, '高减低组相关性': hmlCorr}

    def FamaMacBeth_summary_(self,
                            DF,
                            reg_lst,
                            reg_order,
                            reg_names=None,
                            params_format='{:.3f}',
                            tvalues_format='{:.2f}'):
        '''
        A function for Fama-MacBeth regression and results summary.

        Parameters
        ----------
        DF: DataFrame
            A panel date of which multi-index is stock and month (datetime64[ns]),
            containing all the dependent and independent variables.
        reg_lst: list
            A list containing multiple lists of dependent variable and independent
            variables, e.g., [['Y', 'X1', ...],..., ['Y', 'X1', ...,]].
        reg_order: list
            The order of independent variables in result table.
        reg_names: list
            The names for each regression.
        params_format: str
            The number of decimal places for parameters, e.g., '{:.3f}'.
        tvalues_format: str
            The number of decimal places for t-values, e.g., '{:.2f}'.
        '''

        # Create a DataFrame
        rows = sum([[var, f'{var}_t'] for var in ['const'] + reg_order], [])
        if reg_names is None:
            reg_names = [f'({i+1})' for i in range(len(reg_lst))]
        show = pd.DataFrame(index=rows, columns=reg_names)

        for reg, reg_name in zip(reg_lst, reg_names):
            df = DF.loc[:, reg].copy().dropna()
            T = len(df.index.get_level_values(df.index.names[1]).unique())
            lag = math.floor(4*(T/100)**(2/9))
            try:
                fmb = FamaMacBeth(df[reg[0]], sm.add_constant(df[reg[1:]]))
            except:
                fmb = FamaMacBeth(df[reg[0]], sm.add_constant(
                    df[reg[1:]]), check_rank=False)
            # Newey-West adjust
            fmb = fmb.fit(cov_type='kernel', bandwidth=lag)

            # params, tvalues(tstats) and pvalues
            params = fmb.params
            tvalues = fmb.tstats
            pvalues = fmb.pvalues

            # Obs.
            total_obs = fmb.nobs
            # mean_obs = fmb.time_info['mean']

            # average rsquared_adj
            dft = df.reset_index(level=df.index.names[0], drop=True).copy()
            rsquared_adj = []
            for month in dft.index.unique():
                dftm = dft.loc[month].copy()
                ols = sm.OLS(dftm[reg[0]], sm.add_constant(
                    dftm[reg[1:]])).fit()
                rsquared_adj.append(ols.rsquared_adj)
            ar2a = np.mean(rsquared_adj)

            # params and significance
            ps_lst = []
            for param, pvalue in zip(params, pvalues):
                param = params_format.format(param)
                if (pvalue <= 0.1) & (pvalue > 0.05):
                    param = param + '*'
                elif (pvalue <= 0.05) & (pvalue > 0.01):
                    param = param + '**'
                elif pvalue <= 0.01:
                    param = param + '***'
                ps_lst.append(param)

            # params and tvalues
            tvalues = [tvalues_format.format(t) for t in tvalues]
            t_lst = [f'{t}' for t in tvalues]
            pt_lst = [[i, j] for i, j in zip(ps_lst, t_lst)]

            # put them in place
            for var, pt in zip(['const'] + reg[1:], pt_lst):
                show.loc[var, reg_name] = pt[0]
                show.loc[f'{var}_t', reg_name] = pt[1]
            show.loc['No. Obs.', reg_name] = str(total_obs)
            show.loc['Adj. R2', reg_name] = '{:.2f}%'.format(ar2a * 100)

        rename_index = sum([[var, '']
                           for var in ['Intercept'] + reg_order], [])
        show.index = rename_index + ['No. Obs.', 'Adj. R2']

        return show.fillna('')
    
    
    def FamaMacBeth_summary(self,
                         DF,
                         reg_lst,
                         reg_order,
                         reg_names=None,
                         params_format='{:.3f}',
                         tvalues_format='{:.2f}'):
     '''
     A function for Fama-MacBeth regression and results summary.

     Parameters
     ----------
     DF: DataFrame
         A panel date of which multi-index is stock and month (datetime64[ns]),
         containing all the dependent and independent variables.
     reg_lst: list
         A list containing multiple lists of dependent variable and independent
         variables, e.g., [['Y', 'X1', ...],..., ['Y', 'X1', ...,]].
     reg_order: list
         The order of independent variables in result table.
     reg_names: list
         The names for each regression.
     params_format: str
         The number of decimal places for parameters, e.g., '{:.3f}'.
     tvalues_format: str
         The number of decimal places for t-values, e.g., '{:.2f}'.
     '''

     # Create a DataFrame
     count = sum(1 for sublist in reg_lst if len(sublist) == 2)
     rows =['simple','simple_t','','var'] + sum([['M{}'.format(num), 'M{}_t'.format(num)] for num in range(1,len(reg_lst)+1-count)], [])
     
     if reg_names is None:
          reg_names = ['M{}'.format(num) for num in range(1,len(reg_lst)+1-count)]
     show = pd.DataFrame(index=rows, columns=reg_order)
     show.loc['var'] = reg_order
     
     i = 0
     for reg in reg_lst:
         df = DF.loc[:, reg].copy().dropna()
         T = len(df.index.get_level_values(df.index.names[1]).unique())
         lag = math.floor(4*(T/100)**(2/9))
         try:
             fmb = FamaMacBeth(df[reg[0]], sm.add_constant(df[reg[1:]]))
         except:
             fmb = FamaMacBeth(df[reg[0]], sm.add_constant(df[reg[1:]]), check_rank=False)
         # Newey-West adjust
         fmb = fmb.fit(cov_type='kernel', bandwidth=lag)
         
         # params, tvalues(tstats) and pvalues
         params = fmb.params[1:]
         tvalues = fmb.tstats[1:]
         pvalues = fmb.pvalues[1:]

         # params and significance
         ps_lst = []
         for param, pvalue in zip(params, pvalues):
             param = params_format.format(param)
             if (pvalue <= 0.1) & (pvalue > 0.05):
                 param = param + '*'
             elif (pvalue <= 0.05) & (pvalue > 0.01):
                 param = param + '**'
             elif pvalue <= 0.01:
                 param = param + '***'
             ps_lst.append(param)

         # params and tvalues
         tvalues = [tvalues_format.format(t) for t in tvalues]
         t_lst = [f'{t}' for t in tvalues]
         pt_lst = [[i, j] for i, j in zip(ps_lst, t_lst)]
         i += 1

         if i-count >0:
         # put them in place
             for var, pt in zip(reg[1:], pt_lst):
                 show.loc['M{}'.format(i-count), var] = pt[0]
                 show.loc['M{}_t'.format(i-count), var] = pt[1]
         else:
             show.loc['simple',reg[1:][0]] = pt_lst[0][0]
             show.loc['simple_t',reg[1:][0]] = pt_lst[0][1]
             
     rename_index = ['','','',''] + sum([['M{}'.format(num), '']
                        for num in range(1,len(reg_lst)+1-count)], [])

     show.index = rename_index

     return show.fillna('')    
    



def TimeSeriesRegression_summary(self,
                                 DF,
                                 reg_lst,
                                 reg_order,
                                 reg_names=None,
                                 params_format='{:.3f}',
                                 tvalues_format='{:.2f}'):
    '''
    A function for Time Series regression and results summary.

    Parameters
    ----------
    DF: DataFrame
        A panel date of which multi-index is stock and month (datetime64[ns]),
        containing all the dependent and independent variables.
    reg_lst: list
        A list containing multiple lists of dependent variable and independent
        variables, e.g., [['Y', 'X1', ...],..., ['Y', 'X1', ...,]].
    reg_order: list
        The order of independent variables in result table.
    reg_names: list
        The names for each regression.
    params_format: str
        The number of decimal places for parameters, e.g., '{:.3f}'.
    tvalues_format: str
        The number of decimal places for t-values, e.g., '{:.2f}'.
    '''

    # Create a DataFrame
    count = sum(1 for sublist in reg_lst if len(sublist) == 2)
    rows =['simple','simple_t','','var'] + sum([['M{}'.format(num), 'M{}_t'.format(num)] for num in range(1,len(reg_lst)+1-count)], [])
    
    if reg_names is None:
         reg_names = ['M{}'.format(num) for num in range(1,len(reg_lst)+1-count)]
    show = pd.DataFrame(index=rows, columns=reg_order)
    show.loc['var'] = reg_order
    
    i = 0
    for reg in reg_lst:
        df = DF.loc[:, reg].copy().dropna()
        try:
            tsr = sm.OLS(df[reg[0]], sm.add_constant(df[reg[1:]])).fit()
        except:
            tsr = sm.OLS(df[reg[0]], sm.add_constant(df[reg[1:]])).fit()
        
        # params, tvalues(tstats) and pvalues
        params = tsr.params[1:]
        tvalues = tsr.tvalues[1:]
        pvalues = tsr.pvalues[1:]

        # params and significance
        ps_lst = []
        for param, pvalue in zip(params, pvalues):
            param = params_format.format(param)
            if (pvalue <= 0.1) & (pvalue > 0.05):
                param = param + '*'
            elif (pvalue <= 0.05) & (pvalue > 0.01):
                param = param + '**'
            elif pvalue <= 0.01:
                param = param + '***'
            ps_lst.append(param)

        # params and tvalues
        tvalues = [tvalues_format.format(t) for t in tvalues]
        t_lst = [f'{t}' for t in tvalues]
        pt_lst = [[i, j] for i, j in zip(ps_lst, t_lst)]
        i += 1

        if i-count >0:
        # put them in place
            for var, pt in zip(reg[1:], pt_lst):
                show.loc['M{}'.format(i-count), var] = pt[0]
                show.loc['M{}_t'.format(i-count), var] = pt[1]
        else:
            show.loc['simple',reg[1:][0]] = pt_lst[0][0]
            show.loc['simple_t',reg[1:][0]] = pt_lst[0][1]
            
    rename_index = ['','','',''] + sum([['M{}'.format(num), '']
                       for num in range(1,len(reg_lst)+1-count)], [])

    show.index = rename_index

    return show.fillna('')  

    


class Factors():
    """
    >>> main_dir_path=r'D:\个人项目\数据集合\学术研究_股票数据'
    """

    def __init__(self, main_dir_path):
        # super()
        self._stock_data = DataCleaner(main_dir_path)
        self._APT = AssetPricingTool()

    def FF5_Factors_Return(self, timeType='D'):

        # I. Identify Stock's belong to which factor's group
        # get stocks' month trade data
        Data_Source = self._stock_data._CSMAR
        stock_mon_trade_data = Data_Source.StockMonTradeData_df.copy()

        # get stocks' last month MV data of each year
        stock_mon_trade_data['Trdyr'] = stock_mon_trade_data['Trdmnt'].str.extract(
            '(\d{4})')
        mv_lastmon_data = stock_mon_trade_data.groupby(['Stkcd', 'Trdyr']).tail(1)[
            ['Stkcd', 'Trdyr', 'Msmvttl']].copy()

        # Do the data cleaning process
        # # 1. drop the data that don't meet the trade day number requirement
        # 2. select specific ExchangeTradBoard data
        # 3. drop stocks that have been tagged ST
        # 4. drop stocks whose listdt less than 12 months
        # 5. drop stocks whose have abnormal ret(mostly due to suspension)
        # stock_mon_trade_data = self._stock_data.Drop_Stock_InadeTradData_(df=stock_mon_trade_data)
        stock_mon_trade_data = self._stock_data.Select_Stock_ExchangeTradBoard(
            df=stock_mon_trade_data, board_list=[1, 4, 16, 32])
        stock_mon_trade_data = self._stock_data.Drop_Stock_STData(
            df=stock_mon_trade_data)
        stock_mon_trade_data = self._stock_data.Drop_Stock_InadeListdtData(
            df=stock_mon_trade_data, drop_Mon_num=12)
        stock_mon_trade_data = self._stock_data.Drop_Stock_SuspOverRet(
            df=stock_mon_trade_data)

        # get financial statment data
        balance_sheet_data = Data_Source.StockFSData_FS_Combas_df.copy()
        incomes_sheet_data = Data_Source.StockFSData_FS_Comins_df.copy()
        OprInc = incomes_sheet_data[['Stkcd', 'Accper', '营业利润']]
        OprInc = OprInc.rename(columns={'Accper': 'Trdyr', '营业利润': 'OprInc'})
        OprInc['Trdyr'] = OprInc['Trdyr'].str.extract('(\d{4})')
        OprInc = OprInc.drop_duplicates(subset=['Stkcd', 'Trdyr'], keep='last')

        # get balance sheet data
        using_data = balance_sheet_data[['Stkcd', 'Accper']].copy()
        using_data['TotAsset'] = balance_sheet_data['资产总计'].copy()
        using_data['NetAsset'] = balance_sheet_data['资产总计'] - \
            balance_sheet_data['负债合计']
        using_data['TotOnrEqt'] = balance_sheet_data['所有者权益合计'].copy()
        using_data['Trdyr'] = using_data['Accper'].str.extract('(\d{4})')
        # drop data whose NA is nagetive and TA is zero
        using_data = using_data[using_data.NetAsset > 0]
        using_data = using_data[using_data.TotAsset > 0]

        # add mv data and operational income
        using_data = pd.merge(using_data, mv_lastmon_data, left_on=['Stkcd', 'Trdyr'], right_on=['Stkcd', 'Trdyr'],
                              how='left')
        using_data = pd.merge(using_data, OprInc, left_on=['Stkcd', 'Trdyr'], right_on=['Stkcd', 'Trdyr'],
                              how='left')

        # calculate factor features
        using_data.sort_values(['Stkcd', 'Trdyr'], inplace=True)
        using_data = using_data[using_data.Accper.str[-5:] == '12-31']
        using_data['NetAsset_s1'] = using_data.groupby(
            'Stkcd').NetAsset.shift(1)
        using_data['TotAsset_s1'] = using_data.groupby(
            'Stkcd').TotAsset.shift(1)
        using_data['TotAsset_s2'] = using_data.groupby(
            'Stkcd').TotAsset.shift(2)
        using_data['OprInc_s1'] = using_data.groupby('Stkcd').OprInc.shift(1)
        using_data['TotOnrEqt_s1'] = using_data.groupby(
            'Stkcd').TotOnrEqt.shift(1)
        using_data['Msmvttl_s1'] = using_data.groupby('Stkcd').Msmvttl.shift(1)
        using_data['Inv'] = (using_data['TotAsset_s1'] -
                             using_data['TotAsset_s2'])/using_data['TotAsset_s2']
        using_data['OP'] = using_data['OprInc_s1']/using_data['NetAsset_s1']
        using_data['BM'] = using_data['NetAsset_s1']/using_data['Msmvttl_s1']/1000
        using_data = using_data.drop(['Msmvttl', 'Msmvttl_s1'], axis=1)

        # combine factors features datas and stocks basic datas
        stock_mon_data = pd.merge(stock_mon_trade_data, using_data, left_on=['Stkcd', 'Trdyr'],
                                  right_on=['Stkcd', 'Trdyr'], how='left')
        
        # stock_mon_data[['Stkcd','Trdmnt','BM']].to_csv(r'F:\数据集合\学术研究_股票数据\CSMAR\Combined_Data\BM.csv',index=False)
        
        stock_mon_data['MV'] = stock_mon_data.groupby(
            'Stkcd').Msmvttl.shift(1).copy()
        # stock_mon_data['MV_'] = stock_mon_data.Msmvttl.copy()
        stock_mon_data = stock_mon_data[[
            'Stkcd', 'Trdyr', 'Trdmnt', 'Mretwd', 'Inv', 'OP', 'BM', 'MV']].copy()
        # can set to param
        stock_mon_data = stock_mon_data[(stock_mon_data.Trdyr >= '1994') & (
            stock_mon_data.Trdyr <= '2023')]

        # define assign group function
        def group_assigned(stock_mon_data, timeTag, retTag, sortTag1, sortTag2, groupsNum1, groupsNum2):

            stock_mon_group_data = stock_mon_data[stock_mon_data[timeTag].str[-2:] == '04'].copy()
            stock_mon_group_data = stock_mon_group_data.dropna(
                subset=[retTag, sortTag1, sortTag2])
            group_res = self._APT.Exec_DoubleSort(avgTag=retTag, sortTag1=sortTag1,
                                                  sortTag2=sortTag2, groupsNum1=groupsNum1, groupsNum2=groupsNum2,
                                                  df=stock_mon_group_data, timeTag=timeTag)[1]

            new_group_name = 'Group_{}'.format(sortTag1+'@'+sortTag2)
            group_res = group_res[['Stkcd', timeTag, 'Group']].rename(
                columns={'Group': new_group_name})
            stock_mon_data = pd.merge(stock_mon_data, group_res, how='left')
            stock_mon_data = stock_mon_data.sort_values(['Stkcd', timeTag])
            # stock_mon_data[new_group_name] = stock_mon_data.groupby('Stkcd')[new_group_name].shift(1)
            stock_mon_data[new_group_name] = stock_mon_data.groupby(
                ['Stkcd'])[new_group_name].fillna(method='ffill')

            return stock_mon_data[['Stkcd', 'Trdmnt', new_group_name]]

        # assign group for all stocks all months
        for sortTag2 in ['BM', 'Inv', 'OP']:
            group_res = group_assigned(stock_mon_data, 'Trdmnt', 'Mretwd', 'MV',
                                       sortTag2, 2, [0, 0.3, 0.7, 1])
            stock_mon_data = pd.merge(stock_mon_data, group_res, left_on=['Stkcd', 'Trdmnt'],
                                      right_on=['Stkcd', 'Trdmnt'], how='left')

        # II. denpend on param calculating different frequency's factor return
        if timeType.upper() == 'M':

            rf = Data_Source.riskfree_rate_df.copy()
            rf.drop_duplicates(subset='Trdmnt', keep='last', inplace=True)
            rf['Nrrmtdt'] = rf['Nrrmtdt']/100
            stock_mon_data = pd.merge(stock_mon_data, rf[['Trdmnt', 'Nrrmtdt']],
                                      left_on='Trdmnt', right_on='Trdmnt', how='left')

            # Calculate mkt factor
            RF = stock_mon_data.groupby(['Trdmnt']).Nrrmtdt.mean()
            MKT = stock_mon_data.groupby(['Trdmnt']).apply(self._APT.Cpt_ValueWeight, avgTag='Mretwd', weightTag='MV') \
                .rename({0: 'MKT'}) - RF

            CMA_df = stock_mon_data.groupby(['Trdmnt', 'Group_MV@Inv']) \
                .apply(self._APT.Cpt_ValueWeight, avgTag='Mretwd', weightTag='MV') \
                .reset_index().rename(columns={0: 'Mretwd'})
            HML_df = stock_mon_data.groupby(['Trdmnt', 'Group_MV@BM']) \
                .apply(self._APT.Cpt_ValueWeight, avgTag='Mretwd', weightTag='MV') \
                .reset_index().rename(columns={0: 'Mretwd'})
            RMW_df = stock_mon_data.groupby(['Trdmnt', 'Group_MV@OP']) \
                .apply(self._APT.Cpt_ValueWeight, avgTag='Mretwd', weightTag='MV') \
                .reset_index().rename(columns={0: 'Mretwd'})

            HML = -1/2*(HML_df[HML_df['Group_MV@BM'] == 'MV_G01@BM_G01']['Mretwd'].values +
                        HML_df[HML_df['Group_MV@BM'] == 'MV_G02@BM_G01']['Mretwd'].values) + \
                1/2*(HML_df[HML_df['Group_MV@BM'] == 'MV_G01@BM_G03']['Mretwd'].values +
                     HML_df[HML_df['Group_MV@BM'] == 'MV_G02@BM_G03']['Mretwd'].values)

            CMA = 1/2*(CMA_df[CMA_df['Group_MV@Inv'] == 'MV_G01@Inv_G01']['Mretwd'].values +
                       CMA_df[CMA_df['Group_MV@Inv'] == 'MV_G02@Inv_G01']['Mretwd'].values) + \
                -1/2*(CMA_df[CMA_df['Group_MV@Inv'] == 'MV_G01@Inv_G03']['Mretwd'].values +
                      CMA_df[CMA_df['Group_MV@Inv'] == 'MV_G02@Inv_G03']['Mretwd'].values)

            RMW = -1/2*(RMW_df[RMW_df['Group_MV@OP'] == 'MV_G01@OP_G01']['Mretwd'].values +
                        RMW_df[RMW_df['Group_MV@OP'] == 'MV_G02@OP_G01']['Mretwd'].values) + \
                1/2*(RMW_df[RMW_df['Group_MV@OP'] == 'MV_G01@OP_G03']['Mretwd'].values +
                     RMW_df[RMW_df['Group_MV@OP'] == 'MV_G02@OP_G03']['Mretwd'].values)

            SMB_hml = 1/3*(HML_df[HML_df['Group_MV@BM'] == 'MV_G01@BM_G01']['Mretwd'].values +
                           HML_df[HML_df['Group_MV@BM'] == 'MV_G01@BM_G02']['Mretwd'].values +
                           HML_df[HML_df['Group_MV@BM'] == 'MV_G01@BM_G03']['Mretwd'].values) - \
                1/3*(HML_df[HML_df['Group_MV@BM'] == 'MV_G02@BM_G01']['Mretwd'].values +
                     HML_df[HML_df['Group_MV@BM'] == 'MV_G02@BM_G02']['Mretwd'].values +
                     HML_df[HML_df['Group_MV@BM'] == 'MV_G02@BM_G03']['Mretwd'].values)

            SMB_cma = 1/3*(CMA_df[CMA_df['Group_MV@Inv'] == 'MV_G01@Inv_G01']['Mretwd'].values +
                           CMA_df[CMA_df['Group_MV@Inv'] == 'MV_G01@Inv_G02']['Mretwd'].values +
                           CMA_df[CMA_df['Group_MV@Inv'] == 'MV_G01@Inv_G03']['Mretwd'].values) - \
                1/3*(CMA_df[CMA_df['Group_MV@Inv'] == 'MV_G02@Inv_G01']['Mretwd'].values +
                     CMA_df[CMA_df['Group_MV@Inv'] == 'MV_G02@Inv_G02']['Mretwd'].values +
                     CMA_df[CMA_df['Group_MV@Inv'] == 'MV_G02@Inv_G03']['Mretwd'].values)

            SMB_rmw = 1/3*(RMW_df[RMW_df['Group_MV@OP'] == 'MV_G01@OP_G01']['Mretwd'].values +
                           RMW_df[RMW_df['Group_MV@OP'] == 'MV_G01@OP_G02']['Mretwd'].values +
                           RMW_df[RMW_df['Group_MV@OP'] == 'MV_G01@OP_G03']['Mretwd'].values) - \
                1/3*(RMW_df[RMW_df['Group_MV@OP'] == 'MV_G02@OP_G01']['Mretwd'].values +
                     RMW_df[RMW_df['Group_MV@OP'] == 'MV_G02@OP_G02']['Mretwd'].values +
                     RMW_df[RMW_df['Group_MV@OP'] == 'MV_G02@OP_G03']['Mretwd'].values)

            SMB = (SMB_hml + SMB_cma + SMB_rmw)/3

            factors_df = pd.DataFrame(
                {'RF': RF, 'MKT': MKT, 'SMB': SMB, 'HML': HML, 'RMW': RMW, 'CMA': CMA}, index=HML_df.Trdmnt.unique())

        else:
            rf = Data_Source.riskfree_rate_df.copy()
            rf['Nrrdaydt'] = rf['Nrrdaydt']/100
            stock_day_trade_data = Data_Source.StockDayTradeData_df[['Stkcd', 'Trddt', 'Trdmnt', 'Dsmvtll',
                                                                     'Dretwd']].copy()
            stock_day_trade_data = pd.merge(stock_day_trade_data, rf[['Trddt', 'Nrrdaydt']],
                                            left_on='Trddt', right_on='Trddt', how='left')
            stock_day_data = pd.merge(stock_day_trade_data, stock_mon_data,
                                      left_on=['Stkcd', 'Trdmnt'], right_on=['Stkcd', 'Trdmnt'])

            RF = stock_day_data.groupby(['Trddt']).Nrrdaydt.mean()
            MKT = stock_day_data.dropna(subset='Dretwd').groupby(['Trddt']). \
                apply(self._APT.Cpt_ValueWeight, avgTag='Dretwd', weightTag='MV') \
                .rename({0: 'MKT'}) - RF

            CMA_df = stock_day_data.groupby(['Trddt', 'Group_MV@Inv']) \
                .apply(self._APT.Cpt_ValueWeight, avgTag='Dretwd', weightTag='MV') \
                .reset_index().rename(columns={0: 'Dretwd'})
            HML_df = stock_day_data.groupby(['Trddt', 'Group_MV@BM']) \
                .apply(self._APT.Cpt_ValueWeight, avgTag='Dretwd', weightTag='MV') \
                .reset_index().rename(columns={0: 'Dretwd'})
            RMW_df = stock_day_data.groupby(['Trddt', 'Group_MV@OP']) \
                .apply(self._APT.Cpt_ValueWeight, avgTag='Dretwd', weightTag='MV') \
                .reset_index().rename(columns={0: 'Dretwd'})

            HML = -1/2*(HML_df[HML_df['Group_MV@BM'] == 'MV_G01@BM_G01']['Dretwd'].values +
                        HML_df[HML_df['Group_MV@BM'] == 'MV_G02@BM_G01']['Dretwd'].values) + \
                1/2*(HML_df[HML_df['Group_MV@BM'] == 'MV_G01@BM_G03']['Dretwd'].values +
                     HML_df[HML_df['Group_MV@BM'] == 'MV_G02@BM_G03']['Dretwd'].values)

            CMA = 1/2*(CMA_df[CMA_df['Group_MV@Inv'] == 'MV_G01@Inv_G01']['Dretwd'].values +
                       CMA_df[CMA_df['Group_MV@Inv'] == 'MV_G02@Inv_G01']['Dretwd'].values) + \
                -1/2*(CMA_df[CMA_df['Group_MV@Inv'] == 'MV_G01@Inv_G03']['Dretwd'].values +
                      CMA_df[CMA_df['Group_MV@Inv'] == 'MV_G02@Inv_G03']['Dretwd'].values)

            RMW = -1/2*(RMW_df[RMW_df['Group_MV@OP'] == 'MV_G01@OP_G01']['Dretwd'].values +
                        RMW_df[RMW_df['Group_MV@OP'] == 'MV_G02@OP_G01']['Dretwd'].values) + \
                1/2*(RMW_df[RMW_df['Group_MV@OP'] == 'MV_G01@OP_G03']['Dretwd'].values +
                     RMW_df[RMW_df['Group_MV@OP'] == 'MV_G02@OP_G03']['Dretwd'].values)

            SMB_hml = 1/3*(HML_df[HML_df['Group_MV@BM'] == 'MV_G01@BM_G01']['Dretwd'].values +
                           HML_df[HML_df['Group_MV@BM'] == 'MV_G01@BM_G02']['Dretwd'].values +
                           HML_df[HML_df['Group_MV@BM'] == 'MV_G01@BM_G03']['Dretwd'].values) - \
                1/3*(HML_df[HML_df['Group_MV@BM'] == 'MV_G02@BM_G01']['Dretwd'].values +
                     HML_df[HML_df['Group_MV@BM'] == 'MV_G02@BM_G02']['Dretwd'].values +
                     HML_df[HML_df['Group_MV@BM'] == 'MV_G02@BM_G03']['Dretwd'].values)

            SMB_cma = 1/3*(CMA_df[CMA_df['Group_MV@Inv'] == 'MV_G01@Inv_G01']['Dretwd'].values +
                           CMA_df[CMA_df['Group_MV@Inv'] == 'MV_G01@Inv_G02']['Dretwd'].values +
                           CMA_df[CMA_df['Group_MV@Inv'] == 'MV_G01@Inv_G03']['Dretwd'].values) - \
                1/3*(CMA_df[CMA_df['Group_MV@Inv'] == 'MV_G02@Inv_G01']['Dretwd'].values +
                     CMA_df[CMA_df['Group_MV@Inv'] == 'MV_G02@Inv_G02']['Dretwd'].values +
                     CMA_df[CMA_df['Group_MV@Inv'] == 'MV_G02@Inv_G03']['Dretwd'].values)

            SMB_rmw = 1/3*(RMW_df[RMW_df['Group_MV@OP'] == 'MV_G01@OP_G01']['Dretwd'].values +
                           RMW_df[RMW_df['Group_MV@OP'] == 'MV_G01@OP_G02']['Dretwd'].values +
                           RMW_df[RMW_df['Group_MV@OP'] == 'MV_G01@OP_G03']['Dretwd'].values) - \
                1/3*(RMW_df[RMW_df['Group_MV@OP'] == 'MV_G02@OP_G01']['Dretwd'].values +
                     RMW_df[RMW_df['Group_MV@OP'] == 'MV_G02@OP_G02']['Dretwd'].values +
                     RMW_df[RMW_df['Group_MV@OP'] == 'MV_G02@OP_G03']['Dretwd'].values)

            SMB = (SMB_hml + SMB_cma + SMB_rmw)/3

            factors_df = pd.DataFrame(
                {'RF': RF, 'MKT': MKT, 'SMB': SMB, 'HML': HML, 'RMW': RMW, 'CMA': CMA}, index=HML_df.Trddt.unique())

        return factors_df

    def CH3_Factors_Return(self, timeType='D'):

        SAVIC_Month_df = self._stock_data.Recon_Data_SVIC()
        SAVIC_Month_df = SAVIC_Month_df.sort_values(['Stkcd', 'Trdmnt'])
        # org = SAVIC_Month_df.copy()
        # SAVIC_Month_df = org.copy()
        SAVIC_Month_df['retLag'] = SAVIC_Month_df.groupby(
            'Stkcd').Mretwd.shift(-1)
        SAVIC_Month_df['m_1'] = SAVIC_Month_df.groupby(
            ['Stkcd']).Msmvttl.shift(1)

        # VMG factor
        raw_data = self._stock_data._CSMAR.StockFSData_FS_Comins_df.copy()

        raw_data.drop(
            index=raw_data[raw_data.Typrep == 'B'].index, inplace=True)
        raw_data['type'] = raw_data.Accper.apply(lambda x: x[5:7] + x[8:10])

        raw_data.drop(index=raw_data[
            (raw_data.type != '1231') & (raw_data.type != '0930') & (raw_data.type != '0630') & (raw_data.type != '0331')].index, inplace=True)
        raw_data.Accper = raw_data.Accper.apply(lambda x: x[0:4] + x[5:7])

        # B002000000: 净利润
        # B001400000：营业外收入
        # B001500000：营业外支出
        Net_profit = raw_data[['Stkcd', 'Accper', '净利润', '加：营业外收入', '减：营业外支出']]
        Net_profit.index = range(len(raw_data))

        lag_d = []
        for i in range(Net_profit.shape[0]):
            if Net_profit.loc[i]['Accper'][4:] == '03':
                lag_d.append(Net_profit.loc[i]['Accper'][0:4] + '04')
            if Net_profit.loc[i]['Accper'][4:] == '06':
                lag_d.append(Net_profit.loc[i]['Accper'][0:4] + '07')
            if Net_profit.loc[i]['Accper'][4:] == '09':
                lag_d.append(Net_profit.loc[i]['Accper'][0:4] + '10')
            if Net_profit.loc[i]['Accper'][4:] == '12':
                newyear = str(int(Net_profit.loc[i]['Accper'][0:4]) + 1)
                lag_d.append(newyear + '01')

        # Net_profit['Accper'] = pd.DataFrame(lag_d)
        Net_profit.drop_duplicates(
            subset=['Stkcd', 'Accper'], keep='last', inplace=True)
        Net_profit.rename(columns={'Accper': 'Trdmnt'}, inplace=True)
        Net_profit['Net_Profit_After_Ded_Nr_Lpnet'] = Net_profit['净利润'] - \
            (Net_profit['加：营业外收入'] - Net_profit['减：营业外支出'])
        Net_profit['Trdmnt'] = Net_profit.Trdmnt.apply(
            lambda x: str(x)[:4]+'-'+str(x)[4:])

        # merge data
        # SAVIC_Month_df['flushare'] = SAVIC_Month_df.Msmvosd*1000/SAVIC_Month_df.Mclsprc
        # SAVIC_Month_df['turnover'] = SAVIC_Month_df.Mnshrtrd/SAVIC_Month_df.flushare
        # SAVIC_Month_df['TradVolume'] = SAVIC_Month_df.groupby('Stkcd').turnover.rolling(12).mean().values
        # SAVIC_Month_df['ABT'] = SAVIC_Month_df.turnover/SAVIC_Month_df.TradVolume

        SAVIC_Month_df = pd.merge(SAVIC_Month_df, Net_profit[[
                                  'Stkcd', 'Trdmnt', 'Net_Profit_After_Ded_Nr_Lpnet']], on=['Stkcd', 'Trdmnt'], how='left')
        SAVIC_Month_df.Net_Profit_After_Ded_Nr_Lpnet.fillna(
            method='ffill', inplace=True)

        # calcuate ep
        SAVIC_Month_df['ep'] = SAVIC_Month_df.Net_Profit_After_Ded_Nr_Lpnet / \
            SAVIC_Month_df.m_1
        SAVIC_Month_df = SAVIC_Month_df.sort_values(['Stkcd', 'Trdmnt'])

        SAVIC_Month_df = SAVIC_Month_df.dropna()
        SAVIC_Month_df = SAVIC_Month_df[[
            'Stkcd', 'Trdmnt', 'Msmvttl', 'Mretwd', 'ep', 'retLag']]
        SAVIC_Month_df = SAVIC_Month_df.rename(
            columns={'Msmvttl': 'MarketValue'})

        SAVIC_Month_df = SAVIC_Month_df[SAVIC_Month_df.Trdmnt >= '1998-11']

        market_ttlvalue = SAVIC_Month_df.groupby('Trdmnt').MarketValue.sum(
        ).reset_index().rename(columns={'MarketValue': 'market_ttlvalue'})
        SAVIC_Month_df = pd.merge(SAVIC_Month_df, market_ttlvalue, on=[
                                  'Trdmnt'], how='left')
        SAVIC_Month_df['vwret'] = SAVIC_Month_df.retLag * \
            SAVIC_Month_df.MarketValue / SAVIC_Month_df.market_ttlvalue

        MKT = SAVIC_Month_df.groupby('Trdmnt').vwret.sum().reset_index()

        rf = self._stock_data._CSMAR.riskfree_rate_df.copy()
        rf.Nrrmtdt = rf.Nrrmtdt / 100
        rf.drop_duplicates('Trdmnt', keep='last', inplace=True)

        MKT = pd.merge(MKT, rf, on='Trdmnt', how='left')
        MKT['MKT'] = MKT.vwret - MKT.Nrrmtdt
        MKT = MKT.set_index('Trdmnt')['MKT']
        MKT = MKT.rename('mkt')

        Dsort1, df_ep = self._APT.Exec_DoubleSort(avgTag='retLag',
                                                  sortTag1='ep',
                                                  sortTag2='MarketValue',
                                                  groupsNum1=[0, 0.3, 0.7, 1],
                                                  groupsNum2=2,
                                                  timeTag='Trdmnt',
                                                  weightTag='MarketValue',
                                                  SortMethod='Independent',
                                                  df=SAVIC_Month_df.copy())

        df_ep = df_ep[['Stkcd', 'Trdmnt', 'Group']]
        df_ep = df_ep.rename(columns={'Group': 'ep@MarketValue'})
        SAVIC_Month_df = pd.merge(SAVIC_Month_df, df_ep)

        VMG = -1/2*(Dsort1[Dsort1['Group'] == 'ep_G01@MarketValue_G02']['retLag'] +
                    Dsort1[Dsort1['Group'] == 'ep_G01@MarketValue_G01']['retLag']) + \
            1/2*(Dsort1[Dsort1['Group'] == 'ep_G03@MarketValue_G02']['retLag'] +
                 Dsort1[Dsort1['Group'] == 'ep_G03@MarketValue_G01']['retLag'])
        VMG = VMG.rename('vmg')

        SMB = 1/3*(Dsort1[Dsort1['Group'] == 'ep_G01@MarketValue_G01']['retLag'] +
                   Dsort1[Dsort1['Group'] == 'ep_G02@MarketValue_G01']['retLag'] +
                   Dsort1[Dsort1['Group'] == 'ep_G03@MarketValue_G01']['retLag']) - \
            1/3*(Dsort1[Dsort1['Group'] == 'ep_G01@MarketValue_G02']['retLag'] +
                 Dsort1[Dsort1['Group'] == 'ep_G02@MarketValue_G02']['retLag'] +
                 Dsort1[Dsort1['Group'] == 'ep_G03@MarketValue_G02']['retLag'])
        SMB = SMB.rename('smb')

        RF = self._stock_data._CSMAR.riskfree_rate_df.copy()
        RF = RF[['Trdmnt', 'Nrrmtdt']]
        RF.Nrrmtdt = RF.Nrrmtdt/100
        RF = RF.set_index('Trdmnt').rename(columns={'Nrrmtdt': 'rf'})

        CH3 = pd.concat([MKT, SMB, VMG], axis=1).shift(1).dropna()
        CH3 = pd.merge(RF, CH3, right_index=True, left_index=True)

        # SVIC_df = pd.read_csv(r'F:\数据集合\学术研究_股票数据\多因子模型收益率\Stambaugh SVIC\CH_3_update_20211231.csv',header=8)
        # SVIC_df['Trdmnt'] = SVIC_df.mnthdt.apply(lambda x:str(x)[:4]+'-' + str(x)[4:6] )

        # a = pd.merge(SVIC_df, CH3,left_on='Trdmnt',right_index=True)
        # a.corr()

        if timeType.upper() == 'M':
            return CH3

        else:

            rf = self._stock_data._CSMAR.riskfree_rate_df.copy()
            rf['Nrrdaydt'] = rf['Nrrdaydt']/100
            stock_day_trade_data = self._stock_data._CSMAR.StockDayTradeData_df[[
                'Stkcd', 'Trddt', 'Trdmnt', 'Dsmvtll', 'Dretwd']].copy()

            stock_day_trade_data = pd.merge(stock_day_trade_data, rf[['Trddt', 'Nrrdaydt']],
                                            left_on='Trddt', right_on='Trddt', how='left')

            stock_day_data = pd.merge(stock_day_trade_data, SAVIC_Month_df,
                                      left_on=['Stkcd', 'Trdmnt'], right_on=['Stkcd', 'Trdmnt'])

            RF = stock_day_data.groupby(['Trddt']).Nrrdaydt.mean()
            MKT = stock_day_data.dropna(subset='Dretwd').groupby(['Trddt']). \
                apply(self._APT.Cpt_ValueWeight, avgTag='Dretwd', weightTag='Dsmvtll') \
                .rename({0: 'MKT'}) - RF
            MKT = MKT.rename('mkt')

            VMG_df = stock_day_data.groupby(['Trddt', 'ep@MarketValue']) \
                .apply(self._APT.Cpt_ValueWeight, avgTag='Dretwd', weightTag='Dsmvtll') \
                .reset_index().rename(columns={0: 'Dretwd'})

            VMG = -1/2*(VMG_df[VMG_df['ep@MarketValue'] == 'ep_G01@MarketValue_G02']['Dretwd'].values +
                        VMG_df[VMG_df['ep@MarketValue'] == 'ep_G01@MarketValue_G01']['Dretwd'].values) + \
                1/2*(VMG_df[VMG_df['ep@MarketValue'] == 'ep_G03@MarketValue_G02']['Dretwd'].values +
                     VMG_df[VMG_df['ep@MarketValue'] == 'ep_G03@MarketValue_G01']['Dretwd'].values)
            VMG = pd.Series(VMG, index=VMG_df.Trddt.unique())
            VMG = VMG.rename('vmg')

            SMB = 1/3*(VMG_df[VMG_df['ep@MarketValue'] == 'ep_G01@MarketValue_G01']['Dretwd'].values +
                       VMG_df[VMG_df['ep@MarketValue'] == 'ep_G02@MarketValue_G01']['Dretwd'].values +
                       VMG_df[VMG_df['ep@MarketValue'] == 'ep_G03@MarketValue_G01']['Dretwd'].values) - \
                1/3*(VMG_df[VMG_df['ep@MarketValue'] == 'ep_G01@MarketValue_G02']['Dretwd'].values +
                     VMG_df[VMG_df['ep@MarketValue'] == 'ep_G02@MarketValue_G02']['Dretwd'].values +
                     VMG_df[VMG_df['ep@MarketValue'] == 'ep_G03@MarketValue_G02']['Dretwd'].values)

            SMB = pd.Series(SMB, index=VMG_df.Trddt.unique())
            SMB = SMB.rename('smb')

            # SVIC_df = pd.read_csv(r'F:\数据集合\学术研究_股票数据\多因子模型收益率\Stambaugh SVIC\CH_3_fac_daily_update_20211231.csv',header=8)
            # SVIC_df['Trddt'] = SVIC_df.date.apply(lambda x:str(x)[:4]+'-' + str(x)[4:6] +'-'+ str(x)[6:])
            # # SVIC_df[['rf_dly','mktrf','VMG','SMB']] = SVIC_df[['rf_dly','mktrf','VMG','SMB','PMO']]/100

            # a = pd.merge(SVIC_df, CH3,left_on='Trddt',right_index=True)
            # C = a.corr()

        CH3 = pd.concat([MKT, SMB, VMG], axis=1)

        RF = self._stock_data._CSMAR.riskfree_rate_df.copy()
        RF = RF[['Trddt', 'Nrrdaydt']]
        RF.Nrrdaydt = RF.Nrrdaydt/100
        RF = RF.set_index('Trddt').rename(columns={'Nrrdaydt': 'rf'})

        CH3 = pd.merge(RF, CH3, right_index=True, left_index=True)

        return CH3

    def CH4_Factors_Return(self, timeType='D'):

        turnover_df = self._stock_data._CSMAR.StockDayTradeData_df[[
            'Stkcd', 'Trddt', 'Clsprc', 'Dnshrtrd', 'Dsmvtll', 'Dretwd', 'Trdmnt']]
        turnover_df = turnover_df.sort_values(['Stkcd', 'Trddt'])
        turnover_df['flushare'] = turnover_df.Dsmvtll*1000/turnover_df.Clsprc
        turnover_df['turnover'] = turnover_df.Dnshrtrd/turnover_df.flushare
        turnover_df['TradVolume_250'] = turnover_df.groupby(
            'Stkcd').turnover.rolling(250).mean().values
        turnover_df['TradVolume_20'] = turnover_df.groupby(
            'Stkcd').turnover.rolling(20).mean().values
        turnover_df['ABT'] = turnover_df.TradVolume_20 / \
            turnover_df.TradVolume_250

        turnover_df = turnover_df.groupby(
            ['Stkcd', 'Trdmnt']).ABT.mean().reset_index()

        # turnover_df = turnover_df.drop_duplicates(subset=['Stkcd','Trdmnt'], keep='last')
        # turnover_df = turnover_df[['Stkcd', 'Trdmnt','ABT']]

        SAVIC_Month_df = self._stock_data.Recon_Data_SVIC()
        SAVIC_Month_df = SAVIC_Month_df.sort_values(['Stkcd', 'Trdmnt'])
        # org = SAVIC_Month_df.copy()
        # SAVIC_Month_df = org.copy()
        SAVIC_Month_df['retLag'] = SAVIC_Month_df.groupby(
            'Stkcd').Mretwd.shift(-1)
        SAVIC_Month_df['m_1'] = SAVIC_Month_df.groupby(
            ['Stkcd']).Msmvttl.shift(1)
        SAVIC_Month_df = pd.merge(SAVIC_Month_df, turnover_df)

        # VMG factor
        raw_data = self._stock_data._CSMAR.StockFSData_FS_Comins_df.copy()

        raw_data.drop(
            index=raw_data[raw_data.Typrep == 'B'].index, inplace=True)
        raw_data['type'] = raw_data.Accper.apply(lambda x: x[5:7] + x[8:10])

        raw_data.drop(index=raw_data[
            (raw_data.type != '1231') & (raw_data.type != '0930') & (raw_data.type != '0630') & (raw_data.type != '0331')].index, inplace=True)
        raw_data.Accper = raw_data.Accper.apply(lambda x: x[0:4] + x[5:7])

        # B002000000: 净利润
        # B001400000：营业外收入
        # B001500000：营业外支出
        Net_profit = raw_data[['Stkcd', 'Accper', '净利润', '加：营业外收入', '减：营业外支出']]
        Net_profit.index = range(len(raw_data))

        lag_d = []
        for i in range(Net_profit.shape[0]):
            if Net_profit.loc[i]['Accper'][4:] == '03':
                lag_d.append(Net_profit.loc[i]['Accper'][0:4] + '04')
            if Net_profit.loc[i]['Accper'][4:] == '06':
                lag_d.append(Net_profit.loc[i]['Accper'][0:4] + '07')
            if Net_profit.loc[i]['Accper'][4:] == '09':
                lag_d.append(Net_profit.loc[i]['Accper'][0:4] + '10')
            if Net_profit.loc[i]['Accper'][4:] == '12':
                newyear = str(int(Net_profit.loc[i]['Accper'][0:4]) + 1)
                lag_d.append(newyear + '01')

        # Net_profit['Accper'] = pd.DataFrame(lag_d)
        Net_profit.drop_duplicates(
            subset=['Stkcd', 'Accper'], keep='last', inplace=True)
        Net_profit.rename(columns={'Accper': 'Trdmnt'}, inplace=True)
        Net_profit['Net_Profit_After_Ded_Nr_Lpnet'] = Net_profit['净利润'] - \
            (Net_profit['加：营业外收入'] - Net_profit['减：营业外支出'])
        Net_profit['Trdmnt'] = Net_profit.Trdmnt.apply(
            lambda x: str(x)[:4]+'-'+str(x)[4:])

        # merge data
        # SAVIC_Month_df['flushare'] = SAVIC_Month_df.Msmvosd*1000/SAVIC_Month_df.Mclsprc
        # SAVIC_Month_df['turnover'] = SAVIC_Month_df.Mnshrtrd/SAVIC_Month_df.flushare
        # SAVIC_Month_df['TradVolume'] = SAVIC_Month_df.groupby('Stkcd').turnover.rolling(12).mean().values
        # SAVIC_Month_df['ABT'] = SAVIC_Month_df.turnover/SAVIC_Month_df.TradVolume

        SAVIC_Month_df = pd.merge(SAVIC_Month_df, Net_profit[[
                                  'Stkcd', 'Trdmnt', 'Net_Profit_After_Ded_Nr_Lpnet']], on=['Stkcd', 'Trdmnt'], how='left')
        SAVIC_Month_df.Net_Profit_After_Ded_Nr_Lpnet.fillna(
            method='ffill', inplace=True)

        # calcuate ep
        SAVIC_Month_df['ep'] = SAVIC_Month_df.Net_Profit_After_Ded_Nr_Lpnet / \
            SAVIC_Month_df.m_1
        SAVIC_Month_df = SAVIC_Month_df.sort_values(['Stkcd', 'Trdmnt'])

        SAVIC_Month_df = SAVIC_Month_df.dropna()
        SAVIC_Month_df = SAVIC_Month_df[[
            'Stkcd', 'Trdmnt', 'Msmvttl', 'Mretwd', 'ep', 'ABT', 'retLag']]
        SAVIC_Month_df = SAVIC_Month_df.rename(
            columns={'Msmvttl': 'MarketValue'})

        SAVIC_Month_df = SAVIC_Month_df[SAVIC_Month_df.Trdmnt >= '1998-11']

        market_ttlvalue = SAVIC_Month_df.groupby('Trdmnt').MarketValue.sum(
        ).reset_index().rename(columns={'MarketValue': 'market_ttlvalue'})
        SAVIC_Month_df = pd.merge(SAVIC_Month_df, market_ttlvalue, on=[
                                  'Trdmnt'], how='left')
        SAVIC_Month_df['vwret'] = SAVIC_Month_df.retLag * \
            SAVIC_Month_df.MarketValue / SAVIC_Month_df.market_ttlvalue

        MKT = SAVIC_Month_df.groupby('Trdmnt').vwret.sum().reset_index()

        rf = self._stock_data._CSMAR.riskfree_rate_df.copy()
        rf.Nrrmtdt = rf.Nrrmtdt / 100
        rf.drop_duplicates('Trdmnt', keep='last', inplace=True)

        MKT = pd.merge(MKT, rf, on='Trdmnt', how='left')
        MKT['MKT'] = MKT.vwret - MKT.Nrrmtdt
        MKT = MKT.set_index('Trdmnt')['MKT']
        MKT = MKT.rename('mkt')

        Dsort1, df_ep = self._APT.Exec_DoubleSort(avgTag='retLag',
                                                  sortTag1='ep',
                                                  sortTag2='MarketValue',
                                                  groupsNum1=[0, 0.3, 0.7, 1],
                                                  groupsNum2=2,
                                                  timeTag='Trdmnt',
                                                  weightTag='MarketValue',
                                                  SortMethod='Independent',
                                                  df=SAVIC_Month_df.copy())

        df_ep = df_ep[['Stkcd', 'Trdmnt', 'Group']]
        df_ep = df_ep.rename(columns={'Group': 'ep@MarketValue'})
        SAVIC_Month_df = pd.merge(SAVIC_Month_df, df_ep)

        VMG = -1/2*(Dsort1[Dsort1['Group'] == 'ep_G01@MarketValue_G02']['retLag'] +
                    Dsort1[Dsort1['Group'] == 'ep_G01@MarketValue_G01']['retLag']) + \
            1/2*(Dsort1[Dsort1['Group'] == 'ep_G03@MarketValue_G02']['retLag'] +
                 Dsort1[Dsort1['Group'] == 'ep_G03@MarketValue_G01']['retLag'])
        VMG = VMG.rename('vmg')

        SMB1 = 1/3*(Dsort1[Dsort1['Group'] == 'ep_G01@MarketValue_G01']['retLag'] +
                    Dsort1[Dsort1['Group'] == 'ep_G02@MarketValue_G01']['retLag'] +
                    Dsort1[Dsort1['Group'] == 'ep_G03@MarketValue_G01']['retLag']) - \
            1/3*(Dsort1[Dsort1['Group'] == 'ep_G01@MarketValue_G02']['retLag'] +
                 Dsort1[Dsort1['Group'] == 'ep_G02@MarketValue_G02']['retLag'] +
                 Dsort1[Dsort1['Group'] == 'ep_G03@MarketValue_G02']['retLag'])

        Dsort2, df_abt = self._APT.Exec_DoubleSort(avgTag='retLag',
                                                   sortTag1='ABT',
                                                   sortTag2='MarketValue',
                                                   groupsNum1=[0, 0.3, 0.7, 1],
                                                   groupsNum2=2,
                                                   timeTag='Trdmnt',
                                                   weightTag='MarketValue',
                                                   SortMethod='Independent',
                                                   df=SAVIC_Month_df.copy())

        df_abt = df_abt[['Stkcd', 'Trdmnt', 'Group']]
        df_abt = df_abt.rename(columns={'Group': 'ABT@MarketValue'})
        SAVIC_Month_df = pd.merge(SAVIC_Month_df, df_abt)

        PMO = 1/2*(Dsort2[Dsort2['Group'] == 'ABT_G01@MarketValue_G02']['retLag'] +
                   Dsort2[Dsort2['Group'] == 'ABT_G01@MarketValue_G01']['retLag']) - \
            1/2*(Dsort2[Dsort2['Group'] == 'ABT_G03@MarketValue_G02']['retLag'] +
                 Dsort2[Dsort2['Group'] == 'ABT_G03@MarketValue_G01']['retLag'])
        PMO = PMO.rename('pmo')

        SMB2 = 1/3*(Dsort2[Dsort2['Group'] == 'ABT_G01@MarketValue_G01']['retLag'] +
                    Dsort2[Dsort2['Group'] == 'ABT_G02@MarketValue_G01']['retLag'] +
                    Dsort2[Dsort2['Group'] == 'ABT_G03@MarketValue_G01']['retLag']) - \
            1/3*(Dsort2[Dsort2['Group'] == 'ABT_G01@MarketValue_G02']['retLag'] +
                 Dsort2[Dsort2['Group'] == 'ABT_G02@MarketValue_G02']['retLag'] +
                 Dsort2[Dsort2['Group'] == 'ABT_G03@MarketValue_G02']['retLag'])
        SMB = 1/2*(SMB1+SMB2)
        SMB = SMB.rename('smb')

        CH4 = pd.concat([MKT, SMB, VMG, PMO], axis=1).shift(1).dropna()

        # SVIC_df = pd.read_csv(r'F:\数据集合\学术研究_股票数据\多因子模型收益率\Stambaugh SVIC\CH_4_fac_update_20211231.csv',header=9)
        # SVIC_df['Trdmnt'] = SVIC_df.mnthdt.apply(lambda x:str(x)[:4]+'-' + str(x)[4:6] )

        # a = pd.merge(SVIC_df, CH4,left_on='Trdmnt',right_index=True)
        # a.corr()

        if timeType.upper() == 'M':
            return CH4

        else:

            rf = self._stock_data._CSMAR.riskfree_rate_df.copy()
            rf['Nrrdaydt'] = rf['Nrrdaydt']/100
            stock_day_trade_data = self._stock_data._CSMAR.StockDayTradeData_df[[
                'Stkcd', 'Trddt', 'Trdmnt', 'Dsmvtll', 'Dretwd']].copy()

            stock_day_trade_data = pd.merge(stock_day_trade_data, rf[['Trddt', 'Nrrdaydt']],
                                            left_on='Trddt', right_on='Trddt', how='left')

            stock_day_data = pd.merge(stock_day_trade_data, SAVIC_Month_df,
                                      left_on=['Stkcd', 'Trdmnt'], right_on=['Stkcd', 'Trdmnt'])

            RF = stock_day_data.groupby(['Trddt']).Nrrdaydt.mean()
            MKT = stock_day_data.dropna(subset='Dretwd').groupby(['Trddt']). \
                apply(self._APT.Cpt_ValueWeight, avgTag='Dretwd', weightTag='Dsmvtll') \
                .rename({0: 'MKT'}) - RF
            MKT = MKT.rename('mkt')

            VMG_df = stock_day_data.groupby(['Trddt', 'ep@MarketValue']) \
                .apply(self._APT.Cpt_ValueWeight, avgTag='Dretwd', weightTag='Dsmvtll') \
                .reset_index().rename(columns={0: 'Dretwd'})

            ABT_df = stock_day_data.groupby(['Trddt', 'ABT@MarketValue']) \
                .apply(self._APT.Cpt_ValueWeight, avgTag='Dretwd', weightTag='Dsmvtll') \
                .reset_index().rename(columns={0: 'Dretwd'})

            VMG = -1/2*(VMG_df[VMG_df['ep@MarketValue'] == 'ep_G01@MarketValue_G02']['Dretwd'].values +
                        VMG_df[VMG_df['ep@MarketValue'] == 'ep_G01@MarketValue_G01']['Dretwd'].values) + \
                1/2*(VMG_df[VMG_df['ep@MarketValue'] == 'ep_G03@MarketValue_G02']['Dretwd'].values +
                     VMG_df[VMG_df['ep@MarketValue'] == 'ep_G03@MarketValue_G01']['Dretwd'].values)
            VMG = pd.Series(VMG, index=VMG_df.Trddt.unique())
            VMG = VMG.rename('vmg')

            PMO = 1/2*(ABT_df[ABT_df['ABT@MarketValue'] == 'ABT_G01@MarketValue_G02']['Dretwd'].values +
                       ABT_df[ABT_df['ABT@MarketValue'] == 'ABT_G01@MarketValue_G01']['Dretwd'].values) - \
                1/2*(ABT_df[ABT_df['ABT@MarketValue'] == 'ABT_G03@MarketValue_G02']['Dretwd'].values +
                     ABT_df[ABT_df['ABT@MarketValue'] == 'ABT_G03@MarketValue_G01']['Dretwd'].values)
            PMO = pd.Series(PMO, index=ABT_df.Trddt.unique())
            PMO = PMO.rename('pmo')

            SMB_vmg = 1/3*(VMG_df[VMG_df['ep@MarketValue'] == 'ep_G01@MarketValue_G01']['Dretwd'].values +
                           VMG_df[VMG_df['ep@MarketValue'] == 'ep_G02@MarketValue_G01']['Dretwd'].values +
                           VMG_df[VMG_df['ep@MarketValue'] == 'ep_G03@MarketValue_G01']['Dretwd'].values) - \
                1/3*(VMG_df[VMG_df['ep@MarketValue'] == 'ep_G01@MarketValue_G02']['Dretwd'].values +
                     VMG_df[VMG_df['ep@MarketValue'] == 'ep_G02@MarketValue_G02']['Dretwd'].values +
                     VMG_df[VMG_df['ep@MarketValue'] == 'ep_G03@MarketValue_G02']['Dretwd'].values)

            SMB_pmo = 1/3*(ABT_df[ABT_df['ABT@MarketValue'] == 'ABT_G01@MarketValue_G01']['Dretwd'].values +
                           ABT_df[ABT_df['ABT@MarketValue'] == 'ABT_G02@MarketValue_G01']['Dretwd'].values +
                           ABT_df[ABT_df['ABT@MarketValue'] == 'ABT_G03@MarketValue_G01']['Dretwd'].values) - \
                1/3*(ABT_df[ABT_df['ABT@MarketValue'] == 'ABT_G01@MarketValue_G02']['Dretwd'].values +
                     ABT_df[ABT_df['ABT@MarketValue'] == 'ABT_G02@MarketValue_G02']['Dretwd'].values +
                     ABT_df[ABT_df['ABT@MarketValue'] == 'ABT_G03@MarketValue_G02']['Dretwd'].values)

            SMB = (SMB_vmg + SMB_pmo)/2
            SMB = pd.Series(SMB, index=ABT_df.Trddt.unique())
            SMB = SMB.rename('smb')

            # SVIC_df = pd.read_csv(r'F:\数据集合\学术研究_股票数据\多因子模型收益率\Stambaugh SVIC\CH_4_fac_daily_update_20211231.csv',header=9)
            # SVIC_df['Trddt'] = SVIC_df.date.apply(lambda x:str(x)[:4]+'-' + str(x)[4:6] +'-'+ str(x)[6:])
            # SVIC_df[['rf_dly','mktrf','VMG','SMB','PMO']] = SVIC_df[['rf_dly','mktrf','VMG','SMB','PMO']]/100

            # VMG_df.Dretwd = VMG_df.Dretwd.shift(1)
            # a = pd.merge(SVIC_df, CH4,left_on='Trddt',right_index=True)
            # C = a.corr()

        CH4 = pd.concat([MKT, SMB, VMG, PMO], axis=1)
        return CH4

    def Crt_HFweight_of_StockRet(self, stock_hf_data_path):
        """
        stock_hf_data_path = r'D:\个人项目\数据集合\学术研究_股票数据\5minHFprice_csv\000001.csv'

        Returns
        -------
        None.

        """
        stock_hf_data = pd.read_csv(stock_hf_data_path, index_col=0)  \
                          .rename(columns={'stock_code': 'Stkcd', 'day': 'Trddt'})
        Stkcd = stock_hf_data.Stkcd.unique()[0]
        Data_Source = self._stock_data._CSMAR
        stock_day_trade_data = Data_Source.StockDayTradeData_df.copy()

        # get stocks' last month MV data of each year
        # stock_mon_trade_data['Trdyr'] = stock_mon_trade_data['Trdmnt'].str.extract('(\d{4})')
        mv_day_data = stock_day_trade_data.groupby(['Stkcd', 'Trddt']) \
            .tail(1)[['Stkcd', 'Trddt', 'Dsmvtll', 'PreClosePrice']].copy()
        mv_day_data = mv_day_data[mv_day_data.Stkcd == Stkcd]
        stock_hf_data = pd.merge(stock_hf_data, mv_day_data, left_on=[
                                 'Stkcd', 'Trddt'], right_on=['Stkcd', 'Trddt'])
        # stock_hf_data.groupby(['stock_code', 'day']).close.sh

    def HF_Factors_return(self):
        pass


class Anomalies():
    """
    >>> main_dir_path=r'D:\个人项目\数据集合\学术研究_股票数据'
    """

    def __init__(self, main_dir_path, init=False):
        self._stock_data = DataCleaner(main_dir_path, init=init)
        self._
        APT = AssetPricingTool()

    # 画个beta disperion 的时序图看看
    def beta_dispersion(beta_dispersion_dir, stock_mon_trade_data):
        """
        >>> beta_dispersion_dir = r'F:\Intrady Beta Pattern\DS\500\15'
        >>> beta_dispersion_dir = r'F:\Intrady Beta Pattern\DD\500\120'
        >>> file = file_list[0]

        >>> stock_mon_trade_data = pd.read_csv(r'F:\Intrady Beta Pattern\SVIC.csv')
        """
        file_list = TB.Tag_FilePath_FromDir(beta_dispersion_dir)
        DS_df = pd.DataFrame()
        for stock in file_list:
            stkcd = re.findall('(\d+).csv', stock)[0]
            stock_data = pd.read_csv(stock, index_col=0).T
            stock_data = stock_data.unstack().reset_index().rename(
                columns={'level_0': 'Trddt', 'level_1': 'time', 0: stkcd})
            stock_data['datetime'] = stock_data.Trddt + ' ' + stock_data.time
            stock_data.index = pd.to_datetime(stock_data.datetime)
            stock_day_data = stock_data.resample('M').mean()
            DS_df = pd.concat([DS_df, stock_day_data], axis=1)

        # DS_df.to_csv(r'F:\Intrady Beta Pattern\DS_15_day.csv')

        DS_day_df = DS_df.unstack().reset_index().rename(
            columns={'level_0': 'Stkcd', 0: 'DS'})
        DS_day_df = DS_day_df.set_index('datetime')
        DS_Mon_df = DS_day_df.groupby('Stkcd').resample(
            'M').mean().reset_index().dropna()
        DS_Mon_df['Trdmnt'] = DS_Mon_df.datetime.apply(lambda x: str(x)[:7])
        DS_Mon_df['Stkcd'] = DS_Mon_df.Stkcd.astype(int)
        DS_Mon_df = pd.merge(stock_mon_trade_data, DS_Mon_df, left_on=[
                             'Trdmnt', 'Stkcd'], right_on=['Trdmnt', 'Stkcd'])
        DS_Mon_df = DS_Mon_df.reset_index(
            drop=True).sort_values(['Stkcd', 'Trdmnt'])
        DS_Mon_df['retShit'] = DS_Mon_df.groupby('Stkcd').Mretwd.shift(-1)

        self._APT.Exec_SingleSort()

        ret, res = APT.Exec_SingleSort(
            avgTag='retShit', sortTag='DS', groupsNum=5, df=DS_Mon_df, weightTag='Msmvttl')

        ret_pivot = ret.reset_index().pivot(
            index='Trdmnt', columns='Group', values='retShit')
        ret_pivot['HML'] = ret_pivot['DS_G01'] - ret_pivot['DS_G05']
        ret_pivot = ret_pivot.loc[:'2022-11', ]
        APT.Exec_NWTest(ret_pivot.HML)

        cum_ret = (ret_pivot + 1).cumsum()
        cum_ret['HML'] = cum_ret['DS_G01'] - cum_ret['DS_G05']
        cum_ret['HML'].plot()
        ret_pivot.mean(axis=0)
        pass

# main_dir_path = r'D:\个人项目\数据集合\学术研究_股票数据'
# ano = Anomalies(main_dir_path)


# file_list = TB.Tag_FilePath_FromDir(beta_dispersion_dir)
# hf_index = pd.read_csv(r'D:\个人项目\Intrady Beta Pattern\hf_index.csv')
# index_tag = hf_index.time.unique().tolist()
# col_tag = hf_index.Trddt.unique().tolist()
# for file in file_list:
#     stkcd = re.findall('(\d+).csv',file)[0]

#     DS = pd.read_csv(file, header=None)
#     DS.index = index_tag
#     DS.columns = col_tag
#     DS = DS.unstack().reset_index().rename(columns={'level_0':'Trddt', 'level_1':'time', 0:'beta_dispersion'})

#     DS['time'] = DS['time'].apply(lambda x:str(int(x)).zfill(4)[:2] + ':' + str(int(x)).zfill(4)[2:] + ':00')
#     DS['datetime'] = DS['Trddt'] + ' ' + DS['time'].astype('str')
#     DS['datetime'] = pd.to_datetime(DS.datetime)
#     DS = DS.set_index('datetime')

#     DS_daily = DS.resample('d').mean().dropna()
#     DS_weekly = DS_daily.resample('w').mean()

    # 思考，到底要不要删掉那么多的交易日？？？

###############################################################################
##
###############################################################################

# base_line = pd.read_csv(r'D:\桌面\Fama-French-五因子模型（经典算法）月收益率（截至到20230531）.csv')
# base_line = base_line.set_index('date')
# base_line_mon_des = base_line.loc['2000-01':'2023-01'].describe()

# base_line = pd.read_csv(r'D:\桌面\Fama-French-五因子模型（经典算法）日收益率（截至到20230531）.csv')
# base_line = base_line.set_index('date')
# base_line_day_des = base_line.loc['2000-01':'2023-01'].describe()


# # np.corrcoef(factors_df.loc['2000-01':'2022-12'].RMW.values, base_line.loc['2000-01':'2023-01'].RMW.values)
# # np.corrcoef(MKT.loc['2000-01':'2022-12'].values, base_line.loc['2000-01':'2023-01'].MKT.values)

# corr_dict = {}
# factors_df = ff5_mon.loc['2000-01':'2023-01']
# base_line = base_line.loc['2000-01':'2023-02']
# for col in factors_df.columns:
#     corr = np.corrcoef(factors_df[col].values, base_line[col].values)[0,1]
#     corr_dict.update({col:corr})


# Values.dropna(inplace=True)
# main_dir_path=r'D:\个人项目\数据集合\学术研究_股票数据'
# self = Factors(main_dir_path)
# Data_Source = self._stock_data._CSMAR

# stock_mon_trade_data = Data_Source.StockMonTradeData_df.copy()
# # drop listdt less than 6 months
# stock_mon_trade_data = self._stock_data.Drop_Stock_InadeListdtData(df=stock_mon_trade_data)

# balance_sheet_data = Data_Source.StockFSData_FS_Combas_df.copy()
# balance_using_data = balance_sheet_data[['Stkcd','Accper','资产总计','负债合计']].copy()
# balance_using_data = balance_using_data[balance_using_data.Accper.str[-5:]=='12-31']
# balance_using_data['NetAsset'] = balance_using_data['资产总计'] - balance_using_data['负债合计']
# balance_using_data = balance_using_data.drop(['资产总计','负债合计'],axis=1)
# balance_using_data['Trdmnt'] = balance_using_data.Accper.apply(lambda x:str(x)[:7])
# balance_using_data = balance_using_data.drop('Accper',axis=1)

# Values = pd.merge(stock_mon_trade_data,balance_using_data,
#              left_on=['Stkcd','Trdmnt'],right_on=['Stkcd','Trdmnt'])[['Stkcd','Trdmnt','Msmvosd','NetAsset','Mretwd']]
# Values = Values.sort_values(['Stkcd','Trdmnt'])
# Values[['NetAsset','Msmvosd']] = Values.groupby(['Stkcd'])[['NetAsset','Msmvosd']].shift(1)


# Values.dropna(inplace=True)
# self._APT.Exec_DoubleSort(avgTag='Mretwd', sortTag1='Msmvosd', sortTag2='NetAsset',
#                           groupsNum1=2, groupsNum2=3, labels1='MV', labels2='BM',
#                           df=Values, timeTag='Trdmnt')


# # 异象类
# class Anomalies(AssetPricingTool):

#     # 计算特质波动率
#     def Cpt_IVOL(self,
#                  x_tag,
#                  y_tag,
#                  window,
#                  intercept=True):
#         # 计算残差
#         df = self.data_set
#         res = AssetPricingTool().Cpt_ResAndBeta(
#             x_tag, y_tag, df, intercept)['res']
#         # 计算特质波动率
#         istd = pd.DataFrame(res).rolling(window).std().values
#         return istd

#     # 提取市值

#     def Cpt_ME(self):
#         return self.data_set.rename(columns={'MarketValue': 'ME'})[['ME']]

#     # 提取账面市值比

#     def Cpt_BM(self):
#         return self.data_set.rename(columns={'NAVToP': 'BM'})[['BM']]

#     # 计算动量

#     def Cpt_MOM(self,
#                 window_long=252,
#                 window_short=21):
#         return self.data_set['close_adj'].shift(window_long)/self.data_set['close_adj'].shift(window_short) - 1

#     # 计算周反转、MAX、MIN

#     def Cpt_REV_MAX_MIN(self):
#         data = self.data_set.copy()
#         try:
#             data.index = pd.to_datetime(data['TradingDate'])
#         except:
#             data.index = pd.to_datetime(data['datetime'])

#         # 将数据重采样为周频率，并获取均值、最大值、最小值
#         rev = data.resample('w')['return'].sum()
#         max = data.resample('w')['return'].max()
#         min = data.resample('w')['return'].min()

#         rev.rename('REV', inplace=True)
#         max.rename('MAX', inplace=True)
#         min.rename('MIN', inplace=True)

#         data = pd.merge(rev, data, right_index=True,
#                         left_index=True, how='outer')
#         data = pd.merge(max, data, right_index=True,
#                         left_index=True, how='outer')
#         data = pd.merge(min, data, right_index=True,
#                         left_index=True, how='outer')

#         # 填充数据
#         data['REV'].fillna(method='bfill', inplace=True)
#         data['MAX'].fillna(method='bfill', inplace=True)
#         data['MIN'].fillna(method='bfill', inplace=True)

#         # 剔除原本不应存在的数据
#         drop_index = list(data[data.iloc[:, -1].isnull()].index)
#         data.drop(index=drop_index, inplace=True)
#         return (data['REV'], data['MAX'], data['MIN'])

#     # 计算流动性ILLIQ

#     def Cpt_ILLIQ(self):
#         data = self.data_set
#         return abs(data['return'].rolling(5).mean())/data['volume'].rolling(5).mean() \
#             / data['close_adj'].rolling(5).mean()

#     # 计算协偏度，协峰度

#     def Cpt_CSK_CKT(self,
#                     index_ret,
#                     window=20):

#         data = self.data_set
#         data = pd.merge(data, index_ret['Retindex'],
#                         left_index=True, right_index=True)
#         # 分别计算股票和指数滚动窗口中的均值，在计算两者的demean
#         index_ret_mean = data['Retindex'].rolling(window).mean()
#         stock_ret_mean = data['return'].rolling(window).mean()
#         index_demean = data['Retindex']-index_ret_mean
#         stock_demean = data['return']-stock_ret_mean

#         # 计算共同的分母部分
#         deno1 = pow(pow(stock_demean, 2).rolling(window).mean(), 0.5)
#         deno2 = pow(index_demean, 2).rolling(window).mean()

#         mole_csk = stock_demean*pow(index_demean, 2)
#         CSK = mole_csk.rolling(window).mean()/deno1/deno2

#         mole_ckt = stock_demean*pow(index_demean, 3)
#         CKT = mole_ckt.rolling(window).mean()/deno1/pow(deno2, 1.5)
#         return (CSK, CKT)


# # 因子类
# class Factors1(Anomalies):
#     # 提问，哪种才是正确的
#     def MKT_SAVIC(self,
#                   retTag,
#                   mkvTag,
#                   rfRate,
#                   dataSet=None,
#                   freq='d'):

#         if dataSet is None:
#             dataSet = self.data_set.copy()

#         timeTag = {'d': 'TradingDate', 'm': 'YearAndMonth'}[freq]
#         riskfree_rate = pd.read_csv(
#             r'D:\个人项目\Bollerslve(2020)\基础数据\无风险利率数据\TRD_Nrrate.csv')
#         if freq == 'd':
#             rfRate = riskfree_rate[['Clsdt', 'Nrrdaydt']]
#             rfRate.rename(columns={'Clsdt': 'TradingDate',
#                           'Nrrdaydt': 'rf'}, inplace=True)

#         elif freq == 'm':
#             rfRate = riskfree_rate[['YearAndMonth', 'Nrrmtdt']]
#             rfRate.rename(columns={'Nrrmtdt': 'rf'}, inplace=True)
#             rfRate = rfRate.groupby('YearAndMonth').tail(1)
#             rfRate = rfRate.groupby(timeTag).mean().reset_index()

#         mktRet = dataSet.groupby(timeTag).apply(self.Cpt_ValueWeight, retTag, mkvTag) \
#                         .rename('return').reset_index()
#         mkt_df = pd.merge(mktRet, rfRate)
#         mkt_df.dropna(subset=['return'], inplace=True)
#         mkt_df['MKT'] = mkt_df['return']*100 - mkt_df.rf
#         mkt_df['MKT'] = mkt_df.MKT.shift(1)
#         mkt_df.index = mkt_df[timeTag]
#         mkt_df['MKT'].loc['2000-01':'2017-01'].describe()

#         return mkt_df.dropna()

#         # fa = pd.read_csv(r'D:\Size and Value in China\CH_4_fac_daily_update_20211231.csv',skiprows=9)
#         fa = pd.read_csv(
#             r'D:\Size and Value in China\CH_4_fac_update_20211231.csv', skiprows=9)
#         fa.index = fa.mnthdt.apply(lambda x: str(x)[0:4]+'-'+str(x)[4:6])
#         # fa.index = fa.date.apply(lambda x:str(x)[0:4]+'-'+str(x)[4:6]+'-'+str(x)[6:])
#         a = pd.merge(mkt_df['MKT'], fa, left_index=True,
#                      right_index=True).MKT.corr(a.mktrf)
#         a.MKT.corr(a.mktrf)

#         # bas_mkt = fa.mktrf.loc['2000-01':'2017-01'].describe()
#         # my_mkt = mkt_df['MKT'].loc['2000-01':'2017-01'].describe()
#         # res = pd.merge(bas_mkt,my_mkt,left_index=True,right_index=True)
#         # res['corr'] = pd.merge(mkt_df['MKT'],fa,left_index=True,right_index=True).MKT.corr(a.mktrf)
#         # res.to_csv(r'D:\Size and Value in China\结果对比\MKT_Mon_1.csv')

#         fa.VMG.loc['2000-01':'2017-01'].describe()
#         fa.SMB.loc['2000-01':'2017-01'].describe()
#         fa.PMO.loc['2000-01':'2017-01'].describe()


#     # 计算SAVIC版本的规模因子和价值因子
#     def SMB_VMG_PMO_SAVIC(self,
#                           dataSet=None,
#                           freq='d'):

#         if dataSet is None:
#             dataSet = self.data_set.copy()
#         timeTag = {'d': 'TradingDate', 'm': 'YearAndMonth'}[freq]

#         # MonData = pd.read_csv(r'D:\个人项目\Bollerslve(2020)\基础数据\SAVIC_MonData.csv',index_col=0)
#         # MonData['Symbol'] = MonData.Symbol.apply(TB.Fill_StockCode)
#         finData = pd.read_csv(
#             r'D:\个人项目\Bollerslve(2020)\基础数据\FinReportData.csv', index_col=0)
#         finData.rename(columns={'A股股票代码_A_Stkcd': 'Symbol',
#                                 '报表类型_Reporttype': 'ReportType',
#                                 '截止日期_EndDt': 'EndDt',
#                                 '信息发布日期_Infopubdt': 'Infopubdt',
#                                 '扣除非经常性损益后净利润(元)_NetprfCut': 'NetprfCut',
#                                 '净利润(元)_Netprf': 'Netprf'}, inplace=True)

#         finData.dropna(subset='Symbol', inplace=True)
#         finData['Symbol'] = finData.Symbol.astype(int).apply(TB.Fill_StockCode)

#         def func(x):
#             if str(x)[:2] in ['60', '00', '30']:
#                 return x
#             else:
#                 return 0
#         finData = finData[finData.Symbol.apply(func) != 0]
#         finData.dropna(subset=['Netprf'], inplace=True)
#         finData = finData.drop_duplicates(subset=['Symbol', 'EndDt'])
#         finData = finData.sort_values(
#             ['Symbol', 'EndDt']).reset_index(drop=True)
#         value = finData[finData.NetprfCut.isnull()].Netprf
#         finData.loc[value.index, 'NetprfCut'] = value.values
#         finData = TB.Tag_TradeYAM(finData, 'Infopubdt')[['Symbol',
#                                                         'NetprfCut',
#                                                          'YearAndMonth']]
#         dataSet['Symbol'] = dataSet.Symbol.apply(TB.Fill_StockCode)
#         dataSet = pd.merge(dataSet, finData, how='left')
#         dataSet['NetprfCut'] = dataSet.groupby(
#             'Symbol').NetprfCut.fillna(method='ffill')

#         err_list = []
#         for npc in dataSet[dataSet.NetprfCut.isnull()].groupby('Symbol'):
#             df = npc[1]
#             index_col = df.index[0]
#             stkcd = npc[0]
#             dt = df.loc[index_col, 'YearAndMonth']

#             try:
#                 comp_df = finData[finData.Symbol == stkcd]
#                 data = comp_df[comp_df.YearAndMonth <=
#                                dt].iloc[-1, :].NetprfCut
#                 dataSet.loc[index_col, 'NetprfCut'] = data
#             except:
#                 err_list.append(stkcd)
#         dataSet['NetprfCut'] = dataSet.groupby(
#             'Symbol').NetprfCut.fillna(method='ffill')

#         # 使用CSMAR数据补齐
#         netpData = pd.read_csv(
#             r'D:\个人项目\Bollerslve(2020)\基础数据\财务报表1\FS_Comins.csv')
#         netpData = netpData[netpData.Typrep == 'A']
#         netpData = netpData[['Stkcd', 'Accper', 'B002000000']]
#         outData = pd.read_csv(
#             r'D:\个人项目\Bollerslve(2020)\基础数据\财务报表1\FN_FN009.csv')
#         outData = outData[outData.typrep == 1]
#         outData = outData[outData.FN_Fn00901 == '合计']
#         outData.rename(columns={'stkcd': 'Stkcd',
#                        'accper': 'Accper'}, inplace=True)
#         outData = outData[['Stkcd', 'Accper', 'FN_Fn00902']]
#         npcData = pd.merge(netpData, outData, how='left')
#         npcData.FN_Fn00902.fillna(0, inplace=True)
#         npcData['npc'] = npcData.B002000000 - npcData.FN_Fn00902
#         npcData.dropna(subset='npc', inplace=True)
#         npcData = TB.Tag_TradeYAM(npcData, 'Accper')
#         npcData.rename(columns={'Stkcd': 'Symbol'}, inplace=True)
#         npcData['Symbol'] = npcData.Symbol.apply(TB.Fill_StockCode)
#         npcData = npcData[['Symbol', 'npc', 'YearAndMonth']]

#         # 先不做任何滞后处理
#         dataSet = pd.merge(dataSet, npcData, how='left')
#         dataSet['npc'] = dataSet.groupby('Symbol').npc.fillna(method='ffill')
#         fillIndex = dataSet[dataSet.NetprfCut.isnull()].index
#         dataSet.loc[fillIndex, 'NetprfCut'] = dataSet.loc[fillIndex, 'npc']
#         dataSet.dropna(subset=['NetprfCut'], inplace=True)
#         dataSet['EP'] = dataSet.NetprfCut/dataSet.MarketValue*1000
#         # dataMon1.loc[dataMon1[dataMon1.EP<0].index,'EP'] = 0

#         # #######################################################################

#         # # df_day = pd.read_csv(r'D:\个人项目\Bollerslve(2020)\基础数据\StockDayData.csv',usecols=['Symbol','YearAndMonth','PE1A','MarketValue'])
#         # # dfMon = df_day.groupby(['Symbol','YearAndMonth']).tail(1)
#         # # dfMon['Symbol'] = dfMon.Symbol.apply(TB.Fill_StockCode)
#         # # dfMon['EP'] = 1/dfMon['PE1A']
#         # # df = dataMon.copy()
#         # # df= pd.merge(df,dfMon)
#         # # df = df[['YearAndMonth', 'Symbol', 'EP', 'Msmvttl', 'Mretwd', 'retLag']]
#         # # df.EP.fillna(0,inplace=True)

#         # #######################################################################

#         # DataMon = pd.read_csv(r'D:\个人项目\Bollerslve(2020)\基础数据\SAVIC_MonData.csv',index_col=0)
#         # netp =  pd.read_csv(r'D:\个人项目\Bollerslve(2020)\基础数据\财务报表1\FS_Comins.csv')
#         # netp = netp[netp.Typrep=='A']
#         # netp = netp[['Stkcd', 'Accper', 'B002000000']]
#         # # outData = pd.read_csv(r'D:\个人项目\Bollerslve(2020)\基础数据\财务报表1\FN_FN009.csv')
#         # # outData = outData[outData.typrep==1]
#         # # outData = outData[outData.FN_Fn00901=='合计']
#         # # outData.sort_values(['stkcd','accper'],inplace=True)

#         # def ChangeDate(date):
#         #     Year = date[:4]
#         #     month = date[5:7]
#         #     if month == '12':
#         #         return str(int(Year)+1)+'-'+'01'
#         #     if month == '03':
#         #         return Year+'-'+'04'
#         #     if month == '06':
#         #         return Year+'-'+'07'
#         #     if month == '09':
#         #         return Year+'-'+'10'

#         # netp['YearAndMonth'] = netp['Accper'].apply(ChangeDate)
#         # netp = netp[~netp.YearAndMonth.isnull()]
#         # netp.rename(columns={'Stkcd':'Symbol'},inplace=True)
#         # netp['Symbol'] = netp['Symbol'].apply(TB.Fill_StockCode)
#         # dataMon1 = pd.merge(dataMon,netp,how='left')
#         # # dataMon = pd.merge(DataMon,netp,how='left')
#         # # dataMon.dropna(subset='Clsdt',inplace=True)
#         # dataMon1['B002000000'] = dataMon1.B002000000.fillna(method='ffill')
#         # dataMon1.dropna(subset=['B002000000'],inplace=True)
#         # # dataMon['EP'] = dataMon['B002000000']/dataMon['Msmvttl']
#         # dataMon1['EP'] = dataMon1['B002000000']/dataMon1['MarketValueA1']
#         # dataMon1.sort_values(['Symbol','YearAndMonth'],inplace=True)

#         Dsort1 = self.Exec_DoubleSort('retLag',
#                                       'EP',
#                                       'MarketValue',
#                                       [0, 0.3, 0.7, 1],
#                                       2,
#                                       timeTag=timeTag,
#                                       weightTag='MarketValue',
#                                       df=dataSet)[0]

#         VMG = -1/2*(Dsort1[Dsort1['Group'] == 'EP_G01@MarketValue_G02']['retLag'] +
#                     Dsort1[Dsort1['Group'] == 'EP_G01@MarketValue_G01']['retLag']) + \
#             1/2*(Dsort1[Dsort1['Group'] == 'EP_G03@MarketValue_G02']['retLag'] +
#                  Dsort1[Dsort1['Group'] == 'EP_G03@MarketValue_G01']['retLag'])
#         VMG.loc['2000-01':'2017-01'].describe()

#         SMB1 = 1/3*(Dsort1[Dsort1['Group'] == 'EP_G01@MarketValue_G01']['retLag'] +
#                     Dsort1[Dsort1['Group'] == 'EP_G02@MarketValue_G01']['retLag'] +
#                     Dsort1[Dsort1['Group'] == 'EP_G03@MarketValue_G01']['retLag']) - \
#             1/3*(Dsort1[Dsort1['Group'] == 'EP_G01@MarketValue_G02']['retLag'] +
#                  Dsort1[Dsort1['Group'] == 'EP_G02@MarketValue_G02']['retLag'] +
#                  Dsort1[Dsort1['Group'] == 'EP_G03@MarketValue_G02']['retLag'])

#         Dsort2 = self.Exec_DoubleSort('retLag',
#                                       'ABT',
#                                       'MarketValue',
#                                       [0, 0.3, 0.7, 1],
#                                       2,
#                                       timeTag=timeTag,
#                                       weightTag='MarketValue',
#                                       df=dataSet)[0]

#         PMO = 1/2*(Dsort2[Dsort2['Group'] == 'ABT_G01@MarketValue_G02']['retLag'] +
#                    Dsort2[Dsort2['Group'] == 'ABT_G01@MarketValue_G01']['retLag']) - \
#             1/2*(Dsort2[Dsort2['Group'] == 'ABT_G03@MarketValue_G02']['retLag'] +
#                  Dsort2[Dsort2['Group'] == 'ABT_G03@MarketValue_G01']['retLag'])
#         PMO.loc['2000-01':'2017-01'].describe()

#         SMB2 = 1/3*(Dsort2[Dsort2['Group'] == 'ABT_G01@MarketValue_G01']['retLag'] +
#                     Dsort2[Dsort2['Group'] == 'ABT_G02@MarketValue_G01']['retLag'] +
#                     Dsort2[Dsort2['Group'] == 'ABT_G03@MarketValue_G01']['retLag']) - \
#             1/3*(Dsort2[Dsort2['Group'] == 'ABT_G01@MarketValue_G02']['retLag'] +
#                  Dsort2[Dsort2['Group'] == 'ABT_G02@MarketValue_G02']['retLag'] +
#                  Dsort2[Dsort2['Group'] == 'ABT_G03@MarketValue_G02']['retLag'])
#         SMB = 1/2*(SMB1+SMB2)
#         SMB.loc['2000-01':'2017-01'].describe()

#         b = pd.merge(SMB.shift(1), fa, left_index=True, right_index=True)
#         b.SMB.corr(b.retLag)
#         C = pd.merge(VMG.shift(1), fa, left_index=True, right_index=True)
#         C.VMG.corr(C.retLag)
#         d = pd.merge(PMO.shift(1), fa, left_index=True, right_index=True)
#         d.PMO.corr(d.retLag)

#         # bas_vmg = fa.VMG.loc['2000-01':'2017-01'].describe()
#         # my_vmg = VMG.loc['2000-01':'2017-01'].describe().rename('myVMG')
#         # res = pd.merge(bas_vmg,my_vmg,left_index=True,right_index=True)
#         # b = pd.merge(VMG.shift(1),fa,left_index=True,right_index=True)
#         # res['corr'] = b.VMG.corr(b.retLag)
#         # res.to_csv(r'D:\Size and Value in China\结果对比\VMG_Mon.csv')

#         # bas_vmg = fa.VMG.loc['2000-01':'2017-01'].describe()
#         # my_vmg = VMG.loc['2000-01':'2017-01'].describe().rename('myVMG')
#         # res = pd.merge(bas_vmg,my_vmg,left_index=True,right_index=True)
#         # b = pd.merge(VMG.shift(1),fa,left_index=True,right_index=True)
#         # res['corr'] = b.VMG.corr(b.retLag)
#         # res.to_csv(r'D:\Size and Value in China\结果对比\VMG_Mon.csv')

#         # bas_smb = fa.SMB.loc['2000-01':'2017-01'].describe()
#         # my_smb = SMB.loc['2000-01':'2017-01'].describe().rename('mySMB')
#         # res = pd.merge(bas_smb,my_smb,left_index=True,right_index=True)
#         # c = pd.merge(SMB.shift(1),fa,left_index=True,right_index=True)
#         # res['corr'] = c.SMB.corr(c.retLag)
#         # res.to_csv(r'D:\Size and Value in China\结果对比\SMB_Mon.csv')

#         # bas_pmo = fa.PMO.loc['2000-01':'2017-01'].describe()
#         # my_pmo = PMO.loc['2000-01':'2017-01'].describe().rename('myPMO')
#         # res = pd.merge(bas_pmo,my_pmo,left_index=True,right_index=True)
#         # d = pd.merge(PMO.shift(1),fa,left_index=True,right_index=True)
#         # res['corr'] = d.PMO.corr(d.retLag)
#         # res.to_csv(r'D:\Size and Value in China\结果对比\PMO_Mon.csv')
