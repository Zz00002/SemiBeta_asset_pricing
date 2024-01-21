# -*- coding: utf-8 -*-
"""
Created on Tue Jan 17 14:29:16 2023

@author: asus
"""

# Asset Pricing Foundation Module

import numpy as np
import pandas as pd
import scipy
import math
import statsmodels.api as sm
from linearmodels import FamaMacBeth

class AssetPricingTool:

    def __init__(self,
                 df=None):

        if type(df) == pd.core.frame.DataFrame:
            self.data_set = df.copy()
        else:
            self.data_set = pd.DataFrame(df)


    # Exec Newey-West Test
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


    # Calculate OLS residuals and beta
    def Cpt_ResAndBeta(self,
                       x_tag,
                       y_tag,
                       df=None,
                       intercept=True,
                       NWtest=False):

        if df is None:
            df = self.data_set

        X = pd.DataFrame(columns=x_tag)
        X[x_tag] = df[x_tag]
        if intercept == True:
            X['intercept'] = 1

        X = np.mat(X)
        Y = np.mat(df[[y_tag]])
        try:
            beta = ((X.T.dot(X)).I).dot(X.T).dot(Y)
            res = Y - X.dot(beta)

        except:

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


    ###########################################################################
    ## Related to stock sorting
    ###########################################################################

    ## Intermediary function, must be passed df
    # Calculation of weighted average
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


    # Calculate the cross-sectional mean of each Group in the DataFrame
    def Cpt_CrossWeightMean(self,
                            df,
                            avgTag,
                            timeTag,
                            weightTag=None):

        if weightTag == None:
            ew = df.groupby([timeTag, 'Group'], as_index=False)[avgTag].mean()
            ew = ew.set_index(timeTag)
            return ew

        else:
            vw = df.groupby([timeTag, 'Group'], as_index=False) \
                .apply(self.Cpt_ValueWeight, avgTag, weightTag) \
                .rename(columns={None: avgTag})
            vw = vw.set_index(timeTag)
            return vw


    # Calculate stock grouping
    def Creat_StockGroups(self,
                          data,
                          sortTag,
                          groupsNum,
                          labels=None):

        df = data.copy()
        if labels is None:
            if type(groupsNum) is int:
                labels = ['{}_G0'.format(sortTag) + str(i)
                          for i in range(1, groupsNum + 1)]
            else:
                labels = ['{}_G0'.format(sortTag) + str(i)
                          for i in range(1, len(groupsNum))]
        try:
            groups = pd.DataFrame(pd.qcut(df[sortTag], groupsNum, labels=labels).astype(str)) \
                .rename(columns={sortTag: 'Group'})
        except:
            groups = pd.DataFrame(pd.qcut(df[sortTag].rank(method='first'), groupsNum, labels=labels).astype(str)) \
                .rename(columns={sortTag: 'Group'})
        return groups


    # Perform computational grouping operations on DataFrame
    def Exec_GroupStockDf(self,
                          df,
                          groupby_list,
                          sortTag,
                          groupsNum,
                          labels=None):

        df.index = pd.MultiIndex.from_frame(df[groupby_list])
        df = df.sort_index()

        df['Group'] = np.nan
        for temp in df.groupby(df.index):
            df_temp = temp[1]
            if df_temp.shape[0]>=groupsNum:
                group_temp = self.Creat_StockGroups(
                    df_temp, sortTag, groupsNum, labels)
                index_temp = temp[0]
                df.loc[index_temp, 'Group'] = group_temp

        df.reset_index(drop=True, inplace=True)
        return df.dropna()


    ###########################################################################

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
        df = self.Exec_GroupStockDf(
            df, [timeTag], sortTag1, groupsNum1, labels1)
        df.rename(columns={'Group': 'G1'}, inplace=True)

        if SortMethod == 'Independent':
            df = self.Exec_GroupStockDf(
                df, [timeTag], sortTag2, groupsNum2, labels2)
        else:
            df = self.Exec_GroupStockDf(
                df, [timeTag, 'G1'], sortTag2, groupsNum2, labels2)
        df.rename(columns={'Group': 'G2'}, inplace=True)

        df['Group'] = df.G1 + '@' + df.G2
        result = self.Cpt_CrossWeightMean(df, avgTag, timeTag, weightTag)
        result[avgTag] = result[avgTag]
        result['Group_' +
               sortTag1] = result.Group.apply(lambda x: x.split('@')[0])
        result['Group_' +
               sortTag2] = result.Group.apply(lambda x: x.split('@')[1])
        result = result.sort_values([timeTag, 'Group'])

        return (result, df)


    # Re-aggregate the results of the cross-section grouping aggregation on a time-series basis and generate a two-dimensional data table
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

        Ssort['HML'] = -Ssort['{}_G01'.format(
            sortTag)] + Ssort['{}_G0{}'.format(sortTag,groupsNum)]
        Ssort = Ssort.shift(1).dropna()
        SSortRes = pd.DataFrame(Ssort.mean(), columns=['AvgRet'])

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
            bigTag = sort_table.index[-1]
            smallTag = sort_table.index[0]
            bigGroup = df[df['Group_{}'.format(sortTag_key)] == bigTag]
            smallGroup = df[df['Group_{}'.format(sortTag_key)] == smallTag]
            HML = bigGroup[avgTag] - smallGroup[avgTag]
            HML.dropna(inplace=True)

            sortRes.loc['HML', cont] = HML.mean()
            sortRes.loc['t', cont] = self.Exec_NWTest(test_array=HML)['tstat']
            sortRes.loc['p', cont] = self.Exec_NWTest(test_array=HML)['pval']

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


    # Exec fama-macbeth regression and show the summary results
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
    

