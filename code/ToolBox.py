# -*- coding: utf-8 -*-
"""
Created on Sat Sep 10 19:45:36 2022

@author: xgt
"""

import pandas as pd
import datetime
import numpy as np
import os
import mat4py as m4p
import re
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from multiprocessing import Pool
import numba as nb


###################### Calculated Functions #########################################
def z_score(x):
    return (x - np.mean(x))/np.std(x)

@nb.njit
def nb_mean(arr, axis):
    return np.sum(arr, axis=axis) / arr.shape[axis]

###############################################################################

# Given a dictionary of folder hierarchies, automatically generate folders and subfolders
def create_folders(parent_path, folders_dict):
    for folder_name, sub_folders in folders_dict.items():
        folder_path = os.path.join(parent_path, folder_name)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
            print("Folder {} created.".format(folder_path))
        else:
            print("Folder {} already exists.".format(folder_path))
        if sub_folders:
            create_folders(folder_path, sub_folders)


# Get the absolute path of a file from a folder
def Tag_FilePath_FromDir(file_dir, is_all=True, suffix='csv'):
    path_list = []
    # 获取当前文件夹，以及所有子文件夹下的所有文件
    for root,dirs,files in os.walk(file_dir):
        for file in files:
            # 获取指定文件类型的文件
            if suffix is None:
                path_list.append(os.path.join(root,file))
                
            elif file.split('.')[-1] in suffix:
                path_list.append(os.path.join(root,file))
                
        # 只获取当前文件夹下的所有文件
        if is_all is False:
            break
    print('number of files: ',len(path_list))
    
    return path_list


# Consolidation of batch-downloaded base data into a single dataset
def Concat_RawData(dataDir,skip_cols=[]):
    dataFile_list = Tag_FilePath_FromDir(dataDir)
    dataDf = pd.DataFrame()
    for file in dataFile_list:
        try:
            try:
                df = pd.read_csv(file, usecols=lambda column: column not in skip_cols, encoding='gbk')
            except:
                df = pd.read_csv(file, usecols=lambda column: column not in skip_cols, encoding='utf-8')
            dataDf = pd.concat([dataDf,df])
        except:
            print('error hanppens while concating data into df')
            continue
            
    return dataDf


# Get the given stock Rex 1min data
def Fetch_Stock_HFdata_from_Resset(stkcd, 
                                   data_dir='F:\\HF_MIN\\Resset\\Csv_data', asset_type='stock', minType=1):
    
    # get high frequency stock data
    hf_data_df = pd.DataFrame()
    for year in range(2005,2023):
        base_dir = data_dir + '\\{0}\\{1}'.format(year, asset_type)
        data_path = base_dir + '\\{0}.csv'.format(str(stkcd).zfill(6))
        try:
            df = pd.read_csv(data_path)
            hf_data_df = pd.concat([hf_data_df, df])
        except:
            try:
                data_path = base_dir + '\\{0}.csv'.format(int(stkcd))
                df = pd.read_csv(data_path)
                hf_data_df = pd.concat([hf_data_df, df])
            except:
                continue
        
    # select data that time within 09:30:00-11:30:00 and 13:00:00-15:00:00
    all_time_list = CreateTimeStrList(start='09:31:00', end='15:00:00', standard='M', spacing=1)
    drop_time_list = CreateTimeStrList(start='11:31:00', end='13:00:00', standard='M', spacing=1)
    use_time_list = list(set(all_time_list) - set(drop_time_list))
    hf_data_df = hf_data_df.loc[hf_data_df.time.isin(use_time_list)]
    
    hf_data_df = hf_data_df.rename(columns={'date':'Trddt'})    
    hf_data_df = hf_data_df.sort_values(['Trddt','time'])
    
    hf_data_df['datetime'] = pd.to_datetime(hf_data_df['Trddt'] + ' ' + hf_data_df['time'])
    hf_data_df.set_index('datetime', inplace=True)
    hf_data_df.index = hf_data_df.index + pd.DateOffset(minutes=minType-1)

    # Define the resampling frequency to 5 minutes ('5T')
    hf_data_df = hf_data_df.groupby('stkcd').resample('{}T'.format(minType)).agg({'open': 'first',
                                                                                    'high': 'max',
                                                                                    'low': 'min',
                                                                                    'close': 'last',
                                                                                    'volume': 'sum',
                                                                                    'turnover': 'sum'}).reset_index()  
    hf_data_df = hf_data_df.reset_index(drop=True)
    hf_data_df['Trddt'] = hf_data_df['datetime'].dt.date.astype(str)
    hf_data_df['time'] = hf_data_df['datetime'].dt.time.astype(str)
    
    
    # Drop the original 'datetime' column if not needed
    hf_data_df.drop(columns=['datetime'], inplace=True)    
    n = 240/minType
    hf_data_df = hf_data_df.loc[hf_data_df.time.isin(use_time_list)]
    count_df = hf_data_df.groupby('Trddt').count()
    hf_data_df = hf_data_df[hf_data_df.Trddt.isin(count_df[count_df.open>=(3/4*n)].index)]
    
    hf_data_df = hf_data_df.set_index('Trddt')
    hf_data_df = hf_data_df.groupby('Trddt').fillna(method='ffill')
    hf_data_df = hf_data_df.groupby('Trddt').fillna(method='bfill')
    hf_data_df = hf_data_df.reset_index()
    
    # hf_data_df = pd.read_csv(r'F:\HF_MIN\Resset\A_Index_15.csv')
    # minType = 15
    # all_dates = pd.DataFrame({'Trddt': sorted(hf_data_df.Trddt.unique())})
    # all_times = pd.DataFrame({'time': hf_data_df.time.unique().tolist()})
    # merged_df = pd.merge(all_dates, all_times, how='cross')
    # hf_data_df = pd.merge(merged_df, hf_data_df, on=['time', 'Trddt'], how='left')
    
    
    # hf_data_df.to_csv(r'F:\HF_MIN\Resset\A_Index_15.csv',index=False)
    return hf_data_df


