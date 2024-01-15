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



################################ 绘图函数 #####################################
# 给出一个df，以column为X，利用对应列的数据绘制箱型图
def Box_Diagram(df,title='Box_Diagram',save_path=None,standardize=True,whis=2):
    
    plt.figure()
    plt.title(title)
    if standardize:
        plt.boxplot(df.apply(z_score).values, labels=df.columns, 
                    showmeans=True, showfliers=True,whis=whis)
    else:
        plt.boxplot(df.values, labels=df.columns, 
                    showmeans=True, showfliers=True,whis=whis)
    if save_path is not None:
        plt.savefig(save_path + '\\' + title + '.png', transparent=True)
    else:
        plt.show()

        

# 给出一个df，以index为x，column为积累对象，绘制积累柱状图
def Cumulated_BarChart(df,title='Cumulated_BarChart', save_path=None, colors=None):
    # 定义颜色
    if colors is None:
        colors = plt.cm.Set3.colors
        
    # 绘图
    fig, ax = plt.subplots(figsize=(15,10))
    plt.title(title)
    data = df.values
    legend_patches = []  # 记录legend patch

    # 以index为横轴，有多少个index就绘制多少个条形
    for i, index in enumerate(df.index):
        bottom = 0
        index_patches = []  # 记录当前条形的patches

        # 获取当前index下，各个column的信息
        for j, column in enumerate(df.columns):
            sales_column = data[i][j]
            color = colors[j % len(colors)]  # 根据循环取色，每个column取固定的颜色

            # 绘制当前条形图，并储存
            patch = ax.bar(i, sales_column, bottom=bottom, color=color)

            # 记录当前条形图对应的legend patch，创建一个Patch对象，每个column存储一次，
            # 从而形成一一对应的颜色和label
            if column not in [p.get_label() for p in legend_patches]:
                legend_patch = mpatches.Patch(facecolor=color, edgecolor="black", label=column)
                legend_patches.append(legend_patch)
            index_patches.append(patch)

            bottom += sales_column

        ax.set_xticks(range(len(df.index)))
        ax.set_xticklabels(df.index,rotation=45)
        ax.legend(handles=legend_patches)
        
    if save_path is not None:
        plt.savefig(save_path + '\\' + title + '.png', transparent=True)
    else:
        plt.show()

# 给出pivot后的df，绘制热力图
def HeatMap(df_pivot,title='HeatMap',figsize=(15,10),dpi=200,save_path=None):
    
    plt.figure(figsize=figsize,dpi=dpi)
    sns.heatmap(df_pivot, cmap='coolwarm',annot=True, fmt=".4f")
    plt.title(title)
    if save_path is not None:
        plt.savefig(save_path + '\\' + title + '.png', transparent=True)
    else:
        plt.show()


# def Plot_HeatMap(df,mapnum=4,xlabel=None,ylabel=None,title=None,day=None):
    
#     info_num = int(len(df.columns)/mapnum)+1
    
#     mpl.rcParams['font.family'] = 'SimHei'
#     plt.rcParams['axes.unicode_minus'] = False 
    
#     fig=plt.figure(figsize=(200,160),dpi=100)
#     fig.suptitle(title,fontsize=30,color='red')
#     plt.subplots_adjust(left=None, bottom=0.15, right=None, top=None, wspace=0.2, hspace=0.4)
#     c=0
#     for i in range(4):
#         c=c+1
#         ax=fig.add_subplot(2,2,c)
#         df1 = df.iloc[:,info_num*(i):info_num*(i+1)]
        
#         x = [i for i in range(len(df1.columns))]
#         xticks = list(df1.columns)
#         ax.set_xticks(x,xticks,rotation=-45,fontsize=10)
#         y = [i for i in range(len(df1.index))]
#         yticks = list(df1.index)
#         ax.set_yticks(y,yticks,rotation=-45,fontsize=10)
      
#         # ax.set_xlabel(xlabel)
#         # ax.set_ylabel(ylabel)
#         quat80 = df.max().sort_values(ascending=True).quantile(0.8)
#         quat20 = df.min().sort_values(ascending=True).quantile(0.2)
#         sns.heatmap(df1,cmap='OrRd',linecolor='black',linewidths=0.5,vmax=quat80,vmin=quat20)
#         plt.xlabel(xlabel)
#         plt.ylabel(ylabel)

#     plt.show()
#     plt.savefig('./HeatMap_{}.png'.format(day))


###################### 数据计算型函数 #########################################
def z_score(x):
    return (x - np.mean(x))/np.std(x)


@nb.njit
def nb_mean(arr, axis):
    return np.sum(arr, axis=axis) / arr.shape[axis]


@nb.njit
def nb_percentile(arr, percentile):
    sorted_arr = np.sort(arr)
    index = int(sorted_arr.shape[0] * percentile)
    return sorted_arr[index]

@nb.njit
def nb_isin(arr, values):
    result = [i for i in range(arr.shape[0])]
    for i in range(arr.shape[0]):
        result[i] = arr[i] in values
    return np.array(result)


###############################################################################

def Ist_TopRow_into_df(df, tag=''):
    df = df.reset_index(drop=True)
    df.loc[-1] = tag
    df.index = df.index + 1
    df = df.sort_index()  
    
    return df


def Crt_Time_merged_df(date_arr, spacing=1):
    
    all_dates = pd.DataFrame({'Trddt': date_arr})
    all_time_list = CreateTimeStrList(start='09:31:00', end='15:00:00', standard='M', spacing=spacing)
    drop_time_list = CreateTimeStrList(start='11:31:00', end='13:00:00', standard='M', spacing=spacing)
    use_time_list = list(set(all_time_list) - set(drop_time_list))
    all_times = pd.DataFrame({'time': use_time_list})
    merged_df = pd.merge(all_dates, all_times, how='cross').sort_values(['Trddt','time'])

    return merged_df


# 给出文件夹层次字典，自动生成文件夹和子文件夹
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


# 从文件夹中获取文件的绝对路径
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


# 将分批下载的基础数据合并成一个数据集
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


# 将股票代码补全
def Fill_StockCode(stockcode):
    stockcode = str(stockcode)
    num = len(stockcode)
    gap = 6-num
    if gap>0:
        code = '0'*gap+str(stockcode)
        return code
    else:
        return stockcode
        

# 读取个股分时数据
def Get_StockTsData_FromLoc(code,day='2022-09-16'):
    try:
        path = r'D:\个人项目\每日股票分时数据\{0}\{1}.csv'.format(day,code)
        df = pd.read_csv(path,usecols=['股票代码_含前缀',
                                       '日期',
                                       '秒级时间戳_不含日期信息',
                                       '当前价',
                                       # '成交量'
                                       ])

    except:
        df = pd.DataFrame()
        pass
    
    return df




def CreateTimeStrList(start=None, 
                      end=None,
                      standard='seconds',
                      spacing=3):
    '''
    目标：
        构建一个函数，实现在输入一个起始日期与结束日期，选择按照时、分、秒作为间隔基准，
        并给定一定的时间间隔，按照该时间间隔插入数据，形成一个元素为字符串型的时间列表;

    Parameters
    ----------
    start : str, optional
        数据开始的时间，格式为%H:%M:%S,时分秒处都要保留两位;. The default is None.
        
    end : str, optional
        数据结束的时间，格式为%H:%M:%S,时分秒处都要保留两位;. The default is None.
        
    standard : str, optional
        按照时、分、秒进行数据间隔写入，
        1.如果按照小时进行数据间隔计算，则可以可选参数为：H、HOURS
        2.如果按照分钟进行数据间隔计算，则可以可选参数为：M、MINUTES
        3.如果按照秒进行数据间隔计算，则可以可选参数为：S、SECONDS
        (不区分大小写）;. The default is 'seconds'.
        
    spacing : int, optional
        列表中每一个元素的时间间隔. The default is 3.

    Returns
    -------
    time_list : list
        带有一连串时间字符串的列表
    '''                                                    

    # 创建日期辅助表
    if start is None:
        start = '00:00:00'
    if end is None:
        end = datetime.datetime.now().strftime('%H:%M:%S')

    # 转为日期格式
    time_list = []
    time_list.append(start)

    start = datetime.datetime.strptime(start, '%H:%M:%S')  # 这里的strptime是将str格式转为时间格式
    end = datetime.datetime.strptime(end, '%H:%M:%S')
    standard = standard.lower()
    
    while start < end:
        if standard in ['h','hours']:
            # 日期叠加一小时
            start += datetime.timedelta(hours=+spacing)
            # 日期转字符串存入列表
            time_list.append(start.strftime('%H:%M:%S'))   # 这里的strftime是将时间格式转换为字符串格式
            
        if standard in ['m','minutes']:
            # start
            start += datetime.timedelta(minutes=+spacing)
            # 日期转字符串存入列表
            time_list.append(start.strftime('%H:%M:%S'))

        if standard in ['s','seconds']:
            # 日期叠加一小时
            start += datetime.timedelta(seconds=+spacing)
            # 日期转字符串存入列表
            time_list.append(start.strftime('%H:%M:%S'))
            
    return time_list 


def Fill_TADToStockTsData(modified_df):
    '''
    该函数的耦合度很高，且设定比较死板
    
    目标：
        构建一个函数，实现将盘后获取的数据通过插值将缺失3s数据填充;

    Parameters
    ----------
    modified_df : DataFrame
        需要进行数据转换的数据，以dataframe的形式传入
        该dataframe需要为某一股票当天含秒级时间戳的交易数据.

    Returns
    -------
    modify_df : DataFrame
        经过数据处理的填充了值的df
        
    '''
                                                                                                                                                                        
    # 创建基准3秒交易时间列表
    time_morning_list = CreateTimeStrList('09:30:00','11:30:00')
    time_afternoon_list = CreateTimeStrList('13:00:00','14:57:00')
    time_standard_list = time_morning_list+time_afternoon_list
    
    # 创建盘后记录的3秒交易时间列表，并使用基准列表与其做差，得到缺失时间信息列表，并对应生成插入数据字典
    time_list = modified_df['秒级时间戳_不含日期信息'].astype(str).tolist()
    time_diff = list(set(time_standard_list)-set(time_list))
    stock_code = modified_df.iloc[0,0]  # 获取股票代码
    day = modified_df.at[0,'日期']  # 获取日期
    
    # 将需要插入的信息存储到字典    
    info_dict = {'股票代码_含前缀':stock_code,
                 '日期':day,
                 '秒级时间戳_不含日期信息':'',
                 '当前价':np.nan,
                 '成交量':0,
                 # '成交额':decimal.Decimal(0)
                 }  
    
    # 将缺失时间对应的数据插入dataframe
    info_list = []  # 用于储存插入信息
    long = len(time_diff)
    for num in range(long):
        # info_dict['秒级时间戳_不含日期信息'] = datetime.datetime.strptime(time_diff[num], '%H:%M:%S').time()
        info_dict['秒级时间戳_不含日期信息'] = time_diff[num]
        info_list.append(info_dict.copy())
    df = pd.DataFrame(info_list)  # 将一系列缺失的数据转化为dataframe
    modified_df = pd.concat([modified_df,df])
    
    # 对插值后的数据按时间进行排序，再填充当前价信息，根据实际意义，使用前价进行填充
    modified_df = modified_df.sort_values(by='秒级时间戳_不含日期信息',ascending=True).reset_index(drop=True)
    modified_df = modified_df.fillna(method='ffill')
    modified_df = modified_df.fillna(method='bfill')
            
    return modified_df


def Tag_StockExchange(stock_code):
    """

    目标：
        构建一个函数，传入股票代码后其可以识别出其所属的交易所;

    实现原理:
        如果股票代码前2位数是60则判断为A股-上交所-主板,
        如果股票代码前3位数是000或001则判断为A股-深交所-主板,
        如果股票代码前3位数是002或003则判断为A股-深交所-中小板,
        如果股票代码前2位数是30则判断为A股-深交所-创业板,
        如果股票代码前2位数是68则判断为A股-上交所-科创板,
        如果股票代码首位数是4或8，则判断为A股-北交所-新三板精选层,
    
    ### 感觉这个函数没多大用处

    Parameters
    ----------
    stock_code : str
        纯净的6位数股票代码;

    Returns
    -------
    stockexchange_dict : dict
        包含股票所属市场交易所信息的字典,
        第一个键为股票代码,值为输入的股票代码,
        第二个键为股票所属市场,值为A股,
        第三个键为股票所属交易所,值为上交所/深交所/北交所,
        第四个键为股票所属板块,值为主板/中小板/创业板/科创板/新三d'f'g'n'j板精选层

    """
    # 构建字典用于记录股票信息
    stockexchange_dict = {'股票代码':stock_code,
                          '股票所属市场':'A股',
                          '股票所属交易所':'',
                          '股票所属板块':''}
    
    
    # 获取股票代码的前三位用于判断该股票属于哪个市场
    flag = stock_code[:3]
    
    try:
        # 通过分支判断该股票属于哪一市场            
        if flag[0:-1] == '60':
            stockexchange_dict['股票所属板块'] = '主板'        
            stockexchange_dict['股票所属交易所'] = '上交所'
            
        elif flag == '000' or flag == '001':
            stockexchange_dict['股票所属板块'] = '主板'        
            stockexchange_dict['股票所属交易所'] = '深交所'
            
        elif flag == '002' or flag == '003':
            stockexchange_dict['股票所属板块'] = '中小板'        
            stockexchange_dict['股票所属交易所'] = '深交所'
            
        elif flag[0:-1] == '30' :
            stockexchange_dict['股票所属板块'] = '创业板'        
            stockexchange_dict['股票所属交易所'] = '深交所'
            
        elif flag[0:-1] == '68' :
            stockexchange_dict['股票所属板块'] = '科创板'        
            stockexchange_dict['股票所属交易所'] = '上交所'
            
        elif flag[0] == '4' or flag[0] == '8':
            stockexchange_dict['股票所属交易所'] = '北交所'            
            stockexchange_dict['股票所属板块'] = '新三板精选层'    
  
        return (stockexchange_dict)

              
    except:
        print('无法识别该股票所属交易所类别')
 


# 将从wind得到的数据转化为常用的dataframe格式
def Change_WindMinStockData_to_DF(path):
    '''
    该函数存在将mat格式转换为df的步骤，适用性不强
    
    应该改成最后返回哪些column可以自己选择的模式

    Parameters
    ----------
    path : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    '''
    # 数据提取
    bas_dict = m4p.loadmat(path)
    bas_df = pd.DataFrame(bas_dict)
    bas_df['stock_code'] = re.findall('\d{6}', path)[0]
    tag_list = ['day','time','open','high','low','close','volume','turnover']
    for i in range(len(tag_list)):
        bas_df[tag_list[i]] = bas_df['datamat'].str[i]
    bas_df.drop('datamat',axis=1,inplace=True)
    
    # 数据格式修改
    # 修改日期
    def turndate(date):
        date = date[0:4]+'-'+date[4:6]+'-'+date[6:8]
        return date
    bas_df['day'] = bas_df['day'].astype(str).apply(turndate)
    
    # 修改时间戳
    def turntime(time):
        if len(time)==5:
            time = '0{}:'.format(time[0])+'{}:'.format(time[1:3])+'00'
        elif len(time)==6:
            time = '{}:'.format(time[0:2])+'{}:'.format(time[2:4])+'00'
        return time
    bas_df['time'] = bas_df['time'].astype(str).apply(turntime)
    
    # # 创建beg_time 和 end_time列
    # bas_df['end_time'] = bas_df['time']
    # bas_df['beg_time'] = bas_df['time'].shift(1)
    # index = bas_df[bas_df['beg_time']=='15:00:00'].index
    # bas_df.loc[index,'beg_time'] = '09:30:00'
    # bas_df['beg_time'].fillna('09:30:00',inplace=True)
    return bas_df
 

# 根据交易日期在df中新增year-month列   
def Tag_TradeYAM(df,time_tag='TradingDate'):
    year = df[time_tag].str.split('-').str[0]
    month = df[time_tag].str.split('-').str[1]
    df['YearAndMonth'] = year+'-'+month
    return df


# file_path = r'D:\个人项目\Bollerslve(2020)\基础数据\StockDayData.csv'
# usecols = ['TradingDate', 'Symbol', 'CumulateBwardFactor',
#            'Listdt',  'Clsprc', 'Markettype', 'Trdsta', 
#            'ChangeRatio', 'PE1A','PBV1A', 'MarketValueA1', 'MarketValue']
# 读取本地数据
def Fetch_LocalStockDayData(file_path,
                            usecols,):
    
    df = pd.read_csv(file_path,usecols=usecols)
    
    return df


def decode_bytes(byte_value):
    if isinstance(byte_value, bytes):
        return byte_value.decode('utf-8')  # Assuming 'utf-8' encoding
    else:
        return byte_value  # Return non-bytes values as is

 
'''
1.给出一个文件夹路径，识别该文件夹里所有的py文件以及子文件夹中的py文件，并做出一一对应
2.识别每一个py文件内部，使用了那些py文件内部的函数/类，并标出对应的名称且一一对应
3.当函数名/类名发生变化后，自动在原来使用了该函数/类的py文件内进行修改
'''   

# 获取给定股票锐思1min数据
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



###############################################################################
## resset high frequancy data 
###############################################################################

####################################
## transfrom .sas7bdat into .csv
####################################

# Convert minutes to time format
def convert_to_time(stdtime):
    raw_minutes = (stdtime - 32460) / 60
    hour, minute = divmod(raw_minutes, 60)
    return "{:02d}:{:02d}:00".format(int(hour + 9), int(minute))


def Tsm_HFdata_worker(file, save_data_dir, skip_list):
    
    
    # only transform index or stock data
    file_tag = file.split('\\')[-1][:3]
    file_tag_dict = {'stk':'stock','ind':'index','fun':'fund','bon':'bond'}
    stkcd = re.findall('\d{6}', file)[0]
    min_ = file.split('_')[-1].split('.')[0]
    
    # only save stk or save both stk and index
    # if  file_len in [stk_len, ind_len]:
    if len(file.split('\\')[-1].split('_'))==3 and (file_tag in ['stk','ind','fun','bon']) and int(min_)==1:

        file_name = save_data_dir + '\\{}\\{}.csv'.format(file_tag_dict[file_tag],stkcd)
            
        if file_name not in skip_list:

        # if something wrong happened in the transfrom process, save those file_path
            try:
                hf_data = pd.read_sas(file)
                hf_data['stkcd'] = hf_data.Code.apply(lambda x:x.decode('utf-8'))
                hf_data['time'] = hf_data.Stdtime.apply(convert_to_time)
                hf_data['date'] = hf_data.Qdate.apply(lambda x:str(x.date()))
                
                hf_data = hf_data.rename(columns={'Begpr':'open','Highpr':'high','Lowpr':'low','TPrice':'close',
                                                  'TVolume_accu1':'volume','TSum_accu1':'turnover'})
                hf_data = hf_data[['stkcd','date','time','open','high','low','close','volume','turnover']]    
                
                
                hf_data.to_csv(file_name, index=False)
                print('{} finished conversion'.format(file))
           
            except:
                print('something wrong occur when transform {}'.format(file))
                return file
        
        
def Tsm_HFdata_older_worker(file):
    '''
    
    file = 'E:\HF2005M\shy2005m01_1.sas7bdat'
    
    '''
    

    # only transform index or stock data
    file_tag = file.split('\\')[-1].split('_')[-1].split('.')[0]
    year = re.findall('\d{4}', file)[0]
    
    # only save stk or save both stk and index
    # if  file_len in [stk_len, ind_len]:
    if int(file_tag) == 1:

        # if something wrong happened in the transfrom process, save those file_path
        try:
            hf_data = pd.read_sas(file)
            hf_data['stkcd'] = hf_data.Code.apply(lambda x:x.decode('utf-8'))
            hf_data['Class'] = hf_data.Class.apply(decode_bytes)
            hf_data = hf_data[~hf_data.Class.isin([np.nan])]
            
            hf_data['time'] = hf_data.Stdtime.apply(convert_to_time)
            hf_data['date'] = hf_data.Qdate.apply(lambda x:str(x.date()))
            
            hf_data = hf_data.rename(columns={'Begpr':'open','Highpr':'high','Lowpr':'low','TPrice':'close',
                                              'TVolume_accu':'volume','TSum_accu':'turnover'})
            hf_data = hf_data[['Class','stkcd','date','time','open','high','low','close','volume','turnover']]    
            hf_data.to_csv(r'F:\HF_MIN\Resset\Csv_data\{}\base\{}.csv'.format(year, file.split('\\')[-1]))
            
            # for group,df in hf_data.groupby(['Class','stkcd']):
            #     file_tag, stkcd = group
            
            #     df = df.drop('Class', axis=1)
            #     if file_tag == 'Stk':
            #         df.to_csv(save_data_dir + '\\stock\\{}.csv'.format(stkcd), index=False)
            #     elif file_tag == 'Indx':
            #         df.to_csv(save_data_dir + '\\index\\{}.csv'.format(stkcd), index=False)
            #     elif file_tag == 'Fund':
            #         df.to_csv(save_data_dir + '\\fund\\{}.csv'.format(stkcd), index=False)
            #     elif file_tag == 'Bond':
            #         df.to_csv(save_data_dir + '\\bond\\{}.csv'.format(stkcd), index=False)
            
            print('{} finished conversion'.format(file))
       
        except:
            print('something wrong occur when transform {}'.format(file))
            return file



def Tsm_HFdata_Year(year):
    raw_data_dir = 'E:\\HF{}M'.format(year)
    save_data_dir = 'F:\\HF_MIN\\Resset\\Csv_data\\{}'.format(year)
    file_list = Tag_FilePath_FromDir(raw_data_dir, suffix='sas7bdat')
    
    # num_processes = 16
    # # Create a Pool of processes
    # with Pool(num_processes) as pool:
        
    #     results = [pool.apply_async(Tsm_HFdata_older_worker, (file,)) for file in file_list]

    #     # Close the pool and wait for all processes to finish
    #     pool.close()
    #     pool.join()
    
    [Tsm_HFdata_older_worker(file) for file in file_list]
    
    
    file_list = Tag_FilePath_FromDir(r'F:\HF_MIN\Resset\Csv_data\{}\base'.format(year), suffix='csv')
    file_tag_dict = {'Stk':'stock','Indx':'index','Fund':'fund','Bond':'bond',
                     'Repo':'Repo','Wrnt':'Wrnt'}
    
    
    df_all = pd.DataFrame()
    for file in file_list:
        df = pd.read_csv(file,index_col=0)
        df_all = pd.concat([df_all, df])
        
    skip_list = Tag_FilePath_FromDir(save_data_dir)
    for group,df in df_all.groupby(['Class','stkcd']):
        file_tag, stkcd = group
        df = df.drop('Class', axis=1)
        try:

            file_name = save_data_dir + '\\{}\\{}.csv'.format(file_tag_dict[file_tag], stkcd)
            if file_name not in skip_list:
                df.to_csv(file_name, index=False)
        except:
            pass
            
        # if file_tag == 'Stk':
        #     file_name = 
                
        # elif file_tag == 'Indx':
        #     df.to_csv(save_data_dir + '\\index\\{}.csv'.format(stkcd), index=False)
        # elif file_tag == 'Fund':
        #     df.to_csv(save_data_dir + '\\fund\\{}.csv'.format(stkcd), index=False)
        # elif file_tag == 'Bond':
        #     df.to_csv(save_data_dir + '\\bond\\{}.csv'.format(stkcd), index=False)
    
    print('{} finished conversion'.format(file))
        

    # Get the results from the processes
    # erro_path_list = [result.get() for result in results if result.get() is not None]
    # pd.DataFrame(erro_path_list).to_csv(save_data_dir + '\\erro_path.csv')


        
def Tsm_Resset_HFdata_into_csv(raw_data_dir, save_data_dir):
        
    file_list = Tag_FilePath_FromDir(raw_data_dir, suffix='sas7bdat')
    skip_list = Tag_FilePath_FromDir(save_data_dir, suffix='csv')


    num_processes = 16
    # Create a Pool of processes
    with Pool(num_processes) as pool:
        
        results = [pool.apply_async(Tsm_HFdata_worker, (file, save_data_dir, skip_list, )) for file in file_list]

        # Close the pool and wait for all processes to finish
        pool.close()
        pool.join()
        
    # Get the results from the processes
    erro_path_list = [result.get() for result in results if result.get() is not None]
    pd.DataFrame(erro_path_list).to_csv(save_data_dir + '\\erro_path.csv')



def Tsm_Resset_HFdata_into_csv_older(raw_data_dir, save_data_dir):
        
    file_list = Tag_FilePath_FromDir(raw_data_dir, suffix='sas7bdat')

    num_processes = 16
    # Create a Pool of processes
    with Pool(num_processes) as pool:
        
        results = [pool.apply_async(Tsm_HFdata_older_worker, (file, save_data_dir,)) for file in file_list]

        # Close the pool and wait for all processes to finish
        pool.close()
        pool.join()
        
    # Get the results from the processes
    erro_path_list = [result.get() for result in results if result.get() is not None]
    pd.DataFrame(erro_path_list).to_csv(save_data_dir + '\\erro_path.csv')



if __name__ == '__main__':
    
    # creat dir to store csv data
    folders_dict = {str(year):{type_:'' for type_ in ['stock', 'index', 'fund', 'bond','base']} for year in range(2005,2012)}
    # create_folders(r'F:\HF_MIN\Resset\Csv_data', 
    #                 folders_dict)

    # define a func that transfrom stocks and index .sas7bdat into .csv 
    # when given a specific original data dir path and the saved data dir path
    
    # for year in [2020]:
    #     raw_data_dir = 'E:\\RESSET\\HF{}M'.format(year)
    #     save_data_dir = 'F:\\HF_MIN\\Resset\\Csv_data\\{}'.format(year)

    #     Tsm_Resset_HFdata_into_csv(raw_data_dir, save_data_dir)
    # for year in [2011]: # 
    #     Tsm_HFdata_Year(year)
    
    # # retry to transform erro file
    # erro_file_list = pd.read_csv(save_data_dir + '\\erro_path.csv', index_col=0)
    # erro_file_twice = [Tsm_HFdata_worker(file, save_data_dir) for file in  ]
    
    # Crt_Merge

