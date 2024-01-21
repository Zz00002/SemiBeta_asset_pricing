# SemiBeta_asset_pricing

## 基础数据构成

在stock文件夹中，我存储了9只股票2005年的1分钟高频数据，在index文件夹中，我储存了沪深300指数的1分钟高频数据以及我自己根据文章中所提到的构建方法所创造的全A综指5分钟高频数据（将上证指数和深证成指市值加权构建而成），用以作为代码运行范例的基础数据，高频数据全部来自RESSET。此外，我还使用来自CSMAR的日/月频数据构建了三个基础的数据csv，由于这部分数据量过大，且代码耦合度较高，因此展示不放在此进行展示，有需要可以联系联系我的邮箱: HaojianZhang002@gmail.com，与之相对的，我会在此说明这三个csv都是如何构建得到的：

* CH3_daily.csv
  * 直接从https://finance.wharton.upenn.edu/~stambaug/网站获得

* BM.csv

  ```python
  using_data['BM'] = using_data['NetAsset_s1']/using_data['Msmvttl_s1']/1000
  
  ```

  * BM即是账面市值比，参照ff3因子模型的构建方式，使用股票的净资产除以股票的总市值。

* SAVIC_saveMV.csv

* SAVIC_saveMV_day.csv

  ```python
  # I. Identify Stock's belong to which factor's group
  # get stocks' month trade data
  Data_Source = self._CSMAR
  stock_mon_trade_data = Data_Source.StockMonTradeData_df.copy()
  
  # get stocks' last month MV data of each year
  stock_mon_trade_data['Trdyr'] = stock_mon_trade_data['Trdmnt'].str.extract(
      '(\d{4})')
  
  # Do the data cleaning process
  # 1. drop the data that don't meet the trade day number requirement
  # 2. select specific ExchangeTradBoard data
  # 3. drop stocks that have been tagged ST
  # 4. drop stocks whose listdt less than 6 months
  # 5. drop stocks whose have abnormal ret (mostly due to suspension)
  stock_mon_trade_data = self.Select_Stock_ExchangeTradBoard(
      df=stock_mon_trade_data, board_list=[1, 4, 16, 32])
  stock_mon_trade_data = self.Drop_Stock_STData(df=stock_mon_trade_data)
  stock_mon_trade_data = self.Drop_Stock_InadeListdtData(
      df=stock_mon_trade_data, drop_Mon_num=6)
  stock_mon_trade_data = self.Drop_Stock_InadeTradData_(
      df=stock_mon_trade_data)
  stock_mon_trade_data = self.Drop_Stock_SuspOverRet(
      df=stock_mon_trade_data)
  
  self.StockData_SVIC = stock_mon_trade_data.copy()
  self.SAVIC_saveMV_day = pd.merge(Data_Source.StockDayTradeData_df, stock_mon_trade_data)
  
  ```

  * 参照Size and Value in China的方法对日频数据进行清洗，不执行其删去市值小于截面30%的操作。



