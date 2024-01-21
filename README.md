# SemiBeta_asset_pricing

## Basic Data Composition

In the stock folder, I have stored 1-minute high-frequency data of 9 stocks from the year 2005. In the index folder, I have stored 1-minute high-frequency data of the CSI 300 index and the 5-minute high-frequency data of the all-A comprehensive index (constructed by market value weighting of the Shanghai and Shenzhen Composite Indices), which I created based on the construction methods mentioned in the articles. These serve as the basic data for the example runs of the code, and all high-frequency data come from RESSET. In addition, I used daily/monthly frequency data from CSMAR to construct three basic data CSVs. Due to the large volume of this data and the high degree of code coupling, this part is not displayed here. If needed, you can contact me at my email: HaojianZhang002@gmail.com. Conversely, I will explain here how these three CSVs were constructed:

* CH3_daily.csv
  * Directly obtained from the website https://finance.wharton.upenn.edu/~stambaug/

* BM.csv

  ```python
  using_data['BM'] = using_data['NetAsset_s1']/using_data['Msmvttl_s1']/1000
  
  ```

  * BM refers to the book-to-market ratio, following the construction method of the FF3 factor model, using the net assets of a stock divided by the total market value of the stock.

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

  * Following the method of Size and Value in China for daily frequency data cleaning, without performing the operation of deleting stocks with market values less than 30% of the cross-section.

## Code Execution Results

Executing main.py will automatically create corresponding folders, estimate key and control variables, and automatically run empirical results (except for Fama-Macbeth regression results, which cannot be successfully run due to too few sample stocks, but the corresponding code is included in the files). Note that third-party libraries must match.



