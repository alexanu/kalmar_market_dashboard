from alpaca.trading.enums import OrderSide, TimeInForce, AssetClass, AssetStatus, AssetExchange, OrderStatus, QueryOrderStatus, CorporateActionType, CorporateActionSubType
from alpaca.trading.requests import GetCalendarRequest, GetAssetsRequest, GetOrdersRequest, MarketOrderRequest, LimitOrderRequest, StopLossRequest, TrailingStopOrderRequest, GetPortfolioHistoryRequest, GetCorporateAnnouncementsRequest
from alpaca.data.requests import StockLatestQuoteRequest, StockTradesRequest, StockQuotesRequest, StockBarsRequest, StockSnapshotRequest, StockLatestBarRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from alpaca.data.enums import Adjustment, DataFeed, Exchange

from Azure_config import *

from github import Github
from urllib.request import urlopen
import json
import pandas as pd
import numpy as np
import datetime as dt
import concurrent.futures
import requests


from alpaca.trading.client import TradingClient
from alpaca.data import StockHistoricalDataClient
trading_client = TradingClient(ALPACA_API_KEY_PAPER, ALPACA_API_SECRET_PAPER) # dir(trading_client)
stock_client = StockHistoricalDataClient(ALPACA_API_KEY_PAPER, ALPACA_API_SECRET_PAPER)




base_fmp_url = 'https://financialmodelingprep.com/api/'

Sector_Dict = {'Basic Materials': 'Mater',
                'Consumer Cyclicals': 'Cons Cy', 
                'Consumer Non-Cyclicals': "Cons NC", 
                'Energy': "Energy", 
                'Financials': "Fin", 
                'Healthcare': "Health",
                'Real Estate': "RE",
                'Technology': "Tech",
                'Utilities': "Util",                
                'Industrials': 'Indu'}

Group_Dict = {'Basic Materials': 'Chemi, Constr Mat, Packag, Metals & Min',
                'Consumer Cyclicals': 'Auto, Retail, Homebuild, Hotel, Enternt, HH Goods, Media, Textile', 
                'Consumer Non-Cyclicals': "Beverag, Food, Drug, Tobacco", 
                'Energy': "Oil, Gas, Equip, Renew", 
                'Financials': "Banks, Insur, IB", 
                'Healthcare': "Biotech, Research, Equip, Providers, Pharma",
                'Real Estate': "RE",
                'Technology': "Communic, Computer, Electronics, FinTech, Semicond, SW",
                'Utilities': "Electric, NatGas, Water",                
                'Industrials': 'Aero, Defense, Constru, Freight, Logi, Machinery, Transport'}


def get_filtered_avg_score(df):
    avg_score_filtered_df = df[(df['AvgScore'] > df['AvgScore1W']) & (df['AvgScore1W'] > df['AvgScore1M']) & (df['AvgScore'] > df['AvgScore1M']) & (df['AvgScore'] > 7)]
    avg_score_filtered_df = avg_score_filtered_df[avg_score_filtered_df['Trades_dec_1isSmall']>1] # filter out passively traded
    avg_score_filtered_df = avg_score_filtered_df[avg_score_filtered_df['Insiders']>20] # filter out with negative insiders passively traded
    avg_score_filtered_df = avg_score_filtered_df[avg_score_filtered_df['Earn_Quality_1YChg']>0] # filter out with negative yoy change in earnings quality
    avg_score_filtered_df = avg_score_filtered_df[avg_score_filtered_df['ShortSqueeze']>20] # filter out with small potential to go up
    avg_score_filtered_df = avg_score_filtered_df[(avg_score_filtered_df['MONTH_vs_AVG_decil_1isSmall']>3) & (avg_score_filtered_df['WEEK_vs_AVG_decil_1isSmall']>3)] # filter out with negative price momentum
    return avg_score_filtered_df

def get_all_alpaca_stocks(trading_client):
    assets = trading_client.get_all_assets(GetAssetsRequest(asset_class=AssetClass.US_EQUITY,status= AssetStatus.ACTIVE))
    exclude_strings = ['Etf', 'ETF', 'Lp', 'L.P', 'Fund', 'Trust', 'Depositary', 'Depository', 'Note', 'Reit', 'REIT']
    assets_in_scope = [asset.symbol for asset in assets
                        if asset.exchange != 'OTC' # OTC stocks play by different rules than Exchange Traded stocks (often referred to as NMS). 
                        and asset.shortable
                        and asset.tradable
                        and asset.marginable # if a stock is not marginable that means it cannot be used as collateral for margin. 
                        and asset.fractionable # indirectly filters out a lot of small volatile stocks:  
                        and asset.easy_to_borrow 
                        and asset.maintenance_margin_requirement == 30
                        and not (any(ex_string in asset.name for ex_string in exclude_strings))]
    return assets_in_scope

def get_ETFs():
    us_etfs=pd.read_excel(ETFDB_path,sheet_name='US_ETFs',skiprows=2,header=0,usecols=ETFDB_columns)
    us_etfs = us_etfs[us_etfs['Scope']=="Yes"]
    us_etfs['Fund Name Short'] = us_etfs['Fund Name'] + " (" + us_etfs['symbol'] + ")"
    delete_words = [" ETF", " Fund;ETF",  " Fd;ETF"," Index", " Idx", " Fund", " UCITS", "   ", "  "]
    for word in delete_words:
        us_etfs['Fund Name Short'] = us_etfs['Fund Name Short'].str.replace(word, '', regex=False)

    us_etfs.to_csv(gh_csv_ETFDB, index=False)
    with open(gh_csv_ETFDB, encoding='utf8') as file:
        content = file.read()
    g = Github(github_strat_token)
    repo = g.get_user().get_repo(dedicated_repo)
    try:
        contents = repo.get_contents(gh_csv_ETFDB)
        repo.update_file(contents.path, "committing files", content, contents.sha, branch="main")
        print(gh_csv_ETFDB + ' UPDATED')
    except:
        repo.create_file(gh_csv_ETFDB, "committing files", content, branch="main")
        print(gh_csv_ETFDB + ' CREATED')


def get_macro_events_from_GH():
    repository = Github(github_strat_token).get_user().get_repo(dedicated_repo)
    macro_events_file = repository.get_contents(gh_csv_macro_events)
    macro_events_df = pd.read_csv(io.StringIO(macro_events_file.decoded_content.decode()))
    today_date = dt.date.today().strftime("%Y-%m-%d")
    macro_events_df['Event_Datetime_CET'] = np.where(macro_events_df['GMT Date'] >= today_date, macro_events_df['GMT Date'], macro_events_df['Next'])
    macro_events_df['Event_Datetime_CET'] = pd.to_datetime(macro_events_df['Event_Datetime_CET']+ macro_events_df['GMT Time'],format='%Y-%m-%d%H:%M:%S')
    macro_events_df['Event_Datetime_CET'] = macro_events_df['Event_Datetime_CET'].dt.tz_localize('UCT').dt.tz_convert('CET').dt.tz_localize(None)
    macro_events_df.sort_values(by='Event_Datetime_CET',inplace=True)
    macro_events_df.reset_index(inplace=True)
    macro_events_df['Event_Datetime_CET'] = macro_events_df['Event_Datetime_CET'].dt.strftime('%a, %d %b, %H:%M')
    macro_events_df = macro_events_df[['Indicator Type','Short Name','Event_Datetime_CET']]
    today_macro = len(macro_events_df[macro_events_df['Event_Datetime_CET'] == today_date])
    # print(f'Read and processed {len(macro_events_df)} of macro events from GH')
    return macro_events_df

def get_macro_events_from_fmp():
    try:
        print('Starting reading macro events from FMP ... ')
        today = dt.datetime.now().strftime("%Y-%m-%d")
        n_days_future = (dt.datetime.now() + dt.timedelta(days=10)).strftime("%Y-%m-%d")
        specific_url = f'v3/economic_calendar?from={today}&to={n_days_future}&apikey={fmp_api_key}'
        response = urlopen(base_fmp_url+specific_url)
        data = response.read().decode("utf-8")
        econ_events = pd.json_normalize(json.loads(data))
        econ_events = econ_events[(econ_events['impact'] == 'High')].reset_index(drop=True)
        econ_events['date'] = pd.to_datetime(econ_events['date'])
        econ_events['date'] = econ_events['date'].dt.tz_localize('UTC')
        econ_events['munich_time'] = econ_events['date'].dt.tz_convert('Europe/Berlin')
        econ_events['munich_time'] = econ_events['munich_time'].dt.tz_localize(None)
        current_time = dt.datetime.now()
        econ_events = econ_events[econ_events['munich_time'] > current_time]
        econ_events = econ_events[['munich_time','country', 'event']]
        econ_events.sort_values(by='munich_time',inplace=True)
        today_macro = len(econ_events[econ_events['munich_time'].dt.date == dt.date.today()])
        econ_events['munich_time'] = econ_events['munich_time'].dt.strftime('%a, %d %b, %H:%M')
        print(f'Done: {len(econ_events)} were collected.')
        return econ_events, today_macro
    except Exception as e:
        print(f"Error while fetching data from API: {e}")
        return pd.DataFrame(), 0  # Return an empty DataFrame on error

def get_earnings_from_gh(sp500_df):
    print('Transforming earnings data from refinitv SP500 file')
    today_date = dt.datetime.now().date()
    next_5d = today_date +  pd.Timedelta('5D')
    earnings_df = sp500_df.copy()
    earnings_df = earnings_df[earnings_df['EPresi_date'].notna()]
    earnings_df = earnings_df[pd.to_datetime(earnings_df['EPresi_date']).dt.date >= today_date]
    earnings_df = earnings_df[pd.to_datetime(earnings_df['EPresi_date']).dt.date < next_5d]
    earnings_df = earnings_df.sort_values(by='EPresi_date').reset_index(drop=True)
    today_earnings = len(earnings_df[pd.to_datetime(earnings_df['EPresi_date']).dt.date == today_date])
    earnings_df['ERel_date'] = pd.to_datetime(earnings_df['ERel_date']).dt.strftime('%a, %d %b, %H:%M')
    earnings_df['EPresi_date'] = pd.to_datetime(earnings_df['EPresi_date']).dt.strftime('%a, %d %b, %H:%M')
    return earnings_df, today_earnings

def get_earnings_from_fmp():
    today = dt.datetime.now().strftime("%Y-%m-%d")
    n_days_future = (dt.datetime.now() + dt.timedelta(days=50)).strftime("%Y-%m-%d")
    specific_url = f'v4/earning-calendar-confirmed?from={today}&to={n_days_future}&apikey={fmp_api_key}'
    response = urlopen(base_fmp_url+specific_url)
    data = response.read().decode("utf-8")
    earn_calls = pd.json_normalize(json.loads(data))
    earn_calls['date'] = pd.to_datetime(earn_calls['date'], format='%Y-%m-%d')
    earn_calls['time US'] = pd.to_datetime(earn_calls['time'], format='%H:%M')
    earn_calls['datetime'] = pd.to_datetime(earn_calls['date'].dt.date.astype(str) + ' ' + earn_calls['time US'].dt.time.astype(str))
    earn_calls['datetime'] = earn_calls['datetime'].dt.tz_localize('US/Eastern')
    earn_calls['time MUC'] = earn_calls['datetime'].dt.tz_convert('Europe/Berlin')
    earn_calls.sort_values(by='time MUC',inplace=True)
    today_earnings = len(earn_calls[earn_calls['time MUC'].dt.date == dt.date.today()])
    earn_calls['time MUC'] = earn_calls['time MUC'].dt.strftime('%a, %d %b, %H:%M')
    # needed_cols = ['symbol', 'publicationDate','time MUC']
    needed_cols = ['symbol', 'time MUC']
    earn_calls = earn_calls[needed_cols]
    return earn_calls, today_earnings

def get_senator_trades_fmp(symbol_scope):

    def get_senators_discl_rss(page_num):
        specific_url = f'v4/senate-disclosure-rss-feed?page={page_num}&apikey={fmp_api_key}'
        response = urlopen(base_fmp_url+specific_url)
        data = response.read().decode("utf-8")
        data_df = pd.json_normalize(json.loads(data))
        return data_df

    fmp_senators_discl = pd.concat((get_senators_discl_rss(num) for num in [0,1])).reset_index()
    needed_cols = ['disclosureDate', 'transactionDate', 'owner','ticker', 'assetDescription', 'type', 'amount', 'representative']
    fmp_senators_discl = fmp_senators_discl[needed_cols]

    def get_senators_trade_rss(page_num):
        specific_url = f'v4/senate-trading-rss-feed?page={page_num}&apikey={fmp_api_key}'
        response = urlopen(base_fmp_url+specific_url)
        data = response.read().decode("utf-8")
        data_df = pd.json_normalize(json.loads(data))
        return data_df

    fmp_senators_trade = pd.concat((get_senators_trade_rss(num) for num in [0,1])).reset_index()
    fmp_senators_trade['representative'] = fmp_senators_trade['firstName'] + " " + fmp_senators_trade['lastName']
    fmp_senators_trade.rename(columns={'dateRecieved': 'disclosureDate', 'symbol': 'ticker',}, inplace=True)
    needed_cols = ['disclosureDate', 'transactionDate', 'owner','ticker', 'assetDescription', 'type', 'amount', 'representative','comment','assetType']
    fmp_senators_trade = fmp_senators_trade[needed_cols]

    fmp_senators_deals = pd.merge(fmp_senators_discl,fmp_senators_trade,how='outer')

    fmp_senators_deals = fmp_senators_deals[fmp_senators_deals.ticker.isin(symbol_scope)]

    fmp_senators_deals['type'] = fmp_senators_deals['type'].replace({
        "Sale (Full)": "sale_full", 
        "Sale (Partial)": "sale_partial",
        "Purchase": "purchase"
    })

    fmp_senators_deals['disclosureDate'] = pd.to_datetime(fmp_senators_deals['disclosureDate'], errors='coerce')
    fmp_senators_deals['transactionDate'] = pd.to_datetime(fmp_senators_deals['transactionDate'], errors='coerce')
    fmp_senators_deals = fmp_senators_deals.dropna(subset=['transactionDate','disclosureDate'])
    fmp_senators_deals['ReportingLag'] = (fmp_senators_deals['disclosureDate'] - fmp_senators_deals['transactionDate']).dt.days

    fmp_senators_deals['amount'] = fmp_senators_deals['amount'].str.replace('[\$, ]', '', regex=True)
    fmp_senators_deals[['From', 'To']] = fmp_senators_deals['amount'].str.split('-', expand=True)
    fmp_senators_deals['From'] = fmp_senators_deals['From'].astype(int)

    bins = [0, 100000, 250000, 500000, 1000000, np.inf]
    labels = ['small','normal', 'big', 'very big', 'huge']
    fmp_senators_deals['amount'] = pd.cut(fmp_senators_deals['From'], bins=bins, labels=labels)

    fmp_senators_deals.drop(['From', 'To'], axis=1, inplace=True)

   # filter out small transactions & transactions which were reported much later than occured
    n_days_ago = (dt.datetime.now() - dt.timedelta(days=30)).strftime("%Y-%m-%d")
    not_small_transcations = fmp_senators_deals[(fmp_senators_deals['amount'] != 'small') & 
                                                (fmp_senators_deals['ReportingLag']<30) &
                                                (fmp_senators_deals['disclosureDate'] > n_days_ago)].reset_index(drop=True)

    return not_small_transcations

clock = trading_client.get_clock()
today = clock.timestamp
previous_day = today - pd.Timedelta('1D')
previous_day_40 = today - pd.Timedelta('40D')

NUM_STOCK_THREADS = 3
NUM_ETF_THREADS = 10


def chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def fetch_data_for_chunk(symbols_chunk):
    bars_request_params = StockBarsRequest(
        symbol_or_symbols=symbols_chunk,
        start=previous_day_40,
        end=previous_day,
        timeframe=TimeFrame.Day,
        adjustment=Adjustment.ALL,
        feed=DataFeed.SIP
    )
    return stock_client.get_stock_bars(bars_request_params).df.reset_index()


def download_data_daily(symbols, daily_rets = False):
    SYMBOLS_PER_CHUNK = (len(symbols) // NUM_STOCK_THREADS)+10
    symbols_chunks = list(chunks(symbols, SYMBOLS_PER_CHUNK))

    dataframes = []

    print(f'Starting concurrent daily data download for last 40 days across {len(symbols)} tickers ...')
    with concurrent.futures.ThreadPoolExecutor(max_workers=NUM_STOCK_THREADS) as executor:
        for df in executor.map(fetch_data_for_chunk, symbols_chunks):
            dataframes.append(df)

    daily_df = pd.concat(dataframes, ignore_index=True)

    daily_df['days'] = (daily_df.timestamp - pd.Timestamp(previous_day)).dt.days
    daily_df.timestamp = daily_df.timestamp.dt.date

    print(f'Done downloading daily data. {len(daily_df)} rows collected.')

    # Initialize dictionary to store the current streak for each symbol
    streak_dict = {}

    print(f'Doing streaks calculation...')

    # Iterate through the DataFrame grouped by symbol
    for symbol, group in daily_df.groupby('symbol'):
        # Sort the group by timestamp
        group = group.sort_values('timestamp')
        
        # Initialize streak variables
        current_streak = 0
        previous_close = None
        
        # Iterate through the rows of the group
        for _, row in group.iterrows():
            close = row['close']
            
            # Skip the first row
            if previous_close is None:
                previous_close = close
                continue

            # Check the streak direction
            if close > previous_close:
                # Positive streak
                if current_streak >= 0:
                    current_streak += 1
                else:
                    current_streak = 1
            elif close < previous_close:
                # Negative streak
                if current_streak <= 0:
                    current_streak -= 1
                else:
                    current_streak = -1
            else:
                # No change
                current_streak = 0

            previous_close = close
        
        # Add the current streak to the dictionary
        streak_dict[symbol] = current_streak


    if daily_rets:
        daily_returns = daily_df[['symbol','timestamp','close']].copy()
        daily_returns['ret'] = daily_df.groupby("symbol")["close"].pct_change(1).fillna(0)
        return daily_returns
    else:
        print('Doing daily returns calculations ...')
        # HACK:This block is needed as there could be no -30 or -5 days because of wknds/holidays
        d0=0
        d5=-5
        d30=-30
        while len(daily_df[daily_df.days==d0])==0:
            d0 = d0 - 1
        while len(daily_df[daily_df.days==d5])==0:
            d5 = d5 - 1
        while len(daily_df[daily_df.days==d30])==0:
            d30 = d30 - 1
        returns_df = daily_df[daily_df.days.isin([d0,d5,d30])].copy()
        returns_df['chg30'] = round(returns_df['close'].pct_change(2)*100,1)
        returns_df['chg5'] = round(returns_df['close'].pct_change(1)*100,1)
        returns_df['chg1'] = round(100*(returns_df.close-returns_df.open)/returns_df.open,1)
        returns_df = returns_df[returns_df.days.isin([d0])]
        returns_df = returns_df[['symbol','chg30','chg5','chg1']].reset_index(drop=True)
        returns_df['chg30']=returns_df['chg30']-returns_df['chg5']
        returns_df['chg5']=returns_df['chg5']-returns_df['chg1']
        returns_df.dropna(inplace=True)

        returns_df['streak'] = returns_df['symbol'].map(streak_dict)
        returns_df['streak'].fillna(0, inplace=True)

        # adding overnight movements:
        print('Fetching overnight snapshots ...')
        snap = stock_client.get_stock_snapshot(StockSnapshotRequest(symbol_or_symbols=symbols, feed = DataFeed.SIP))
        snapshot_data = {stock: [
                                snapshot.latest_trade.timestamp,                        
                                snapshot.latest_trade.price, 
                                snapshot.daily_bar.timestamp,
                                snapshot.daily_bar.open,
                                snapshot.daily_bar.close,
                                snapshot.previous_daily_bar.timestamp,
                                snapshot.previous_daily_bar.close,
                                ]
                        for stock, snapshot in snap.items() if snapshot and snapshot.daily_bar and snapshot.previous_daily_bar
                        }
        snapshot_columns=['price time', 'price', 'today', 'today_open', 'today_close','yest', 'yest_close']
        snapshot_df = pd.DataFrame(snapshot_data.values(), snapshot_data.keys(), columns=snapshot_columns)

        snapshot_df['price time'] = snapshot_df['price time'].dt.tz_convert('America/New_York').dt.tz_localize(None) # convert from UTC to ET and remove +00:00 from datetime
        snapshot_df['price time short'] = snapshot_df['price time'].dt.strftime('%H:%M')
        snapshot_df['today'] = snapshot_df['today'].dt.tz_convert('America/New_York').dt.tz_localize(None)
        snapshot_df['yest'] = snapshot_df['yest'].dt.tz_convert('America/New_York').dt.tz_localize(None)
        snapshot_df = snapshot_df.reset_index().rename(columns={'index':'symbol'}) # needed for merger on symbol

        # adding overnight returns to daily
        returns_df = returns_df.merge(snapshot_df,on=['symbol'],how ='outer')
        
        current_time = snapshot_df['price time short'].max()
        current_time_dt = dt.datetime.strptime(current_time, '%H:%M').time()
        target_time_dt = dt.time(9, 30)


        print('Doing overnight returns calculations ...')
        if clock.is_open: # open market
            returns_df['DAY'] = round(100*(returns_df['price']-returns_df['today_open'])/returns_df['today_open'],1)
            returns_df['PRE'] = round(100*(returns_df['today_open']-returns_df['yest_close'])/returns_df['yest_close'],1)
            returns_df['POST'] = 0
            market_case = {'General':'US Market is open', 
                            'DAY': f'on {today.date().strftime("%d.%b")} from open (15:30 MUC) till now ({current_time} ET)' , 
                            'PRE': f'from close (22:00 MUC) on {snapshot_df["yest"][0].strftime("%d.%b")} till open (15:30 MUC) on {snapshot_df["today"][0].strftime("%d.%b")}', 
                            'POST': 'no data yet as the market is stil open'}
        elif returns_df['price time'].max().date() == today.date():
            if today.hour > 16:
                returns_df['PRE'] = round(100*(returns_df['today_open']-returns_df['yest_close'])/returns_df['yest_close'],1)
                returns_df['DAY'] = round(100*(returns_df['today_close']-returns_df['today_open'])/returns_df['today_open'],1)
                returns_df['POST'] = round(100*(returns_df['price']-returns_df['today_close'])/returns_df['today_close'],1)
                market_case = {'General':'Post-market', 
                               'DAY': f'on {today.date().strftime("%d.%b")} from 15:30 till 22:00 Munich time' , 
                               'PRE': f'from close (22 MUC) on {snapshot_df["yest"][0].strftime("%d.%b")} till open (15::30 MUC) on {snapshot_df["today"][0].strftime("%d.%b")}', 
                               'POST': f'on {today.date().strftime("%d.%b")} from 22 Munich time (16:00 ET) till now'}
            else: # pre-market
                returns_df['PRE'] = round(100*(returns_df['price']-returns_df['today_close'])/returns_df['today_close'],1) # this include also yesterday'S post, but what could I do...
                returns_df['DAY'] = 0
                returns_df['POST'] = 0
                diff = dt.datetime.combine(dt.datetime.today(), target_time_dt) - dt.datetime.combine(dt.datetime.today(), current_time_dt)
                hours_left_till_open = round(diff.total_seconds() / 3600, 1)
                market_case = {'General': f'pre-market: {hours_left_till_open} hours till open in US', 
                               'DAY': f'no data yet: {hours_left_till_open} hours till open in US' , 
                               'PRE': f'from close (22 MUC) on {snapshot_df["today"][0].strftime("%d.%b")} till now, i.e. includes yesterday POST. Still {hours_left_till_open} hours till open in US', 
                               'POST': f'no data yet. Yesterday POST is included in current PRE'}
        else: # weekends, holidays
            returns_df['PRE'] = 0
            returns_df['DAY'] = 0
            returns_df['POST'] = round(100*(returns_df['price']-returns_df['today_close'])/returns_df['today_close'],1)
            market_case = {'General': f'no trading: weekend or holiday in US', 
                            'DAY': f'no trading' , 
                            'PRE': f'no trading', 
                            'POST': f'POST return for latest available trading date, which was {snapshot_df["today"][0].strftime("%d.%b")}'}

        returns_df = returns_df[['symbol','chg30','chg5','chg1','streak','PRE','DAY','POST']].reset_index(drop=True)

        return returns_df, market_case


def download_etf_daily_data(symbols):

    SYMBOLS_PER_CHUNK = (len(symbols) // NUM_ETF_THREADS)+10
    symbols_chunks = list(chunks(symbols, SYMBOLS_PER_CHUNK))

    dataframes = []

    print(f'Starting concurrent daily data download for last 40 days across {len(symbols)} ETFs ...')

    with concurrent.futures.ThreadPoolExecutor(max_workers=NUM_ETF_THREADS) as executor:
        for df in executor.map(fetch_data_for_chunk, symbols_chunks):
            dataframes.append(df)

    daily_df = pd.concat(dataframes, ignore_index=True)

    daily_df['days'] = (daily_df.timestamp - pd.Timestamp(previous_day)).dt.days
    daily_df.timestamp = daily_df.timestamp.dt.date

    print(f'Done downloading daily data for ETFs. {len(daily_df)} rows collected.')

    print('Doing daily returns calculations ...')

    # HACK:This block is needed as there could be no -30 or -5 days because of wknds/holidays
    d0=0
    d5=-5
    d30=-30
    while len(daily_df[daily_df.days==d0])==0:
        d0 = d0 - 1
    while len(daily_df[daily_df.days==d5])==0:
        d5 = d5 - 1
    while len(daily_df[daily_df.days==d30])==0:
        d30 = d30 - 1
    returns_df = daily_df[daily_df.days.isin([d0,d5,d30])].copy()
    returns_df['chg30'] = round(returns_df['close'].pct_change(2)*100,1)
    returns_df['chg5'] = round(returns_df['close'].pct_change(1)*100,1)
    returns_df['chg1'] = round(100*(returns_df.close-returns_df.open)/returns_df.open,1)
    returns_df = returns_df[returns_df.days.isin([d0])]
    returns_df = returns_df[['symbol','chg30','chg5','chg1']].reset_index(drop=True)
    returns_df['chg30']=returns_df['chg30']-returns_df['chg5']
    returns_df['chg5']=returns_df['chg5']-returns_df['chg1']


    returns_df = returns_df[['symbol','chg30','chg5','chg1']].reset_index(drop=True)

    return returns_df


Curr_Dict = {'EURUSD': 'USA',
            'EURJPY': 'JP', 
            'EURGBP': 'UK', 
            'EURCHF': 'CHF', 
            'EURCAD': 'CAD', 
            'EURAUD': 'AUD', 
            'EURCNY': 'China', 
            'EURHKD': 'HK', 
            'EURINR': 'India', 
            'EURZAR': 'ZAR', 
            'EURRUB': 'RUB', 
            'EURTRY': 'Turk'}

def download_fx_daily_data_fmp():

    # https://fmpcloud.io/api/v3/historical-price-full/JPYUSD?timeseries=5&apikey=d0e821d6fc75c551faef9d5c495136cc

    # fmp api allows only up to 5 symbols in 1 go
    keys_list = list(Curr_Dict.keys())
    n = 5  # Maximum number of keys in each sublist
    sublists = [keys_list[i:i + n] for i in range(0, len(keys_list), n)]
    sublists_str = [','.join(sublist) for sublist in sublists]
    last_days = 50

    def get_last_days_fx(cross):
        specific_url = f'v3/historical-price-full/{cross}?timeseries={last_days}&apikey={fmp_api_key}'
        response = urlopen(base_fmp_url+specific_url)
        data = response.read().decode("utf-8")
        data_df = pd.json_normalize(json.loads(data))
        data_df = pd.json_normalize(json.loads(data)['historicalStockList'])
        return data_df

    last_fx_df = pd.concat((get_last_days_fx(lst) for lst in sublists_str)).reset_index()
    last_fx_df = last_fx_df.explode('historical')
    last_fx_df = pd.concat([last_fx_df.drop(['historical'], axis=1), last_fx_df['historical'].apply(pd.Series)], axis=1)
    fx_df = last_fx_df[['symbol','date','close']].copy()
    fx_df['date'] = pd.to_datetime(fx_df['date'])
    latest_date = fx_df['date'].max()
    fx_df['days'] = (latest_date - fx_df['date']).dt.days
    fx_df['days'] = fx_df['days'].replace({1: 0, 2: 0, 6: 5, 7: 5, 8: 5, 31: 30, 32: 30, 33: 30}) # to avoid some dates are missing
    fx_df = fx_df.drop_duplicates(subset=['symbol', 'days'], keep='first')
    returns_df = fx_df[fx_df.days.isin([0,5,30])].copy()
    returns_df['chg30'] = round(returns_df['close'].pct_change(-2)*100,1)
    returns_df['chg5'] = round(returns_df['close'].pct_change(-1)*100,1)
    returns_df = returns_df[returns_df.days.isin([0])]
    returns_df = returns_df[['symbol','chg30','chg5']].reset_index(drop=True)
    returns_df['chg30']=returns_df['chg30']-returns_df['chg5']
    returns_df['Desc'] = returns_df['symbol'].map(Curr_Dict) # Mapping code to text

    return returns_df


# these 2 lists are created taking the data from Riksbank API
# Available series: https://www.riksbank.se/en-gb/statistics/search-interest--exchange-rates/web-services/series-for-web-services/
# Info about yields: https://www.riksbank.se/en-gb/statistics/search-interest--exchange-rates/explanation-of-the-series/international-market-rates/
# API: https://developer.api-test.riksbank.se/api-details#api=swea-api&operation=get-observations-seriesid-from-to
Bonds5 ={
    'USGVB5Y': "US 5Y",
    'JPGVB5Y': "JP 5Y",
    'DEGVB5Y': "DE 5Y",
    'GBGVB5Y': "GB 5Y",
}

Bonds10 ={
    'USGVB10Y': "US 10Y",
    'JPGVB10Y': "JP 10Y",
    'DEGVB10Y': "DE 10Y",
    'GBGVB10Y': "GB 10Y",
}

def get_bond_data(years=1, bond_list = Bonds10):
    today = dt.datetime.today()
    to_date = today.strftime('%Y-%m-%d')
    from_date = (today - dt.timedelta(days=years*365)).strftime('%Y-%m-%d')  # 10 years ago

    Bonds_df = pd.DataFrame()
    print('Starting yields download from Riksbank')
    for seriesId, name in bond_list.items():
        try:
            # there is pretty restricitve api call method: that's why only 4 10Y yields
            url = f'https://api-test.riksbank.se/swea/v1/Observations/{seriesId}/{from_date}/{to_date}'
            response = requests.get(url)
            temp = pd.json_normalize(response.json()).assign(bond=name)
            Bonds_df = pd.concat([Bonds_df, temp])
        except:
            continue

    Bonds_df.reset_index(drop = True, inplace=True)
    Bonds_df['date'] = pd.to_datetime(Bonds_df['date'], format='%Y-%m-%d')

    latest_date = Bonds_df['date'].max()
    Bonds_df['days'] = (latest_date - Bonds_df['date']).dt.days
    Bonds_df['days'] = Bonds_df['days'].replace({1:0,2:0, 6: 5, 7: 5, 8: 5, 31: 30, 32: 30, 33: 30, 181: 180, 182: 180, 183: 180}) # to avoid some dates are missing
    Bonds_df = Bonds_df.drop_duplicates(subset=['bond', 'days'], keep='last') # last because the order is acsending  by date
    returns_df = Bonds_df[Bonds_df.days.isin([0, 5,30,180])].copy()
    returns_df['chg5'] = returns_df['value'].diff(1)
    returns_df['chg30'] = returns_df['value'].diff(2)
    returns_df['chg180'] = returns_df['value'].diff(3)
    returns_df = returns_df[returns_df.days.isin([0])]
    returns_df.reset_index(drop=True, inplace = True)
    returns_df['chg180']=returns_df['chg180']-returns_df['chg30']
    returns_df['chg30']=returns_df['chg30']-returns_df['chg5']
    print('Done with Riksbank')
    return returns_df
