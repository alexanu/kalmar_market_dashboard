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
    return econ_events, today_macro


def get_earnings_from_gh(sp500_df):
    today_date = dt.datetime.now().date()
    earnings_df = sp500_df[['symbol','Name','Group','Earn_Quality','SCORE_VS_AVG','ERel_date','EPresi_date']]
    earnings_df = earnings_df[earnings_df['EPresi_date'].notna()]
    earnings_df = earnings_df[pd.to_datetime(earnings_df['EPresi_date']).dt.date >= today_date]
    today = pd.Timestamp(dt.date.today()) # we need pd.timestamp to convert to datetime
    next_5d = pd.Timestamp(dt.date.today()) +  pd.Timedelta('5D')
    earnings_df = earnings_df[earnings_df.EPresi_date.between(today,next_5d)]
    earnings_df = earnings_df.sort_values(by='EPresi_date').reset_index(drop=True)
    today_earnings = len(earnings_df[earnings_df['EPresi_date'].dt.date == today_date])
    # today_earnings = len(earnings_df[earnings_df.EPresi_date.between(today,pd.Timestamp(dt.date.today()) +  pd.Timedelta('1D'))])
    earnings_df['ERel_date'] = earnings_df['ERel_date'].dt.strftime('%a, %d %b, %H:%M')
    earnings_df['EPresi_date'] = earnings_df['EPresi_date'].dt.strftime('%a, %d %b, %H:%M')
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
