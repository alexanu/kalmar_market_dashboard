import plotly.express as px
import plotly.graph_objects as go
import dash
from dash import html, register_page, dcc, callback
import dash_bootstrap_components as dbc # should be installed separately
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate

register_page(
    __name__,
    name='Kalmar Market',
    top_nav=True,
    path='/'
)


from Universes import *
from Azure_config import *
import utils

import pandas as pd
import numpy as np
from github import Github

import time
import datetime as dt
import random
import os
import io
from threading import Thread

import alpaca

from alpaca.trading.enums import OrderSide, TimeInForce, AssetClass, AssetStatus, AssetExchange, OrderStatus, QueryOrderStatus, CorporateActionType, CorporateActionSubType
from alpaca.trading.requests import GetCalendarRequest, GetAssetsRequest, GetOrdersRequest, MarketOrderRequest, LimitOrderRequest, StopLossRequest, TrailingStopOrderRequest, GetPortfolioHistoryRequest, GetCorporateAnnouncementsRequest
from alpaca.data.requests import StockLatestQuoteRequest, StockTradesRequest, StockQuotesRequest, StockBarsRequest, StockSnapshotRequest, StockLatestBarRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from alpaca.data.enums import Adjustment, DataFeed, Exchange

from alpaca.trading.client import TradingClient
from alpaca.data import StockHistoricalDataClient
from alpaca.broker.client import BrokerClient


#API_KEY_PAPER = os.getenv('API_KEY_PAPER')
#API_SECRET_PAPER = os.getenv('API_SECRET_PAPER')
from Alpaca_config import *

trading_client = TradingClient(API_KEY_PAPER, API_SECRET_PAPER) # dir(trading_client)
stock_client = StockHistoricalDataClient(API_KEY_PAPER,API_SECRET_PAPER)
broker_client = BrokerClient(API_KEY_PAPER,API_SECRET_PAPER,sandbox=False,api_version="v2")


# Depending when the app is run, different dates should be used --------------------------------------------

current_hour_in_muc = pd.Timestamp.now(tz="CET").tz_localize(None).hour


# Read all prep files --------------------------------------------------------------------------------------
# BLB_URL = f'https://{STORAGE_NAME}.blob.core.windows.net/{CONTAINER_NAME}/{FILE_NAME_SP500}?{BLB_SAS}'
# sp500=pd.read_excel(BLB_URL)

repository = Github(github_strat_token).get_user().get_repo(dedicated_repo)

SP500_file = repository.get_contents(gh_csv_stocks)
sp500 = pd.read_csv(io.StringIO(SP500_file.decoded_content.decode()))
sp500['ERel_date'] = pd.to_datetime(sp500['ERel_date'], format='%Y-%m-%d %H:%M:%S')
sp500['EPresi_date'] = pd.to_datetime(sp500['EPresi_date'], format='%Y-%m-%d %H:%M:%S')
ALL_SYMBOLS = sp500.symbol.to_list()
TOP15_US_SECTOR = sp500.groupby('Sector',as_index=False).apply(lambda x: x.nlargest(15, 'Cap')).symbol.to_list() # largest companies in every sector
print(len(ALL_SYMBOLS))

macro_events_df, today_macro = utils.get_macro_events_from_fmp()

earnings_df, today_earnings = utils.get_earnings_from_fmp()
selected_columns = sp500[['symbol', 'Name', 'Group', 'Earn_Quality', 'SCORE_VS_AVG']]
earnings_df = pd.merge(earnings_df, selected_columns, how='inner', on='symbol')



ETF_mapping = repository.get_contents(gh_csv_ETFDB)
ETF_mapping_df = pd.read_csv(io.StringIO(ETF_mapping.decoded_content.decode()),sep=",")
ALL_ETFS = ETF_mapping_df.symbol.to_list()

ETF_constit = repository.get_contents(gh_csv_ETF_constit)
ETF_constit_df = pd.read_csv(io.StringIO(ETF_constit.decoded_content.decode()),sep=",")


CURR_ETF_LIST = list(Curr_dict.keys())
COUNTRY_ETF_LIST = list(Country_equity.keys())
BOND_ETF_LIST = list(Bonds_dict.keys())
USA_ETF = ['SPY']



# Market Time Block -----------------------------------------------------------------------------------------------

from locale import normalize
clock = trading_client.get_clock()
if not clock.is_open:
    time_to_open = (clock.next_open - clock.timestamp).total_seconds()//3600
else:
    time_to_close = (clock.timestamp - clock.next_close).total_seconds()//3600

now = pd.Timestamp.today() + pd.offsets.Day(-1) # we need -1 for yesterday as we need know when last close of market happened
MonthEnds = (now + pd.offsets.MonthEnd(normalize=True)).strftime("%Y-%m-%d")
QuarterEnds = (now + pd.offsets.QuarterEnd(normalize=True)).strftime("%Y-%m-%d")
tradingcal_till_moe = trading_client.get_calendar(GetCalendarRequest(start=now.strftime("%Y-%m-%d"), end=MonthEnds))
tradingcal_till_qe = trading_client.get_calendar(GetCalendarRequest(start=now.strftime("%Y-%m-%d"), end=QuarterEnds))

text0 = f'{pd.Timestamp.now(tz="EST").tz_localize(None).strftime("%H:%M")} ET'
text1 = f'Today is {pd.Timestamp.today().day_name()}, {pd.Timestamp.now(tz="CET").tz_localize(None).strftime("%b %d, %H:%M") } in Munich, which is {pd.Timestamp.now(tz="EST").tz_localize(None).strftime("%H:%M")} in New York'
if not clock.is_open:
    time_to_open = round((clock.next_open - clock.timestamp).total_seconds()/3600,1)
    already_closed_hours = round((dt.datetime.now()- tradingcal_till_moe[0].close).total_seconds() / 3600,1)
    text2 = 'Closed'
    text3 = f'US market is currently closed for already {already_closed_hours} hours. Will open in {time_to_open} hours'
else:
    time_to_close = round((clock.timestamp - clock.next_close).total_seconds()/3600,1)
    text2 = 'Open'
    text3 = f'US market is currently open. Will close in {time_to_close} hours'

text4 = f'There are {len(tradingcal_till_moe)-1} trading days till the month end and {len(tradingcal_till_qe)-1} - till end of quarter'


print('Done market')

blockMarketTime = dbc.Row( # Time of market
                        [
                            dbc.Col([
                                dbc.Row([
                                    dbc.Col([html.H3(id="lbl_time",children = text0),html.P(children=text1)]), # Time
                                    dbc.Col([html.H3(id="lbl_status",children = text2),html.P(children=text3)]), # Open / Closed
                                ]),    
                                dbc.Row([
                                    dbc.Col([html.H3(id="lbl_eom",children = len(tradingcal_till_moe)-1),html.P(children=text4)]), # Days till eom
                                    dbc.Col([html.H3(id="lbl_test",children = f'{today_macro}+{today_earnings}'),
                                             html.P(children=f"Today we have {today_macro} macro events and {today_earnings} earnings pres in US")
                                            ]),
                                ]),
                            ]),
                            dbc.Col([html.Div(
                                              # id='container-macro-events',
                                              dbc.Table.from_dataframe(macro_events_df,striped=True,bordered=True,hover=True,responsive=True, size='sm'),
                                              style={'maxHeight': '200px', 'overflow': 'scroll'}
                                              ),
                                    # html.P(children=f"Data taken on {macro_events_file.last_modified}", style={'color': 'grey', 'fontSize': 10})
                                    ]), # Macro events
                        ],
                        justify='between', # for this to work you need some space left (in total there 12 columns)
                        align = 'center',
                    )

# Getting daily data ---------------------------------------------------------------------------------------------

today = trading_client.get_clock().timestamp
previous_day = today - pd.Timedelta('1D')
previous_day_40 = today - pd.Timedelta('40D')


def download_data_daily(symbols, starting = previous_day_40, ending = previous_day, daily_rets = False):
    bars_request_params = StockBarsRequest(symbol_or_symbols=symbols, start = starting, end = ending, timeframe=TimeFrame.Day, adjustment= Adjustment.ALL,feed = DataFeed.SIP)
    daily_df = stock_client.get_stock_bars(bars_request_params).df
    daily_df = daily_df.reset_index()
    daily_df.timestamp = daily_df.timestamp.dt.date
    daily_df['days'] = (daily_df.timestamp - previous_day.date()).astype('timedelta64[D]')
    if daily_rets:
        daily_returns = daily_df[['symbol','timestamp','close']].copy()
        daily_returns['ret'] = daily_df.groupby("symbol")["close"].pct_change(1).fillna(0)
        return daily_returns
    else:
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
        returns_df = daily_df[daily_df.days.isin([d0,d5,d30])]
        returns_df['chg30'] = round(returns_df['close'].pct_change(2)*100,1)
        returns_df['chg5'] = round(returns_df['close'].pct_change(1)*100,1)
        returns_df['chg1'] = round(100*(returns_df.close-returns_df.open)/returns_df.open,1)
        returns_df = returns_df[returns_df.days.isin([d0])]
        returns_df = returns_df[['symbol','chg30','chg5','chg1']].reset_index(drop=True)
        return returns_df

etf_df = None
etf_data_loaded = False
def download_etf_data():
    global etf_df, etf_data_loaded
    etf_df = download_data_daily(ALL_ETFS)
    etf_data_loaded = True   

returns_df = download_data_daily(ALL_SYMBOLS)
sp500 = sp500.merge(returns_df[['symbol','chg30','chg5','chg1']],on=['symbol'],how ='outer')
sp500['Sec'] = sp500['Sector'].map(utils.Sector_Dict) # Mapping code to text
sector_returns_df = sp500.groupby('Sector')[['chg30','chg5','chg1']].mean().reset_index()
sector_returns_df['Sec'] = sector_returns_df['Sector'].map(utils.Sector_Dict) # Mapping code to text
sector_returns_df['Desc'] = sector_returns_df['Sector'].map(utils.Group_Dict) # Mapping code to text
sector_returns_df['chg30']=sector_returns_df['chg30']-sector_returns_df['chg5']
sector_returns_df['chg5']=sector_returns_df['chg5']-sector_returns_df['chg1']

barSecReturn = px.bar(sector_returns_df, x="Sec", y=["chg30", "chg5", "chg1"], 
            color_discrete_sequence=["PowderBlue","CornflowerBlue", "RoyalBlue"],
            title="Sector % Performance", hover_data=['Desc'])
barSecReturn.update_layout({'legend_title_text': '','transition_duration':500},
                        title_font_size = 14,title_x=0.5,title_y=0.97,
                        xaxis_title = None,yaxis_title = None, xaxis = dict(tickfont = dict(size=10), tickangle = 0),
                        # legend=dict(orientation = "h",yanchor="bottom",y=1,xanchor="center",x=0.5)  
                        legend=dict(orientation = "v",yanchor="top",y=0.7,xanchor="right",x=1.2),
                        margin=dict(b = 0, l = 0, r = 5, t = 35),
                        height = 300,
                        )

blockMarketHistogram = dbc.Row(
                            [
                                dbc.Col([
                                    dcc.Graph(id="ReturnDistrib", config={'displayModeBar': False}),
                                    dcc.RadioItems(
                                            id="radioReturnPeriod",                                        
                                            options = [
                                                {'label':'Yesterday','value':'chg1'}, 
                                                {'label':'5 days','value':'chg5'}, 
                                                {'label':'30 days','value':'chg30'}], 
                                            value = 'chg1', 
                                            inline=True,
                                            className="d-flex justify-content-center",
                                            inputStyle={"margin-right": "7px","margin-left": "13px",},
                                            ),
                                    ],
                                    ),
                                dbc.Col([dcc.Graph(id="RetSectors",figure=barSecReturn, config={'displayModeBar': False}),
                                        html.P(children=f'Only Top10 by capitalisation are taken for every sector', style={'color': 'grey', 'fontSize': 10,'marginTop': 3, 'marginBottom': 0, 'marginLeft': 15}),
                                        html.P(children=f'[chg30] means not the change over last 30 days, but [change over last 30 days] minus [change over last 5 days]', style={'color': 'grey', 'fontSize': 10, 'marginBottom': 0, 'marginTop': 0, 'marginLeft': 15}),
                                        ]),
                            ],
                        )

# Overnight returns ------------------------------------------------------------------------------------------------------

snap = stock_client.get_stock_snapshot(StockSnapshotRequest(symbol_or_symbols=ALL_SYMBOLS, feed = DataFeed.SIP))
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

snapshot_df['FULL'] = round(100*(snapshot_df['price']-snapshot_df['yest_close'])/snapshot_df['yest_close'],1)
snapshot_df['ON'] = round(100*(snapshot_df['today_open']-snapshot_df['yest_close'])/snapshot_df['yest_close'],1)
snapshot_df['DAY'] = round(100*(snapshot_df['today_close']-snapshot_df['today_open'])/snapshot_df['today_open'],1)
snapshot_df['POST'] = round(100*(snapshot_df['price']-snapshot_df['today_close'])/snapshot_df['today_close'],1)

snapshot_df = (snapshot_df
                .reset_index()
                .rename(columns={'index':'symbol'})
                .merge(sp500[['symbol','Name','Sector','Group','SCORE_VS_AVG']],on=['symbol'],how ='inner')
)

snapshot_df['Sec'] = snapshot_df['Sector'].map(utils.Sector_Dict) # Mapping code to text


blockONReturns = dbc.Row(
                            [
                                dbc.Col([
                                    dcc.Graph(id="ONReturns", config={'displayModeBar': False}),
                                    dbc.Row([
                                        dbc.Col(html.H6("Sort tickers by: "),width='auto', className="small"),
                                        dbc.Col(
                                            dcc.RadioItems(
                                                    id="radioDayReturnSort",                                        
                                                    options = [
                                                        {'label':'Return since yestr close','value':'FULL'}, 
                                                        {'label':'ON return','value':'ON'}, 
                                                        {'label':'Day return','value':'DAY'}, 
                                                        {'label':'Post-market return','value':'POST'}], 
                                                    value = 'FULL', 
                                                    className="small",
                                                    inline=True,
                                                    inputStyle={"margin-right": "7px","margin-left": "13px",},
                                                    ),
                                            width='auto'
                                            ),
                                        ],
                                        align="center",
                                        justify="center",
                                    ),
                                ],
                                width = 6,
                                ),    

                                dbc.Col([
                                    dcc.Graph(id="ReturnsScatter", config={'displayModeBar': False}),
                                    dcc.RadioItems(
                                            id="radioReturnsScatter",                                        
                                            options = [
                                                {'label':'Day vs Overnight','value':'ON'}, 
                                                {'label':'Day vs Postmarket','value':'POST'}], 
                                            value = 'ON', 
                                            className="d-flex justify-content-center small",
                                            inline=True,
                                            inputStyle={"margin-right": "7px","margin-left": "13px",},
                                            ),
                                    ]
                                    ),
                            ],
                        )


# Currencies via ETFs ------------------------------------------------------------------------------------------

currency_df = download_data_daily(CURR_ETF_LIST)
currency_df['Desc'] = currency_df['symbol'].map(Curr_dict) # Mapping code to text

barCurrChange = px.bar(currency_df, x="Desc", y=["chg30", "chg5", "chg1"], 
            color_discrete_sequence=["PowderBlue","CornflowerBlue", "RoyalBlue"],
            title="Currency % Performance", hover_data=['symbol'])
barCurrChange.update_layout({'legend_title_text': '','transition_duration':500},
                        title_font_size = 14,title_x=0.5,title_y=0.97,
                        xaxis_title = None,yaxis_title = None, xaxis = dict(tickfont = dict(size=10), tickangle = 0),
                        legend=dict(orientation = "h",yanchor="bottom",y=1,xanchor="center",x=0.5),
                        # legend=dict(orientation = "v",yanchor="top",y=0.7,xanchor="right",x=1.2),
                        margin=dict(b = 0, l = 0, r = 5, t = 50),
                        height = 300,
                        )
barCurrChange.add_annotation(xref='paper', yref='paper',x = 0.1, y = 0.8,
                text="USD becomes weaker",showarrow=False,font=dict(color="black", size=8))
barCurrChange.add_vline(x=4.5, line_width=4, line_color="white") # vertical line


# Country ETF relative performance -------------------------------------------------------

country_df = download_data_daily(COUNTRY_ETF_LIST)
spy_df = download_data_daily(USA_ETF)

country_df[['chg30','chg5','chg1']] = country_df[['chg30','chg5','chg1']].subtract(spy_df[['chg30','chg5','chg1']].iloc[0], axis='columns')
country_df['Desc'] = country_df['symbol'].map(Country_equity) # Mapping code to text
heatCountryvsSPY = px.imshow(country_df[['chg30','chg5','chg1']].T.values, x=country_df.Desc, y=['chg30','chg5','chg1'],
                             color_continuous_scale = px.colors.diverging.RdYlGn,
                             title="Country Stocks Performance vs SPY",
                             )
heatCountryvsSPY.update(layout_coloraxis_showscale=False)
heatCountryvsSPY.update_layout(title_font_size = 14,title_x=0.5,title_y=0.97,
                        margin=dict(b = 0, l = 0, r = 5, t = 50),
                        height = 300,
                        )

blockCurrency = dbc.Row([
                            dbc.Col(
                                dcc.Graph(id="CurrencyChg",figure=barCurrChange, config={'displayModeBar': False}),
                                width = 5,
                                ),
                            dbc.Col(
                                dcc.Graph(id="CountryChg",figure=heatCountryvsSPY, config={'displayModeBar': False}),
                                # width = 5,
                                ),

                        ])


# Bonds ETFs ------------------------------------------------------------------------------------------------------


bonds_daily_ret = download_data_daily(BOND_ETF_LIST,daily_rets=True)
bonds_daily_ret = bonds_daily_ret.join(pd.DataFrame.from_dict(Bonds_dict,orient='index'),on=["symbol"],how="inner")

order_yaxis = pd.DataFrame.from_dict(Bonds_dict,orient='index')[0].tolist()

# below is need for symmetrical scale
max_ret = bonds_daily_ret.ret.max(); min_ret = abs(bonds_daily_ret.ret.min())
real_max = max_ret if max_ret > min_ret else min_ret

heatBondsDaily = go.Figure(data=go.Heatmap(
                        z=bonds_daily_ret['ret'],
                        x=bonds_daily_ret['timestamp'],
                        y=bonds_daily_ret[0],
                        colorscale='RdYlGn',
                        zmax=real_max, zmin=-real_max,
                        )
        )
heatBondsDaily.update_layout(
    title='Bond ETFs daily returns',
    yaxis={'categoryorder':'category ascending'},
    xaxis_nticks=36)


blockBonds = dbc.Row([
                            dbc.Col(
                                dcc.Graph(id="BondChg",
                                          # figure=barCurrChange, 
                                          config={'displayModeBar': False}),
                                width = 3,
                                ),
                            dbc.Col(
                                dcc.Graph(id="BondDailyChg",figure=heatBondsDaily, config={'displayModeBar': False}),
                                # width = 5,
                                ),

                        ])


# Earnings announcements ------------------------------------------------------------------------------------------

blockEarnings = dbc.Row([
                            dbc.Col(
                                dbc.Card(
                                    dbc.CardBody(
                                        [
                                        html.H3('Earnings'),
                                        html.Div(
                                                dbc.Table.from_dataframe(earnings_df,striped=True,bordered=True,hover=True,responsive=True, size='sm'),
                                                style={'maxHeight': '300px', 'overflow': 'scroll'}
                                                ),
                                        html.P(children=f"Data taken on {SP500_file.last_modified}", style={'color': 'grey', 'fontSize': 10})
                                        ]
                                    ),
                                )
                            )
                        ])


# Equity ETFs -----------------------------------------------------------------------------------------------------


blockETFReturns = dbc.Row(
                            [
                                dbc.Col([
                                    html.Button("Download Data", id="download-button"),
                                    html.Div(id="download-status"),
                                    html.P(),
                                    dbc.Row([
                                        dbc.Col([
                                                html.H5('Fund Type'), 
                                                dcc.Dropdown(id = 'dropETFtype',
                                                            options=[{'label': i, 'value': i} for i in ETF_mapping_df['Fund Type'].unique()],
                                                            value=['Equity'],
                                                            multi=True),
                                        ]),
                                        dbc.Col([
                                                html.H5('Geographic Focus'), 
                                                dcc.Dropdown(id = 'dropETFgeo',value=[],multi=True),
                                        ]),
                                    ]
                                    ),
                                    dcc.Graph(id="ETFReturns", config={'displayModeBar': False}),
                                    dbc.Row([
                                        dbc.Col(html.H6("Sort tickers by: "),width='auto', className="small"),
                                        dbc.Col(
                                            dcc.RadioItems(
                                                    id="radioETFDayReturnSort",                                        
                                                    options = [
                                                        {'label':'Yesterday','value':'chg1'}, 
                                                        {'label':'5 days','value':'chg5'}, 
                                                        {'label':'30 days','value':'chg30'}], 
                                                    value = 'chg5', 
                                                    className="small",
                                                    inline=True,
                                                    inputStyle={"margin-right": "7px","margin-left": "13px",},
                                                    ),
                                            width='auto'
                                            ),
                                        ],
                                        align="center",
                                        justify="center",
                                    )
                                ])    
                            ]
                        )

# ETF Constituents ---------------------------------------------------------------------------------------------------------


All_ETF_groups = sorted(ETF_mapping_df[ETF_mapping_df['Constitut'] == "Yes"].Group.unique().tolist())
blockETFConstitReturns = dbc.Row(
                            [
                                dbc.Col([
                                    dbc.Row([
                                        dbc.Col([
                                                html.H5('Fund Type'),],
                                                width = 3
                                        ),
                                        dbc.Col([
                                                dcc.Dropdown(id = 'dropETFStrategies',
                                                            options=[{'label': i, 'value': i} for i in All_ETF_groups],
                                                            value='Events',
                                                            multi=False),],
                                                width = 3
                                        ),
                                    ]
                                    ),
                                    html.P(),
                                    dcc.Graph(id="ETFConstitReturns", config={'displayModeBar': False}),
                                    dbc.Row([
                                        dbc.Col(html.H6("Sort tickers by: "),width='auto', className="small"),
                                        dbc.Col(
                                            dcc.RadioItems(
                                                    id="radioETFConstitReturnSort",                                        
                                                    options = [
                                                        {'label':'Yesterday','value':'chg1'}, 
                                                        {'label':'5 days','value':'chg5'}, 
                                                        {'label':'30 days','value':'chg30'}], 
                                                    value = 'chg5', 
                                                    className="small",
                                                    inline=True,
                                                    inputStyle={"margin-right": "7px","margin-left": "13px",},
                                                    ),
                                            width='auto'
                                            ),
                                        ],
                                        align="center",
                                        justify="center",
                                    )
                                ])    
                            ]
                        )

# VIX ------------------------------------------------------------------------------------------------------------
VIX_dict = {
    'VIXY':'1M maturity CBOE Volatility futures',
    'VIXM':'5M maturity CBOE Volatility futures'
}

# Controls ------------------------------------------------------------------------------------------------

controls = dbc.Row( # always do in rows ...
    [
        dbc.Col( # ... and then split to columns
            [   
                dbc.Row(
                    [
                        dbc.Col(dbc.Label("\N{factory} Focus on a specific sector "),width=9,),
                        dbc.Col(
                            [
                                dbc.Button("\N{books}",id="hover-target3", color="link", n_clicks=0),
                                dbc.Popover(dbc.PopoverBody("Scope of sectors could be different for different emission scenario.\nScope of sectors covered by the tool is constantly growing."),id="hover3",target="hover-target3",trigger="hover"), 
                            ], 
                            width=2,
                            ),                           
                    ],
                    align="center",
                ),
                dcc.Dropdown(
                            id = 'universe',
                            clearable=False,
                            value=TOP15_US_SECTOR[0], 
                            options=[{'label': c, 'value': c} for c in TOP15_US_SECTOR],
                            placeholder="Select a sector"),       
            
            ],
        ),
    ],
)



# Define main Layout ----------------------------------------------------------------------------------------


def layout():
# dash_app.layout = dbc.Container( # always start with container
    layout = dbc.Container( # always start with container
                children=[
                    dcc.Interval(id="interval",interval=1000,n_intervals=0), # for downloading a lot of data
                    html.Hr(), # small space from the top
                    blockMarketTime, html.Hr(),
                    blockMarketHistogram, html.Hr(),
                    blockONReturns, html.Hr(),
                    blockCurrency, html.Hr(),
                    blockBonds, html.Hr(),
                    blockEarnings, html.Hr(),
                    blockETFReturns, html.Hr(),
                    blockETFConstitReturns, html.Hr(),
                    dbc.Row(
                        [
                            dbc.Col(
                                [ # filters pane
                                    dbc.Card(
                                        dbc.CardBody(
                                            [
                                                dbc.Row(
                                                    [ # Row with key figures
                                                        dbc.Col(html.H5("Filters", className="pf-filter")), # PF score
                                                        dbc.Col(
                                                            html.Div(
                                                                dbc.Button("Reset filters", 
                                                                            id="reset-filters-but", 
                                                                            outline=True, color="dark",size="sm",className="me-md-2"
                                                                ),
                                                                className="d-grid gap-2 d-md-flex justify-content-md-end"
                                                            )
                                                        ),
                                                    ]
                                                ),
                                                html.P("Select part of your portfolio", className="text-black-50"),
                                                controls,
                                            ]
                                        )
                                    ),     
                                    html.Br(),      
                                ],
                                width=3,
                            ),
                            dbc.Col(
                                [ # main pane
                                dbc.Card(
                                    dbc.CardBody([
                                        dbc.Row(# Row with key figures
                                            [ 
                                                dbc.Col( # PF score
                                                    dbc.Card(dbc.CardBody(
                                                                        [
                                                                            html.H1(id="output-info"),
                                                                            html.Div('Portfolio-level temperature rating of selected companies', style={'color': 'black', 'fontSize': 16}),
                                                                            html.Div('in delta degree Celcius', style={'color': 'grey', 'fontSize': 10}),
                                                                        ]
                                                                    )
                                                            ),       
                                                    ),
                                                dbc.Col( # Portfolio EVIC
                                                    dbc.Card(dbc.CardBody(
                                                                        [
                                                                            html.H1(id="evic-info"),
                                                                            html.Div('Enterprise Value incl. Cash of selected portfolio', style={'color': 'black', 'fontSize': 16}),
                                                                            html.Div('in billions of template curr', style={'color': 'grey', 'fontSize': 10}),
                                                                        ]
                                                                    )
                                                            ),       
                                                    ),
                                                dbc.Col( # Portfolio notional
                                                    dbc.Card(dbc.CardBody(
                                                                        [
                                                                            html.H1(id="pf-info"),
                                                                            html.Div('Total Notional of a selected portfolio', style={'color': 'black', 'fontSize': 16}),
                                                                            html.Div('in millions of template curr', style={'color': 'grey', 'fontSize': 10}),
                                                                        ]
                                                                    )
                                                            ),       
                                                    ),                                                                                        
                                                dbc.Col( # Number of companies
                                                    dbc.Card(dbc.CardBody(
                                                                        [
                                                                            html.H1(id="comp-info"),
                                                                            html.Div('Number of companies in the selected portfolio', style={'color': 'black', 'fontSize': 16}),
                                                                            html.Div('# of companies', style={'color': 'grey', 'fontSize': 10}),
                                                                        ]
                                                                    )
                                                            ),       
                                                    ),                                                                                        
                                            ],
                                        ),
                                        dbc.Row(# row with 2 graphs
                                            [
                                                dbc.Col(dcc.Graph(id="graph-daily-ts"),width=8), # big bubble graph
                                                dbc.Col(dcc.Graph(id="graph-6"),), # covered graph
                                            ],
                                        ),
                                        dbc.Row(# row with 2 graphs
                                            [
                                                dbc.Col(dcc.Graph(id="graph-3")),
                                                dbc.Col(dcc.Graph(id="graph-4")),
                                            ]
                                        ),
                                        dbc.Row(# row with 1 bar graph
                                            [
                                                dbc.Col(dcc.Graph(id="graph-5")),
                                            ]
                                        ),
                                    ])
                                ),
                                html.Br(),
                                ],
                                width=9,
                            ),
                        ]
                    ),
                    dbc.Row( # Table
                        dbc.Col(
                            dbc.Card(
                                dbc.CardBody( 
                                        [
                                            dbc.Row(
                                                [
                                                    dbc.Col(
                                                        html.H5("Table below contains details about the members of the selected portfolio"),
                                                        width=10,
                                                        ), 
                                                ],
                                                align="center",
                                            ),
                                            html.Br(),
                                            html.Div(id='container-button-basic'),
                                        ]
                                ),
                            ),                            
                        )
                    )
                ],
            style={"max-width": "1500px"},
            )
    return layout


# Define callback to update graph ---------------------------------------------------------------------------


@callback(
    Output('ReturnDistrib', 'figure'),
    Input(component_id='radioReturnPeriod', component_property='value')
)
def build_graph(needed_column):
    if needed_column == 'chg1':
        graph_title = f"Distribution of ystrd ({previous_day.strftime('%Y-%m-%d')}) returns"
    elif needed_column == 'chg5':
        graph_title = f"Distribution of last 5 days returns"
    else:
        graph_title = "Distribution of last 30 days returns"

    hstgrRet = px.histogram(sp500, x=needed_column, color = 'Sec', nbins = 10, title=graph_title)
    hstgrRet.update_layout({'legend_title_text': '','transition_duration':500},
                            title_font_size = 14,title_x=0.5,title_y=0.97,
                            xaxis_title = None,yaxis_title = None,                            
                            legend=dict(orientation = "v",yanchor="top",y=0.9,xanchor="left",x=-0.3),
                            margin=dict(b = 0, l = 0, r = 0, t = 35),
                            height = 300,
                            )
    return hstgrRet


@callback(
    Output('ONReturns', 'figure'),
    Input(component_id='radioDayReturnSort', component_property='value')
)
def build_graph(needed_column):

    global snapshot_df

    graph_title = f"% Performance from close on {snapshot_df['yest'][0].day_name()}"

    snapshot_df = snapshot_df.sort_values(by=needed_column, ascending=False) # sort by capitalization

    small = pd.concat([snapshot_df.nlargest(20,needed_column),snapshot_df.nsmallest(20,needed_column)])

    barONReturn = px.bar(small, x='symbol', y=["ON", "DAY", "POST"], 
                color_discrete_sequence=["PowderBlue","CornflowerBlue", "RoyalBlue"],
                title = graph_title,
                hover_data={'variable': False, 'value': False, 'Name':True,'Group':True,'price time short':True})
    barONReturn.update_layout({'legend_title_text': '','transition_duration':500},
                            title_font_size = 14,title_x=0.5,title_y=0.97,
                            xaxis_title = None,yaxis_title = None, xaxis = dict(tickfont = dict(size=10), tickangle = 270),
                            legend=dict(orientation = "h",yanchor="bottom",y=0.8,xanchor="right",x=1),
                            margin=dict(b = 0, l = 0, r = 0, t = 35),
                            )
    return barONReturn


@callback(
    Output('ReturnsScatter', 'figure'),
    Input(component_id='radioReturnsScatter', component_property='value')
)
def build_graph(needed_column):

    scatDailyReturns = px.scatter(snapshot_df, x='DAY', y=needed_column, 
                                color = "Sec", labels={"color": "Sector"}, 
                                hover_data=["Name", "Group", "FULL"],
                                title="Returns")
    scatDailyReturns.update_layout({'legend_title_text': '','transition_duration':500},
                                title_font_size = 14,title_x=0.5,title_y=0.97,
                                # xaxis_title = None,yaxis_title = None, 
                                xaxis = dict(tickfont = dict(size=10), tickangle = 0),
                                # legend=dict(orientation = "h",yanchor="bottom",y=1,xanchor="center",x=0.5)  
                                # legend=dict(orientation = "v",yanchor="top",y=0.7,xanchor="right",x=1.2),
                                margin=dict(b = 0, l = 0, r = 5, t = 35),
                                # height = 300,
                            )
    return scatDailyReturns


@callback(
    [
        Output("download-status", "children"),
        Output('ETFReturns', 'figure'),
    ],
    [
        Input(component_id='radioETFDayReturnSort', component_property='value'),
        Input("download-button", "n_clicks")
    ],
    prevent_initial_call=True
)
def build_graph(needed_column, n_clicks):

    global download_thread, etf_data_loaded, etf_df
    ctx = dash.callback_context

    if ctx.triggered[0]['prop_id'] == 'download-button.n_clicks':
        download_thread = Thread(target=download_etf_data)
        download_thread.start()
        while not etf_data_loaded:
            time.sleep(1)
        if needed_column == 'chg1':
            graph_title = f"% ETF Perfo sorted by ystrd ({previous_day.strftime('%Y-%m-%d')}) returns"
        elif needed_column == 'chg5':
            graph_title = f"% ETF Perfo sorted by last 5 days returns"
        else:
            graph_title = "% ETF Perfo sorted by last 30 days returns"

        etf_df = etf_df.sort_values(by=needed_column, ascending=False) # sort by capitalization
        small_etf = pd.concat([etf_df.nlargest(30,needed_column),etf_df.nsmallest(30,needed_column)])
        small_etf = small_etf.merge(ETF_mapping_df[['symbol','Fund Name','Fund Type']],on=['symbol'],how ='inner')
        small_etf = small_etf.sort_values(by=needed_column, ascending=False) # sort by capitalization

        barETFReturn = px.bar(small_etf, x='symbol', y=needed_column, 
                    color='Fund Type',
                    title = graph_title,
                    hover_data={'Fund Name':True,'Fund Type':True})
        barETFReturn.update_layout({'legend_title_text': '','transition_duration':500},
                                title_font_size = 14,title_x=0.5,title_y=0.97,
                                xaxis_title = None,yaxis_title = None, xaxis = dict(tickfont = dict(size=10), tickangle = 270),
                                legend=dict(orientation = "h",yanchor="bottom",y=1,xanchor="center",x=0.5),
                                margin=dict(b = 0, l = 0, r = 0, t = 35),
                                )
        return "Data download completed.", barETFReturn
    else:
        return "Press the button to download data.", {}

@callback(
        Output('dropETFgeo', 'options'),
        [Input('dropETFtype', 'value')]
)
def set_ETF_geo_options(ETFtype):
    if len(ETFtype)> 0:
        ETFtypes = ETFtype
        return [{'label': i, 'value': i} for i in sorted(set(ETF_mapping_df['Geographic Focus'].loc[ETF_mapping_df['Fund Type'].isin(ETFtypes)]))]
    else:
        ETFtypes = []
        return [{'label': i, 'value': i} for i in sorted(set(ETF_mapping_df['Geographic Focus'].loc[ETF_mapping_df['Fund Type'].isin(ETFtypes)]))]


@callback(
        Output('ETFConstitReturns', 'figure'),
    [
        Input(component_id='radioETFConstitReturnSort', component_property='value'),
        Input('dropETFStrategies', 'value')
    ],
)
def build_ETF_constit(needed_column, Group_Name):

    title_part1 = f'Constitutes % Perf of [{Group_Name}] group sorted '
    if needed_column == 'chg1':
        graph_title = title_part1 + f"by ystrd ({previous_day.strftime('%Y-%m-%d')}) returns"
    elif needed_column == 'chg5':
        graph_title = title_part1 + "by last 5 days returns"
    else:
        graph_title = title_part1 + "by last 30 days returns"

    ETFs_in_Scope = ETF_mapping_df[ETF_mapping_df['Group'] == Group_Name].symbol.to_list()
    Constit_in_Scope = ETF_constit_df[ETF_constit_df.symbolETF.isin(ETFs_in_Scope)].symbol.unique().tolist()
    small_sp500 = sp500[sp500.symbol.isin(Constit_in_Scope)].sort_values(by=needed_column, ascending=False)
    small_sp500 = small_sp500.sort_values(by=needed_column, ascending=False)

    barConstitReturn = px.bar(small_sp500, x='symbol', y=needed_column, 
                            color='SCORE_VS_AVG', color_continuous_scale='RdYlGn',
                            title = graph_title, 
                            hover_data={'Name':True,'Group':True, 'EarnScore':True}
                            )
    barConstitReturn.update_layout({'legend_title_text': '','transition_duration':500},
                            title_font_size = 14,title_x=0.5,title_y=0.97,
                            xaxis_title = None,yaxis_title = None, xaxis = dict(tickfont = dict(size=10), tickangle = 270),
                            legend=dict(orientation = "h",yanchor="bottom",y=1,xanchor="center",x=0.5),
                            margin=dict(b = 0, l = 0, r = 0, t = 35),
                            )

    return barConstitReturn




@callback(
    [
        Output("graph-daily-ts", "figure"), 
        Output("graph-6", "figure"),
    ],
    [Input("universe", "value")]
)

def update_figure(value):
    today = trading_client.get_clock().timestamp
    previous_day = today - pd.Timedelta('1D')
    previous_day_10 = today - pd.Timedelta('10D')
    bars_request_params = StockBarsRequest(symbol_or_symbols=value, start = previous_day_10, end = previous_day, timeframe=TimeFrame.Day, adjustment= Adjustment.RAW,feed = DataFeed.SIP)
    df = stock_client.get_stock_bars(bars_request_params).df.droplevel(level=0) # drop level is needed as 1st it appears with multiindex with symbol
    fig = px.line(df, x=df.index, y="close", title='Closing Price')
    return (fig, fig,)

