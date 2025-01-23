import plotly.express as px
import plotly.graph_objects as go
import dash
from dash import html, register_page, dcc, callback
import dash_bootstrap_components as dbc # should be installed separately
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
import dash_ag_grid as dag  # pip install dash-ag-grid


register_page(
    __name__,
    name='Vilni Market',
    top_nav=True,
    path='/'
)

import utils
from Universes import *

import pandas as pd
from github import Github

import time
import datetime as dt
import io
import os
from threading import Thread
import logging

from alpaca.trading.requests import GetCalendarRequest
from alpaca.trading.client import TradingClient

from alpaca.data import StockHistoricalDataClient
from alpaca.data.requests import StockSnapshotRequest
from alpaca.data.enums import DataFeed

gh_csv_stocks = 'SP500.csv'
gh_csv_macro_events = 'upcoming_econ_events.csv'
gh_csv_ETFDB = 'ETF_ticker_mapping.csv'
gh_csv_ETF_constit = 'ETF_constit.csv'
gh_csv_watchlist = 'Trading Ideas.csv'

fmp_api_key = os.getenv('fmp_api_key')
github_strat_token = os.getenv('github_strat_token')
dedicated_repo = os.getenv('dedicated_repo')


global sp500_data_fetched, etf_data_fetched, country_etf_fetched, currency_fetched, yield_fetched
sp500_data_fetched = None
etf_data_fetched = None
currency_fetched = None
yield_fetched = None
country_etf_fetched = None


API_KEY_PAPER = os.environ['API_KEY_PAPER']
API_SECRET_PAPER = os.environ['API_SECRET_PAPER']
trading_client = TradingClient(API_KEY_PAPER, API_SECRET_PAPER) # dir(trading_client)
stock_client = StockHistoricalDataClient(API_KEY_PAPER, API_SECRET_PAPER)


# Depending when the app is run, different dates should be used --------------------------------------------
current_hour_in_muc = pd.Timestamp.now(tz="CET").tz_localize(None).hour
print(f'Current hour in Munich: {current_hour_in_muc}')


# Read all prep files --------------------------------------------------------------------------------------

repository = Github(github_strat_token).get_user().get_repo(dedicated_repo)

SP500_file = repository.get_contents(gh_csv_stocks)
sp500 = pd.read_csv(io.StringIO(SP500_file.decoded_content.decode()))
sp500 = sp500.sort_values(by='Cap', ascending=False) # sort by capitalization
sp500['ERel_date'] = pd.to_datetime(sp500['ERel_date'], format='%Y-%m-%d %H:%M:%S')
sp500['EPresi_date'] = pd.to_datetime(sp500['EPresi_date'], format='%Y-%m-%d %H:%M:%S')
ALL_SYMBOLS = sp500.symbol.to_list()
print(f'SP500 file from refinitiv was updated on {SP500_file.last_modified}. There are {len(ALL_SYMBOLS)} tickers in it.')


macro_events_df, today_macro = utils.get_macro_events_from_fmp()

ETF_mapping = repository.get_contents(gh_csv_ETFDB)
ETF_mapping_df = pd.read_csv(io.StringIO(ETF_mapping.decoded_content.decode()),sep=",")
ALL_ETFS = ETF_mapping_df.symbol.to_list()
print(f'ETF file read from GH. It was updated on {ETF_mapping.last_modified}. There are {len(ALL_ETFS)} ETFs in scope.')


COUNTRY_ETF_LIST = list(Country_equity.keys())
BOND_ETF_LIST = list(Bonds_dict.keys())
USA_ETF = ['SPY']

if not sp500_data_fetched:
    returns_df, market_case = utils.download_data_daily(ALL_SYMBOLS)
    sp500 = sp500.merge(returns_df,on=['symbol'],how ='outer')
    sp500['Sec'] = sp500['Sector'].map(utils.Sector_Dict) # Mapping code to text
    sp500_data_fetched = True
else:
    print('Stock data already available.')

if not etf_data_fetched:
    etf_df = utils.download_etf_daily_data(ALL_ETFS)
    spy_df = etf_df[etf_df.symbol =='SPY'].copy()
    etf_data_fetched = True
else:
    print('ETF data already available.')

earnings_df, today_earnings = utils.get_earnings_from_gh(sp500)



# via FMP, but stopped working:
'''
earnings_df, today_earnings = utils.get_earnings_from_fmp()
selected_columns = sp500[['symbol', 'Name', 'Group', 'Earn_Quality', 'SCORE_VS_AVG']]
earnings_df = pd.merge(earnings_df, selected_columns, how='inner', on='symbol')
'''


# Market Time Block -----------------------------------------------------------------------------------------------


status = utils.check_trading_day()

text0 = f'{pd.Timestamp.now(tz="EST").tz_localize(None).strftime("%H:%M")} ET'
text1 = f'Today is {pd.Timestamp.today().day_name()}, {pd.Timestamp.now(tz="CET").tz_localize(None).strftime("%b %d, %H:%M") } in Munich, which is {pd.Timestamp.now(tz="EST").tz_localize(None).strftime("%H:%M")} in New York'
if status['market_status'] == 'closed':
    text2 = 'Closed'
    text3 = f"US market is currently closed for already {round(status['time_since_market_close']/60,1)} hours. Will open in {round(status['time_till_market_open'],1)} hours"
else:
    text2 = 'Open'
    text3 = f"US market is open already for {round(status['time_since_market_open'],1) }. Will close in {round(status['time_till_market_close'],1)} hours"

text4 = f"Today day is {status['trading_day']}. There are {status['trading_days_future']} trading days till the month end."


blockMarketTime = dbc.Row( # Time of market
                        [
                            dbc.Col([
                                dbc.Row([
                                    dbc.Col([html.H3(id="lbl_time",children = text0),html.P(children=text1)]), # Time
                                    dbc.Col([html.H3(id="lbl_status",children = text2),html.P(children=text3)]), # Open / Closed
                                ]),    
                                dbc.Row([
                                    dbc.Col([html.H3(id="lbl_eom",children = status['trading_days_future']),html.P(children=text4)]), # Days till eom
                                    dbc.Col([html.H3(id="lbl_test",children = f'{today_macro}+{today_earnings}'),
                                             html.P(children=f"Today we have {today_macro} macro events and {today_earnings} earnings pres in US")
                                            ]),
                                ]),
                            ]),
                            dbc.Col([
                                    html.Label('Upcoming macro events:', style={'fontSize': 16, 'fontWeight': 'bold', 'marginBottom': '10px'}),
                                    html.Div(
                                              # id='container-macro-events',
                                              dbc.Table.from_dataframe(macro_events_df,striped=True,bordered=True,hover=True,responsive=True, size='sm')
                                              if not macro_events_df.empty else "No data available",
                                              style={'maxHeight': '200px', 'overflow': 'scroll'}
                                              ),
                                    html.Label(f'Source: FMP', style={'color':'grey', 'fontSize': 10})
                                    ]), # Macro events
                        ],
                        justify='between', # for this to work you need some space left (in total there 12 columns)
                        align = 'center',
                    )


# Daily returns with table --------------------------------------------------------------------------------------------------------------

needed_for_table_cols = ['symbol', 'Name', 'Sector', '52W_HL_dec_1isSmall', 'YoY_vs_AVG_decil_1isSmall', 'MONTH_vs_AVG_decil_1isSmall',
                        'AvgVolumeChgPerc', '52W_High_Chg', 'ShortInterest', 'ShortSqueeze', 'Insiders',
                        'Earn_Quality', 'Earn_Quality_1Q_Chg','Earn_Quality_1YChg', 'AvgScore', 
                        'SCORE_VS_AVG','Sec','chg30','chg5','chg1','streak','PRE','DAY','POST']
needed_for_table_cols_integer = ['52W_HL_dec_1isSmall', 'YoY_vs_AVG_decil_1isSmall', 'MONTH_vs_AVG_decil_1isSmall',
                        'AvgVolumeChgPerc', '52W_High_Chg', 'ShortInterest', 'ShortSqueeze', 'Insiders',
                        'Earn_Quality', 'Earn_Quality_1Q_Chg','Earn_Quality_1YChg', 'AvgScore', 
                        'SCORE_VS_AVG','chg30','chg5','chg1','streak','PRE','DAY','POST']

sp500_small = sp500[needed_for_table_cols].copy()

# create a number-based filter for columns with integer data
col_defs = []
for i in sp500_small.columns:
    if i in needed_for_table_cols_integer:
        col_defs.append({"field": i, "filter": "agNumberColumnFilter"})
    else:
        col_defs.append({"field": i})

tableSP500 = dag.AgGrid(
    id="my-table",
    rowData=sp500_small.to_dict("records"),
    columnDefs=col_defs,
    defaultColDef={"resizable": True, "sortable": True, "filter": True, "minWidth":115},
    columnSize="autoSize",
    dashGridOptions={"pagination": True, "paginationPageSize":10},
    className="ag-theme-alpine mt-3",  # https://dashaggrid.pythonanywhere.com/layout/themes
)

blockAllReturnsDaily = dbc.Row(
                            [
                                dbc.Col([
                                    dcc.Graph(id="ReturnsHistogramm", config={'displayModeBar': False}),
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

                                    ], width = 4
                                    ),
                                dbc.Col([dcc.Graph(id="SectorReturns", config={'displayModeBar': False}),
                                        ], width = 6),
                                dbc.Col([dcc.Graph(id="AllvsSPY", config={'displayModeBar': False}),
                                        ], width = 2),
                                html.Br(),
                                tableSP500,
                                html.Label(f'Ratings data in table was updated on {SP500_file.last_modified}. Performance is actual.', style={'color':'grey', 'fontSize': 10,'marginLeft': '20px'})

                            ],
                        )



# Overnight returns ------------------------------------------------------------------------------------------------------

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
                                                        {'label':'Pre-market return','value':'PRE'}, 
                                                        {'label':'Day return','value':'DAY'}, 
                                                        {'label':'Post-market return','value':'POST'}], 
                                                    value = 'PRE', 
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
                                # width = 6,
                                ),    

                            ],
                        )

# Streaks block -------------------------------------------------------------------------------------------------------

graph_title = f"Distribution of current streaks as of yest ({previous_day.strftime('%Y-%m-%d')})"
hstgrStreak = px.histogram(sp500_small, x='streak', nbins = 10, title=graph_title)
hstgrStreak.update_layout({'legend_title_text': '','transition_duration':500},
                        title_font_size = 14,title_x=0.5,title_y=0.97,
                        xaxis_title = None,yaxis_title = None,                            
                        legend=dict(orientation = "v",yanchor="top",y=0.9,xanchor="left",x=-0.3),
                        margin=dict(b = 0, l = 0, r = 0, t = 35),
                        height = 300,
                        )

num_streaks=4 
while len(sp500_small[sp500_small.streak >num_streaks])>10: # just to avoid too many tickers on chart
    num_streaks = num_streaks + 1
barStreakReturn = px.bar(sp500_small[sp500_small.streak >num_streaks], x="symbol", y=["chg30", "chg5", "chg1"], 
            color_discrete_sequence=["PowderBlue","CornflowerBlue", "RoyalBlue"],
            title="% Performance of top positive streaks", hover_data=['Name','Sector'])
barStreakReturn.update_layout({'legend_title_text': '','transition_duration':500},
                        title_font_size = 14,title_x=0.5,title_y=0.97,
                        xaxis_title = None,yaxis_title = None, xaxis = dict(tickfont = dict(size=10), tickangle = 0),
                        legend=dict(orientation = "h",yanchor="bottom",y=1,xanchor="center",x=0.5),  
                        # legend=dict(orientation = "v",yanchor="top",y=0.7,xanchor="right",x=1.2),
                        margin=dict(b = 0, l = 0, r = 5, t = 50),
                        height = 300,
                        )


blockStreaks = dbc.Row(
                            [
                                dbc.Col([
                                    dcc.Graph(id="Streaks_histog", figure=hstgrStreak, config={'displayModeBar': False}),
                                ],
                                # width = 6,
                                ),    

                                dbc.Col([
                                    dcc.Graph(id="TopStreaksPerf", figure = barStreakReturn, config={'displayModeBar': False}),
                                ],
                                # width = 6,
                                ),    

                            ],
                        )


# Currencies via ETFs ------------------------------------------------------------------------------------------

if not currency_fetched:
    currency_df = utils.download_fx_daily_data_fmp()
    currency_fetched = True
else:
    print('Currency data already fetched')


barCurrChange = px.bar(currency_df, x="Desc", y=["chg30", "chg5"], 
            color_discrete_sequence=["PowderBlue","CornflowerBlue"],
            title="Currency % Performance", hover_data=['symbol'])
barCurrChange.update_layout({'legend_title_text': '','transition_duration':500},
                        title_font_size = 14,title_x=0.5,title_y=0.97,
                        xaxis_title = None,yaxis_title = None, xaxis = dict(tickfont = dict(size=10), tickangle = 0),
                        legend=dict(orientation = "h",yanchor="bottom",y=1,xanchor="center",x=0.5),
                        # legend=dict(orientation = "v",yanchor="top",y=0.7,xanchor="right",x=1.2),
                        margin=dict(b = 0, l = 0, r = 5, t = 50),
                        height = 300,
                        )
barCurrChange.add_annotation(xref='paper', yref='paper',x = 0.03, y = 0.9,
                text="EUR becomes more expensive",showarrow=False,font=dict(color="black", size=8))


# Country ETF relative performance -------------------------------------------------------

if not country_etf_fetched:
    country_df, case_market_country = utils.download_data_daily(COUNTRY_ETF_LIST)
    country_df[['chg30','chg5','chg1']] = country_df[['chg30','chg5','chg1']].subtract(spy_df[['chg30','chg5','chg1']].iloc[0], axis='columns')
    country_df['Desc'] = country_df['symbol'].map(Country_equity) # Mapping code to text
    country_etf_fetched = True
else:
    print('Country ETFs data already fetched')


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
                            dbc.Col([
                                dcc.Graph(id="CurrencyChg",figure=barCurrChange, config={'displayModeBar': False}),
                                html.Label('Change of worth of EUR in different currency. Source: FMP', style={'color':'grey', 'fontSize': 10,'marginLeft': '20px'})
                                ],
                                width = 5,
                                ),
                            dbc.Col(
                                dcc.Graph(id="CountryChg",figure=heatCountryvsSPY, config={'displayModeBar': False}),
                                # width = 5,
                                ),

                        ])


# Bonds ETFs ------------------------------------------------------------------------------------------------------

Bonds = pd.DataFrame.from_dict(Bonds_dict, orient='index', columns=['desc', 'order']).reset_index()
Bonds.columns = ['symbol', 'Desc', 'order']
small_df = etf_df[etf_df['symbol'].isin(Bonds_dict.keys())]
bonds_ETFs_df = pd.merge(small_df, Bonds, on='symbol', how='left')
bonds_ETFs_df= bonds_ETFs_df.sort_values(by='order')

barBondsETFReturn = px.bar(bonds_ETFs_df, x="Desc", y=["chg30", "chg5", "chg1"], 
            color_discrete_sequence=["PowderBlue","CornflowerBlue", "RoyalBlue"],
            title="Bonds ETFs % Performance")
barBondsETFReturn.update_layout({'legend_title_text': '','transition_duration':500},
                        title_font_size = 14,title_x=0.5,title_y=0.97,
                        xaxis_title = None,yaxis_title = None, xaxis = dict(tickfont = dict(size=10), tickangle = 0),
                        legend=dict(orientation = "v",yanchor="top",y=1.2,xanchor="right",x=1),
                        margin=dict(b = 0, l = 0, r = 5, t = 35),
                        height = 300,
                        )


blockBonds = dbc.Row([
                            dbc.Col(
                                dcc.Graph(id="BondChg",
                                          # figure=bar10YYieldChange, 
                                          config={'displayModeBar': False}),
                                width = 4,
                                ),
                            dbc.Col(
                                dcc.Graph(id="BondDailyChg", figure=barBondsETFReturn,config={'displayModeBar': False}),
                                # width = 5,
                                ),
                            html.Div(id='hidden-div', style={'display': 'none'})  # hidden div to trigger the API call
                        ])


# Earnings announcements ------------------------------------------------------------------------------------------

if earnings_df.empty:
    blockEarnings = html.Div()  # Empty div if DataFrame is empty
else:
    earning_symbols = earnings_df.symbol.to_list()
    # tickers_returns_df = utils.download_data_daily(earning_symbols+['SPY'])
    earnings_df = earnings_df.sort_values('chg30',ascending=False)
    barEarningsSymbolsChange = px.bar(earnings_df, x="symbol", y=["chg30", "chg5", "chg1",'PRE','DAY','POST'], 
                color_discrete_sequence=["PowderBlue","CornflowerBlue", "RoyalBlue", "plum", "palevioletred","purple"],
                title=f"Tickers % Performance as of {previous_day.strftime('%Y-%m-%d')}")
    barEarningsSymbolsChange.update_layout({'legend_title_text': '','transition_duration':500},
                            title_font_size = 14,title_x=0.5,title_y=0.97,
                            xaxis_title = None,yaxis_title = None, xaxis = dict(tickfont = dict(size=10), tickangle = 0),
                            legend=dict(orientation = "h",yanchor="bottom",y=-0.2,xanchor="center",x=0.5),
                            margin=dict(b = 5, l = 0, r = 0, t = 30),
                            height = 330,
                            )

    earnings_df_visual = earnings_df[['symbol','Name','Group','Earn_Quality','SCORE_VS_AVG','ERel_date','EPresi_date']].copy()

    blockEarnings = dbc.Row([
                                dbc.Col(
                                    dbc.Card(
                                        dbc.CardBody(
                                            [
                                            html.H3('Earnings'),
                                            html.Div(
                                                    dbc.Table.from_dataframe(earnings_df_visual,striped=True,bordered=True,hover=True,responsive=True, size='sm'),
                                                    style={'maxHeight': '300px', 'overflow': 'scroll'}
                                                    ),
                                            html.P(children=f"Data taken on {SP500_file.last_modified}", style={'color': 'grey', 'fontSize': 10}),
                                            dcc.Graph(id="EarningsStocksDailyChg",figure=barEarningsSymbolsChange, config={'displayModeBar': False}),
                                            ]
                                        ),
                                    )
                                )
                            ])


# Equity ETFs -----------------------------------------------------------------------------------------------------



blockETFReturns = dbc.Row([
                    dbc.Col([
                            html.P(),
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
                            ),
                        ]) 
                    ])   


# VIX ------------------------------------------------------------------------------------------------------------
VIX_dict = {
    'VIXY':'1M maturity CBOE Volatility futures',
    'VIXM':'5M maturity CBOE Volatility futures'
}




# Define main Layout ----------------------------------------------------------------------------------------


def layout():
# dash_app.layout = dbc.Container( # always start with container
    layout = dbc.Container( # always start with container
                children=[
                    dcc.Interval(id="interval",interval=1000,n_intervals=0), # for downloading a lot of data
                    html.Hr(), # small space from the top
                    blockMarketTime, html.Hr(),
                    blockAllReturnsDaily, html.Hr(),
                    blockONReturns, html.Hr(),
                    blockStreaks, html.Hr(),
                    blockCurrency, html.Hr(),
                    blockBonds, html.Hr(),
                    blockEarnings, html.Hr(),
                    blockETFReturns, html.Hr(),
                ],
            style={"max-width": "1500px"},
            )
    return layout


# Define callback to update graph ---------------------------------------------------------------------------


@callback(Output("ReturnsHistogramm", "figure"), 
          Output("SectorReturns", "figure"), 
          Output("AllvsSPY", "figure"), 
          Input("my-table", "virtualRowData"),
          Input(component_id='radioReturnPeriod', component_property='value')
)
def display_cell_clicked_on(vdata, needed_column):
    if vdata:
        dff = pd.DataFrame(vdata)
    else:
        dff = sp500_small

    if needed_column == 'chg1':
        graph_title = f"Distribution of ystrd ({previous_day.strftime('%Y-%m-%d')}) returns"
    elif needed_column == 'chg5':
        graph_title = f"Distribution of last 5 days returns"
    else:
        graph_title = "Distribution of last 30 days returns"

    hstgrRet = px.histogram(dff, x=needed_column, nbins = 10, title=graph_title)
    hstgrRet.update_layout({'legend_title_text': '','transition_duration':500},
                            title_font_size = 14,title_x=0.5,title_y=0.97,
                            xaxis_title = None,yaxis_title = None,                            
                            legend=dict(orientation = "v",yanchor="top",y=0.9,xanchor="left",x=-0.3),
                            margin=dict(b = 0, l = 0, r = 0, t = 35),
                            height = 300,
                            )


    sector_returns_df = dff.groupby('Sector')[['chg30','chg5','chg1']].mean().reset_index()
    sector_returns_df['Sec'] = sector_returns_df['Sector'].map(utils.Sector_Dict) # Mapping code to text
    sector_returns_df['Desc'] = sector_returns_df['Sector'].map(utils.Group_Dict) # Mapping code to text


    barSecReturn = px.bar(sector_returns_df, x="Sec", y=["chg30", "chg5", "chg1"], 
                color_discrete_sequence=["PowderBlue","CornflowerBlue", "RoyalBlue"],
                title="Sector % Performance", hover_data=['Desc'])
    barSecReturn.update_layout({'legend_title_text': '','transition_duration':500},
                            title_font_size = 14,title_x=0.5,title_y=0.97,
                            xaxis_title = None,yaxis_title = None, xaxis = dict(tickfont = dict(size=10), tickangle = 0),
                            legend=dict(orientation = "v",yanchor="top",y=1.2,xanchor="right",x=1),
                            margin=dict(b = 0, l = 0, r = 5, t = 35),
                            height = 300,
                            )


    aggreg_returns_df = dff[['chg30','chg5','chg1']].mean()
    aggreg_returns_df = spy_df.set_index('symbol').subtract(aggreg_returns_df.T, axis=1)
    heatSelectionvsSPY = px.imshow(aggreg_returns_df[['chg30','chg5','chg1']].T.values, 
                                    # x=country_df.Desc, 
                                    y=['chg30','chg5','chg1'],
                                    color_continuous_scale = px.colors.diverging.RdYlGn,
                                    title="Selected vs SPY",
                                )
    heatSelectionvsSPY.update(layout_coloraxis_showscale=False)
    heatSelectionvsSPY.update_layout(title_font_size = 14,title_x=0.5,title_y=0.97,
                            margin=dict(b = 0, l = 0, r = 5, t = 50),
                            height = 300,
                            )


    return hstgrRet, barSecReturn, heatSelectionvsSPY



@callback(
    Output('ONReturns', 'figure'),
    Input(component_id='radioDayReturnSort', component_property='value')
)
def build_graph(needed_column):

    global sp500

    graph_title = f"% Performance for {needed_column}<br><sup>{market_case[needed_column]}</sup>"

    sp500 = sp500.sort_values(by=needed_column, ascending=False) # sort by capitalization

    small = pd.concat([sp500.nlargest(20,needed_column),sp500.nsmallest(20,needed_column)])

    barONReturn = px.bar(small, x='symbol', y=["PRE", "DAY", "POST"], 
                color_discrete_sequence=["PowderBlue","CornflowerBlue", "RoyalBlue"],
                title = graph_title,
                hover_data={'variable': False, 'value': False, 'Name':True,'Group':True,})
    barONReturn.update_layout({'legend_title_text': '','transition_duration':500},
                            title_font_size = 14,title_x=0.5,title_y=0.97,
                            xaxis_title = None,yaxis_title = None, xaxis = dict(tickfont = dict(size=10), tickangle = 270),
                            legend=dict(orientation = "h",yanchor="bottom",y=0.8,xanchor="right",x=1),
                            margin=dict(b = 0, l = 0, r = 0, t = 35),
                            )
    sp500 = sp500.sort_values(by='Cap', ascending=False) # sort by capitalization


    return barONReturn



@callback(
    Output('ETFReturns', 'figure'),
    Input(component_id='radioETFDayReturnSort', component_property='value')
    
)
def build_graph(needed_column):

    global etf_df

    graph_title="ETF Overview <br><sup>% Performance of non-leverage, long ETFs, traded in USA</sup>"

    '''
    if needed_column == 'chg1':
        graph_title = f"% ETF Perfo sorted by ystrd ({previous_day.strftime('%Y-%m-%d')}) returns"
    elif needed_column == 'chg5':
        graph_title = "% ETF Perfo sorted by last 5 days returns"
    else:
        graph_title = "% ETF Perfo sorted by last 30 days returns"
    '''
        
    etf_df = etf_df.sort_values(by=needed_column, ascending=False) # sort by capitalization
    small_etf = pd.concat([etf_df.nlargest(30,needed_column),etf_df.nsmallest(30,needed_column)])
    small_etf = small_etf.merge(ETF_mapping_df[['symbol','Fund Name','Fund Type']],on=['symbol'],how ='inner')
    small_etf = small_etf.sort_values(by=needed_column, ascending=False) # sort by capitalization

    barETFReturn = px.bar(small_etf, x='symbol', y=needed_column, 
                color='Fund Type',
                title = graph_title,
                hover_data={'Fund Name':True,'Fund Type':True})
    barETFReturn.update_layout({'legend_title_text': '','transition_duration':500},
                            title_font_size = 16,title_x=0.5,title_y=0.97,
                            xaxis_title = None,yaxis_title = None, xaxis = dict(tickfont = dict(size=10), tickangle = 270),
                            legend=dict(orientation = "v",yanchor="bottom",y=0.02,xanchor="left",x=0.01),
                            margin=dict(b = 0, l = 0, r = 0, t = 35),
                            )
    return barETFReturn


@callback(
    Output('BondChg', 'figure'),
    Input('hidden-div', 'children')
)
def populate_yield_graph(_):
    bonds_10Y_df = utils.get_bond_data()

    bar10YYieldChange = px.bar(bonds_10Y_df, x="bond", y=["chg180", "chg30", "chg5"], 
                color_discrete_sequence=["PowderBlue","CornflowerBlue", "RoyalBlue"],
                title="10Y Yield Chg in pp",)
    bar10YYieldChange.update_layout({'legend_title_text': '','transition_duration':500},
                            title_font_size = 14,title_x=0.5,title_y=0.97,
                            xaxis_title = None,yaxis_title = None, xaxis = dict(tickfont = dict(size=10), tickangle = 0),
                            legend=dict(orientation = "h",yanchor="bottom",y=1,xanchor="center",x=0.5),
                            # legend=dict(orientation = "v",yanchor="top",y=0.7,xanchor="right",x=1.2),
                            margin=dict(b = 0, l = 0, r = 5, t = 50),
                            height = 300,
                            )
    return bar10YYieldChange
