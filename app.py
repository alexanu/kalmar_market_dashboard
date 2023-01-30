import plotly.express as px
import dash
from dash import html
from dash import dcc
import dash_bootstrap_components as dbc # should be installed separately
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate


import Universes

import pandas as pd
import time
import datetime as dt
import random
import os

import alpaca

from alpaca.trading.enums import OrderSide, TimeInForce, AssetClass, AssetStatus, AssetExchange, OrderStatus, QueryOrderStatus, CorporateActionType, CorporateActionSubType
from alpaca.trading.requests import GetCalendarRequest, GetAssetsRequest, GetOrdersRequest, MarketOrderRequest, LimitOrderRequest, StopLossRequest, TrailingStopOrderRequest, GetPortfolioHistoryRequest, GetCorporateAnnouncementsRequest
from alpaca.data.requests import StockLatestQuoteRequest, StockTradesRequest, StockQuotesRequest, StockBarsRequest, StockSnapshotRequest, StockLatestBarRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from alpaca.data.enums import Adjustment, DataFeed, Exchange

from alpaca.trading.client import TradingClient
from alpaca.data import StockHistoricalDataClient
from alpaca.broker.client import BrokerClient

API_KEY_PAPER = os.environ["ApiKeyPaper"]
API_SECRET_PAPER = os.environ["ApiSecretPaper"]
# from Alpaca_config import *

trading_client = TradingClient(API_KEY_PAPER, API_SECRET_PAPER) # dir(trading_client)
stock_client = StockHistoricalDataClient(API_KEY_PAPER,API_SECRET_PAPER)
broker_client = BrokerClient(API_KEY_PAPER,API_SECRET_PAPER,sandbox=False,api_version="v2")


# Define app
dash_app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP], # theme should be written in CAPITAL letters; list of themes https://www.bootstrapcdn.com/bootswatch/
                meta_tags=[{'name': 'viewport', # this thing makes layout responsible to mobile view
                            'content': 'width=device-width, initial-scale=1.0'}]
                )
dash_app.title = "Alpaca Market Dashboard" # this puts text to the browser tab
app = dash_app.server


stocks = ["SPY", "GLD", "TLT"]

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

text0 = f'{pd.Timestamp.now(tz="EST").tz_localize(None).strftime("%H:%M")} CET'
text1 = f'Today is {pd.Timestamp.today().day_name()}, {pd.Timestamp.now(tz="CET").tz_localize(None).strftime("%b %d, %H:%M") } in Munich, which is {pd.Timestamp.now(tz="EST").tz_localize(None).strftime("%H:%M")} in New York'
if not clock.is_open:
    time_to_open = round((clock.next_open - clock.timestamp).total_seconds()/3600,1)
    already_closed_hours = round((dt.datetime.now()- tradingcal_till_moe[0].close).total_seconds() / 3600,1)
    text2 = 'Closed'
    text3 = f'Market is currently closed for already {already_closed_hours} hours. Will open in {time_to_open} hours'
else:
    time_to_close = round((clock.timestamp - clock.next_close).total_seconds()/3600,1)
    text2 = 'Open'
    text3 = f'Market is currently open. Will close in {time_to_close} hours'

text4 = f'There are {len(tradingcal_till_moe)-1} trading days till the month end and {len(tradingcal_till_qe)-1} - till end of quarter'



#------------------------------------------------------------------------------------------------------------------------------------

SPY = stock_client.get_stock_snapshot(StockSnapshotRequest(symbol_or_symbols=['SPY'], feed = DataFeed.SIP))
SPY_gain = (SPY['SPY'].daily_bar.close/SPY['SPY'].previous_daily_bar.close)-1
snap = stock_client.get_stock_snapshot(StockSnapshotRequest(symbol_or_symbols=Universes.TOP10_US_SECTOR, feed = DataFeed.SIP))
snapshot_data = {stock: [snapshot.latest_trade.price, 
                        snapshot.previous_daily_bar.close,
                        snapshot.daily_bar.close,
                        (snapshot.daily_bar.close/snapshot.previous_daily_bar.close)-1,
                        ]
                for stock, snapshot in snap.items() if snapshot and snapshot.daily_bar and snapshot.previous_daily_bar
                }
snapshot_columns=['price', 'prev_close', 'last_close', 'gain']
snapshot_df = pd.DataFrame(snapshot_data.values(), snapshot_data.keys(), columns=snapshot_columns)

now_time = SPY['SPY'].daily_bar.timestamp.strftime('%Y-%m-%d %H:%M:%S')
yesterday_close_time = SPY['SPY'].previous_daily_bar.timestamp.strftime('%Y-%m-%d %H:%M:%S')
returns_histogram = px.histogram(snapshot_df, x="gain")
returns_histogram.update_layout(
    title=f"Distribution of returns <br><sup>[Now({now_time}) vs Yesterday Close({yesterday_close_time})]</sup>",
    title_x=0.5,
    yaxis_title="Number of stocks",
    xaxis_title="Now vs Yesterday Close")
returns_histogram.add_vline(x=snapshot_df.gain.median(), line_dash = 'dash', line_color = 'firebrick')
returns_histogram.add_vline(x=SPY_gain)

#-------------------------------------------------------------------------------------------------------------------------------------



controls = dbc.Row( # always do in rows ...
    [
        dbc.Col( # ... and then split to columns
         children=[   
                    dbc.Row(
                        [
                            dbc.Col(
                                dbc.Label("\N{factory} Focus on a specific sector "), 
                                width=9,
                                ),
                            dbc.Col(
                                [
                                dbc.Button("\N{books}",id="hover-target3", color="link", n_clicks=0),
                                dbc.Popover(dbc.PopoverBody("Scope of sectors could be different for different emission scenario.\nScope of sectors covered by the tool is constantly growing."),id="hover3",target="hover-target3",trigger="hover"), 
                                ], width=2,
                                ),                           
                        ],
                        align="center",
                    ),
                    dcc.Dropdown(
                                id = 'universe',
                                clearable=False,
                                value=stocks[0], 
                                options=[{'label': c, 'value': c} for c in stocks],
                                placeholder="Select a sector"),       
                  
        ],
        ),
    ],
)


# Define Layout
app.layout = dbc.Container( # always start with container
                children=[
                    html.Hr(), # small space from the top
                    dbc.Row( # Time of market
                        [
                            dbc.Col( # Time
                                [
                                    html.H1(id="lbl_time",children = text0),
                                    html.Div(children=text1),                                    
                                ],
                                width = 3,
                            ),
                            dbc.Col( # Open / Closed
                                [
                                    html.H1(id="lbl_status",children = text2),
                                    html.Div(children=text3),                                    
                                ],
                                width = 3,
                            ),
                            dbc.Col( # Days till eom
                                [
                                    html.H1(id="lbl_eom",children = len(tradingcal_till_moe)-1),
                                    html.Div(children=text4),                                    
                                ],
                                width = 3,
                            ),
                            dbc.Col( # empty placeholder
                                [
                                    html.H1(id="lbl_place"),
                                    html.Div(),                                    
                                ],
                                width = 3,
                            ),

                        ],
                        justify='between', # for this to work you need some space left (in total there 12 columns)
                        align = 'center',
                    ),
                    html.Hr(),
                    dbc.Row(
                            [
                                dbc.Col(dcc.Graph(id="market_overview",figure=returns_histogram)), # big bubble graph
                                dbc.Col(dcc.Graph(id="graph12"),), # covered graph
                            ],
                        ),
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


# Define callback to update graph
@app.callback(
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
    return (
        fig, 
        fig,
    )

if __name__ == "__main__":
    dash_app.run_server(debug=True, host='0.0.0.0', port='80')