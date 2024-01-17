from dash import html, register_page, dcc, callback
import plotly.express as px
import dash_bootstrap_components as dbc # should be installed separately
from dash.dependencies import Input, Output, State

import pandas as pd
from github import Github
import io
import os
import datetime as dt
from dateutil.relativedelta import relativedelta
import random
import json

import utils

register_page(
    __name__,
    name='Vilni Watchlist',
    top_nav=True,
    path='/watchlist'
)

from alpaca.trading.client import TradingClient
from alpaca.data import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from alpaca.data.enums import Adjustment, DataFeed

# https://www.computerhope.com/htmcolor.htm#color-codes
colors = ['red', '#0000FF', '#4169E1', '#3090C7','#4EE2EC']  # List of colors


github_strat_token = os.getenv('github_strat_token')
dedicated_repo = os.getenv('dedicated_repo')

gh_csv_stocks = 'SP500.csv'
gh_csv_macro_events = 'upcoming_econ_events.csv'
gh_csv_ETFDB = 'ETF_ticker_mapping.csv'
gh_csv_ETF_constit = 'ETF_constit.csv'
gh_csv_watchlist = 'Trading Ideas.csv'


def to_gh(df, filename):
    print('Uploading to GH')
    df.to_csv(filename, index=False, sep = ";")
    with open(filename, encoding='utf8') as file:
        content = file.read()
    g = Github(github_strat_token)
    repo = g.get_user().get_repo(dedicated_repo)
    try:
        contents = repo.get_contents(filename)
        repo.update_file(contents.path, "committing files", content, contents.sha, branch="main")
        print(filename + ' UPDATED')
    except:
        repo.create_file(filename, "committing files", content, branch="main")
        print(filename + ' CREATED')


API_KEY_PAPER = os.environ['API_KEY_PAPER']
API_SECRET_PAPER = os.environ['API_SECRET_PAPER']
trading_client = TradingClient(API_KEY_PAPER, API_SECRET_PAPER) # dir(trading_client)
stock_client = StockHistoricalDataClient(API_KEY_PAPER, API_SECRET_PAPER)

today = trading_client.get_clock().timestamp
previous_day = today - pd.Timedelta('1D')

repository = Github(github_strat_token).get_user().get_repo(dedicated_repo)
trading_ideas = repository.get_contents(gh_csv_watchlist)
trading_ideas_df = pd.read_csv(io.StringIO(trading_ideas.decoded_content.decode()), sep = ";")

SP500_file = repository.get_contents(gh_csv_stocks)
sp500 = pd.read_csv(io.StringIO(SP500_file.decoded_content.decode()))
ALL_SYMBOLS = sp500.symbol.to_list()

ETF_mapping = repository.get_contents(gh_csv_ETFDB)
ETF_mapping_df = pd.read_csv(io.StringIO(ETF_mapping.decoded_content.decode()),sep=",")
ALL_ETFS = ETF_mapping_df.symbol.to_list()


def create_overview_chart():
    global trading_ideas_df
    tickers_lists = trading_ideas_df['Symbols'].apply(lambda x: x.replace(';', ',').replace(' ', '').split(','))
    all_tickers = [ticker.strip() for sublist in tickers_lists for ticker in sublist] # Flatten the list of lists
    unique_tickers = list(set(all_tickers))
    tickers_returns_df, case_watchlist = utils.download_data_daily(unique_tickers+['SPY'])
    tickers_returns_df = tickers_returns_df.sort_values('chg30',ascending=False)
    barSymbolsChange = px.bar(tickers_returns_df, x="symbol", y=["chg30", "chg5", "chg1"], 
                color_discrete_sequence=["PowderBlue","CornflowerBlue", "RoyalBlue"],
                title=f"Tickers % Performance as of {previous_day.strftime('%Y-%m-%d')}")
    barSymbolsChange.update_layout({'legend_title_text': '','transition_duration':500},
                            title_font_size = 14,title_x=0.5,title_y=0.97,
                            xaxis_title = None,yaxis_title = None, xaxis = dict(tickfont = dict(size=10), tickangle = 0),
                            legend=dict(orientation = "h",yanchor="bottom",y=-0.2,xanchor="center",x=0.5),
                            margin=dict(b = 50, l = 0, r = 5, t = 40),
                            height = 430,
                            )
    return barSymbolsChange



def generate_charts(row):
    symbol_list = row["Symbols"].replace(';', ',').replace(' ', '').split(',')
    idea = row["Idea"]
    start = dt.datetime.strptime(row["Date"], '%Y-%m-%d')
    
    color_map = {ticker: color for ticker, color in zip(['SPY']+symbol_list, colors)} # to make same ticker have same color on both charts

    bars_request_params = StockBarsRequest(symbol_or_symbols=symbol_list+['SPY'], 
                                           start=start, end=today, 
                                           timeframe=TimeFrame.Day, 
                                           adjustment= Adjustment.ALL,
                                           feed=DataFeed.SIP)
    daily_df = stock_client.get_stock_bars(bars_request_params).df
    daily_df = daily_df.reset_index()
    daily_df.timestamp = daily_df.timestamp.dt.date

    # Calculate daily return and cumulative return
    daily_df['Daily Return'] = daily_df.groupby('symbol')['close'].pct_change() + 1
    daily_df['Cumul Return'] = daily_df.groupby('symbol')['Daily Return'].cumprod()
    daily_df['Cumul Return'] = daily_df['Cumul Return'].fillna(1)
    
    # Generate line chart
    daily_fig = px.line(daily_df, x='timestamp', y='Cumul Return', 
                        color='symbol', color_discrete_map=color_map)
    daily_fig.update_traces(line_shape='linear')    
    daily_fig.update_layout({'legend_title_text': '','transition_duration':500},
                        title={'text': "cumul return",'y':0.95,'x':0.98,'xanchor': 'right','yanchor': 'top','font': {'size': 9,'color': "black"}},
                        xaxis_title = None,yaxis_title = None, xaxis = dict(tickfont = dict(size=10), tickangle = 0),
                        legend=dict(orientation = "h",yanchor="bottom",y=0.9,xanchor="left",x=0),
                        margin=dict(b = 50, l = 0, r = 0, t = 10),
                        height = 300,
                        )
    
    # get weekly data
    bars_request_params = StockBarsRequest(symbol_or_symbols=symbol_list+['SPY'], 
                                        start=start, end=today, 
                                        timeframe=TimeFrame(1, TimeFrameUnit.Week), # 'Day', 'Hour', 'Minute', 'Month', 'Week'
                                        adjustment= Adjustment.ALL,
                                        feed=DataFeed.SIP)
    weekly_df = stock_client.get_stock_bars(bars_request_params).df
    weekly_df = weekly_df.reset_index()
    weekly_df.timestamp = weekly_df.timestamp.dt.date

    # Calculate weekly returns
    weekly_df['Weekly Return'] = weekly_df.groupby('symbol')['close'].pct_change()
    weekly_df.dropna(inplace=True)

    # Generate bar chart
    weekly_fig = px.bar(weekly_df, x='timestamp', y='Weekly Return', 
                        color='symbol', color_discrete_map=color_map,
                        barmode='group')
    weekly_fig.update_layout({'legend_title_text': '','transition_duration':500},
                        title={'text': "weekly % change",'y':0.95,'x':0.99,'xanchor': 'right','yanchor': 'top','font': {'size': 9,'color': "black"}},
                        xaxis_title = None,yaxis_title = None, xaxis = dict(tickfont = dict(size=10), tickangle = 0),
                        showlegend=False,
                        margin=dict(b = 50, l = 0, r = 0, t = 10),
                        height = 300,
                        )

    return weekly_fig, daily_fig



control_panel = dbc.Row([
    dbc.Col([
            html.P(),
            dcc.Graph(id="StrategyOverviewGraph", figure = create_overview_chart(), config={'displayModeBar': False}),
            html.P(),
        ],
        width = 8
    ),
    dbc.Col([
        html.P(),
        dbc.Card(
            dbc.CardBody([
                dbc.Row(dbc.Label("Enter new idea to watch in the form below:"), className="mb-3"),
                dbc.Row([
                    dbc.Col(dbc.Input(id="input-id", placeholder="Idea name"), width=4),
                    dbc.Col(dbc.Input(id="input-source", placeholder="Source of idea"), width=8),
                ], className="mb-3"),
                dcc.Dropdown(id = 'dropStartegyAssetSelect',
                            options=[{'label': i, 'value': i} for i in ALL_SYMBOLS+ALL_ETFS],
                            multi=True,
                            className="mb-3"),
                dbc.Tooltip("Dropdown contains 600 companies and 1500 ETFs traded in USA", target="dropStartegyAssetSelect"),
                dbc.Input(id="input-desc", placeholder="Description of new idea", className="mb-3"),
                dbc.Row([
                    dbc.Col([
                        dcc.RadioItems(
                                id="radioStrategyStart",                                        
                                options = [
                                    {'label':'3 months','value':'3M'}, 
                                    {'label':'Now','value':'NOW'}], 
                                value = '3M', 
                                className="mb-1",
                                inline=True,
                                inputStyle={"margin-right": "7px","margin-left": "13px",},
                                ),
                        dbc.Tooltip("Selection how much of historical performance to download from Alpaca", target="radioStrategyStart")
                        ], width=8
                    ),
                    dbc.Col([
                        dbc.Button("Submit Idea", id="buttonAddIdea", color="primary", className="float-right"),
                        dbc.Tooltip("By pressing the button the csv file on the dedicated Github repo will be overwritten", target="buttonAddIdea")
                        ], width=4
                    )
                ], className="mb-1")

            ])
        ),
        html.P(),
        dbc.Card(
            dbc.CardBody([
                dbc.Row(dbc.Label("Remove not-needed idea:")),
                dbc.Row([
                    dbc.Col(
                        dcc.Dropdown(id = 'dropIdeaSelect',
                                    options=[{'label': i, 'value': i} for i in trading_ideas_df["Idea"].unique().tolist()],
                                    multi=False,
                                    className="mb-3"),
                        width=8
                    ),
                    dbc.Col(
                        dbc.Button("Remove Idea", id="buttonRemoveIdea", color="danger", className="float-right"),
                        width=4
                    ),
                ], className="mb-1"),

            ])
        ),
        html.P(),
    ])
])

output = []
row_content = []
output.append(control_panel)
for i, row in trading_ideas_df.iterrows(): # generate separate chart for every idea 
    row_content.append(html.Label(row["Idea"], style={'fontSize': 24,'marginLeft': '20px'}))
    row_content.append(html.Label(row["Description"], style={'fontSize': 15,'marginLeft': '20px'}))
    weekly_fig, daily_fig = generate_charts(row)
    chart_weekly = dcc.Graph(id=f'graphk-{i}-w', figure=weekly_fig, config={'displayModeBar': False})
    row_content.append(dbc.Col(chart_weekly))
    chart_daily = dcc.Graph(id=f'graphk-{i}-d', figure=daily_fig, config={'displayModeBar': False})
    row_content.append(dbc.Col(chart_daily))
    output.append(dbc.Row(row_content))
    row_content = []


def layout():
    layout = dbc.Container(children=output,style={"max-width": "1500px"})
    return layout



# Adding new idea to the list ---------------------------------------------------------------------------------------
@callback(
    Output("input-desc", "value"),
    Output("input-source", "value"),
    Output("input-id", "value"),
    Output("dropStartegyAssetSelect", "value"),
    Output("dropIdeaSelect", "options", allow_duplicate=True),
    Output("StrategyOverviewGraph", "figure"),
    Input("buttonAddIdea", "n_clicks"), # click button
    State('dropStartegyAssetSelect', 'value'), # selected tickers
    State("input-desc", "value"), # long description of strategy
    State("input-source", "value"), # where the idea comes from
    State("input-id", "value"), # strat name
    State("radioStrategyStart", "value"), # when strategy starts
    prevent_initial_call=True
)
def add_strategy(n, selected_tickers, desc, source, id, date_option):
    global trading_ideas_df
    if n:
        symbols = ', '.join(list(set(selected_tickers))) if selected_tickers else "" # concat all selected tickers

        if id in trading_ideas_df["Idea"].values: # if idea with such a name already exist
            id += str(random.randint(1, 100))

        if date_option == "3M":
            date = dt.datetime.now() - relativedelta(months=3)
        else:
            date = dt.datetime.now() - relativedelta(days=2)
        date_str = date.strftime("%Y-%m-%d")  
        # Add a new row using loc
        trading_ideas_df.loc[trading_ideas_df.index.max() + 1] = [date_str, id, desc, source,symbols]

        # refresh drop-down list of strategies
        dropIdeaSelect_options = [{'label': strategy, 'value': strategy} for strategy in trading_ideas_df["Idea"].unique().tolist()]
        print("Writhing to gh ...")
        to_gh(trading_ideas_df, gh_csv_watchlist)

    # Clear the input fields
    return "", "", "", [], dropIdeaSelect_options, create_overview_chart()



# Deleting Idea from the list ------------------------------------------------------------------------------------------
@callback(
    Output("dropIdeaSelect", "value"), # clearing dropdown
    Output("dropIdeaSelect", "options", allow_duplicate=True), # updating list
    Input("buttonRemoveIdea", "n_clicks"),
    State("dropIdeaSelect", "value"),
    prevent_initial_call=True
)
def delete_strategy(n, selected_strategy):
    global trading_ideas_df
    if n and selected_strategy:
        # Delete from dataframe
        trading_ideas_df = trading_ideas_df[trading_ideas_df["Idea"] != selected_strategy]
        to_gh(trading_ideas_df, gh_csv_watchlist)
        dropIdeaSelect_options = [{'label': strategy, 'value': strategy} for strategy in trading_ideas_df["Idea"].unique().tolist()]

    return None, dropIdeaSelect_options
