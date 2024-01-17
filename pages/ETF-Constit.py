from dash import html, register_page, dcc, callback
import plotly.express as px
import dash_bootstrap_components as dbc # should be installed separately
from dash.dependencies import Input, Output

import pandas as pd
from github import Github
import io

from Azure_config import *
import utils

register_page(
    __name__,
    name='Vilni Trading',
    top_nav=True,
    path='/ETF-Constit'
)

'''
from alpaca.trading.client import TradingClient
trading_client = TradingClient(ALPACA_API_KEY_PAPER, ALPACA_API_SECRET_PAPER) # dir(trading_client)
clock = trading_client.get_clock()
today = clock.timestamp
previous_day = today - pd.Timedelta('1D')


# ETF Constituents ---------------------------------------------------------------------------------------------------------
repository = Github(github_strat_token).get_user().get_repo(dedicated_repo)
SP500_file = repository.get_contents(gh_csv_stocks)
sp500 = pd.read_csv(io.StringIO(SP500_file.decoded_content.decode()))
ETF_mapping = repository.get_contents(gh_csv_ETFDB)
ETF_mapping_df = pd.read_csv(io.StringIO(ETF_mapping.decoded_content.decode()),sep=",")
All_ETF_groups = sorted(ETF_mapping_df[ETF_mapping_df['Constitut'] == "Yes"].Group.unique().tolist())
ETF_constit = repository.get_contents(gh_csv_ETF_constit)
ETF_constit_df = pd.read_csv(io.StringIO(ETF_constit.decoded_content.decode()),sep=",")
small_sp500 = sp500[sp500.symbol.isin(ETF_constit_df.symbol.unique().tolist())]
constit_symbols = small_sp500.symbol.to_list()
returns_df = utils.download_data_daily(constit_symbols)
small_sp500 = small_sp500.merge(returns_df[['symbol','chg30','chg5','chg1']],on=['symbol'],how ='outer')
small_sp500['Sec'] = small_sp500['Sector'].map(utils.Sector_Dict) # Mapping code to text


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

needed_col = 'chg5'
barConstitReturnAll = px.bar(small_sp500, x='symbol', y=needed_col, 
                        facet_row_spacing=0.03, facet_col_spacing=0.03, facet_col="Group", facet_col_wrap=2,
                        color='SCORE_VS_AVG', color_continuous_scale='RdYlGn',
                        # title = graph_title, 
                        hover_data={'Name':True,'Group':True, 'EarnScore':True}
                        )
barConstitReturnAll.update_layout({'legend_title_text': '','transition_duration':500},
                        title_font_size = 14,title_x=0.5,title_y=0.97,
                        xaxis_title = None,yaxis_title = None, xaxis = dict(tickfont = dict(size=10), tickangle = 270),
                        legend=dict(orientation = "h",yanchor="bottom",y=1,xanchor="center",x=0.5),
                        margin=dict(b = 0, l = 0, r = 0, t = 35),
                        )
# barConstitReturnAll.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1])) # adjusting headers

blockETFConstitReturns2 = dbc.Row(
                            [
                                dbc.Col([
                                    html.P(),
                                    dcc.Graph(figure=barConstitReturnAll, config={'displayModeBar': False}),
                                ])    
                            ]
                        )
'''




# fig.update_layout(autosize=False,width=1400,height=2000,)
# fig.update_xaxes(matches=None,showticklabels=True)
# fig.update_yaxes(matches=None,showticklabels=True)


def layout():
# dash_app.layout = dbc.Container( # always start with container
    layout = dbc.Container( # always start with container
                children=[
                    # dcc.Interval(id="interval",interval=1000,n_intervals=0), # for downloading a lot of data
                    html.Hr(), # small space from the top
                    # blockETFConstitReturns, html.Hr(),
                    # blockETFConstitReturns2, html.Hr(),

                ],
            style={"max-width": "1500px"},
            )
    return layout


# Define callback to update graph ---------------------------------------------------------------------------


'''
@callback(
        Output('ETFConstitReturns', 'figure'),
    [
        Input(component_id='radioETFConstitReturnSort', component_property='value'),
        Input('dropETFStrategies', 'value')
    ],
)
def build_ETF_constit(needed_column, Group_Name):
    global sp500

    title_part1 = f'Constitutes % Perf of [{Group_Name}] group sorted '
    if needed_column == 'chg1':
        graph_title = title_part1 + f"by ystrd ({previous_day.strftime('%Y-%m-%d')}) returns"
    elif needed_column == 'chg5':
        graph_title = title_part1 + "by last 5 days returns"
    else:
        graph_title = title_part1 + "by last 30 days returns"

    ETFs_in_Scope = ETF_mapping_df[ETF_mapping_df['Group'] == Group_Name].symbol.to_list()
    Constit_in_Scope = ETF_constit_df[ETF_constit_df.symbolETF.isin(ETFs_in_Scope)].symbol.unique().tolist()
    very_small_sp500 = small_sp500[small_sp500.symbol.isin(Constit_in_Scope)].sort_values(by=needed_column, ascending=False)
    very_small_sp500 = very_small_sp500.sort_values(by=needed_column, ascending=False)

    barConstitReturn = px.bar(very_small_sp500, x='symbol', y=needed_column, 
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


'''