import dash 
from dash import html, dcc
import dash_bootstrap_components as dbc
from navbar import create_navbar


NAVBAR = create_navbar()

# To use Font Awesome Icons
FA621 = "https://use.fontawesome.com/releases/v6.2.1/css/all.css"

# Define app ----------------------------------------------------------------------------------
dash_app = dash.Dash(
    __name__, 
    external_stylesheets=[
        dbc.themes.BOOTSTRAP,# theme should be written in CAPITAL letters; list of themes https://www.bootstrapcdn.com/bootswatch/
        FA621,  # Font Awesome Icons CSS
    ], 
    meta_tags=[{'name': 'viewport', # this thing makes layout responsible to mobile view
                'content': 'width=device-width, initial-scale=1.0'}],
    use_pages=True    
                )
dash_app.title = "Vilni Market Dashboard" # this puts text to the browser tab


dash_app.layout = dcc.Loading(  # <- Wrap App with Loading Component
    id='loading_page_content',
    children=[
        html.Div(
            [
                NAVBAR,
                dash.page_container
            ]
        )
    ],
    color='primary',  # <- Color of the loading spinner
    fullscreen=True  # <- Loading Spinner should take up full screen
)

app = dash_app.server

# Run -----------------------------------------------------------------------------------------
if __name__ == "__main__":
    dash_app.run_server(debug=True)