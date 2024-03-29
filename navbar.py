from dash import html
import dash_bootstrap_components as dbc

def create_navbar():
    navbar = dbc.NavbarSimple(
        children=[
            dbc.NavItem(
                dbc.NavLink(
                    [
                        html.I(className="fas fa-sliders"),  # Font Awesome Icon
                                                                # check here: https://fontawesome.com/icons/display-chart-up?f=classic&s=solid
                        " "  # Text beside icon
                    ],
                    href="https://github.com/alexanu",
                    target="_blank" # To make the link open in new tab
                )

            ),
            dbc.NavItem(
                dbc.NavLink(
                    [
                        html.I(className="fas   fa-display-chart-up"),  # Font Awesome Icon
                        " "  # Text beside icon
                    ],
                    href="/page2'",
                    # target="_blank" # To make the link open in new tab
                )

            ),
            dbc.DropdownMenu(
                nav=True,
                in_navbar=True,
                label="Menu",
                align_end=True,
                children=[  # Add as many menu items as you need
                    dbc.DropdownMenuItem("Home", href='/'),
                    # dbc.DropdownMenuItem(divider=True),
                    dbc.DropdownMenuItem("ETF Constituents", href='/ETF-Constit'),
                    dbc.DropdownMenuItem("Watchlist", href='/watchlist'),
                ],
            ),
        ],
        brand='Vilni Trading',
        brand_href="/",
        sticky="top",  # Uncomment if you want the navbar to always appear at the top on scroll.
        color="secondary",  # Change this to change color of the navbar e.g. "primary", "secondary" etc.
        dark=True,  # Change this to change color of text within the navbar (False for dark text)
    )

    return navbar