

# -*- coding: utf-8 -*-
import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objs as go

import pandas as pd
import numpy as np
import datetime
from datetime import datetime as dt
import re
from dash.dependencies import Output
from dash.dependencies import Input


external_stylesheets = ['bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

app.layout = html.Div(children=[
    html.H1(children='News Analytics Dashboard',style={'textAlign': 'center'}),

    # initial selection block
    html.Div(children=[
        html.Div(children='Select Industry'), #, style={'float': 'left', 'clear': 'both', 'text-align': 'center'}),
        html.Div([
        dcc.Dropdown(
            options=[
                {'label': 'Retail and Trade', 'value': 'RT'},
                {'label': 'Manufacturing and Wholesale', 'value': 'MW'},
            ],
            value='RT',
            className='three columns'
        )]),

        ],className='row',style={'width' : '150'}),#,style={'float':'left','clear':'both'}),

    ])


if __name__ == '__main__':
    app.run_server(debug=True)