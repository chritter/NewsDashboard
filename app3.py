
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

def layout():
    return html.Div([  # gene selector
        html.Label('Color by gene expression: ',
                   className='four columns offset-by-two'
                   ),
        dcc.Dropdown(
            options=[{'label':'value'}],
            value='',
            className='six columns'
        ),
    ],
        className='row',
    )

app.layout = layout()


if __name__ == '__main__':
    app.run_server(debug=True)
