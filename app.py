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


##############################################################################
# Define data sources
##############################################################################


#df = pd.read_csv('dataframe_tmp.csv',dtype={'title': str,'content':str})

df = pd.read_csv('jacobs_corpus_body.csv')

#df = df.sample(10).reset_index()

#print(df.columns)
#print(df['content'])


text = 'Hello world. I like you. Good job.'


##############################################################################
# Define preprocessor functions
##############################################################################

def render_document(title='', snippet='', body='', art='', lines=[], colors=['red', 'blue'], labels=[]):
    '''
    Renders news articles and colors substrings within lines

    '''
    document = ''
    line_br = "<br>"
    if len(title) > 0:
        document += ("<h3> <center>" + title + " </center></h3>" + line_br)
        document += line_br
    if len(body) > 0:
        document += ('<p>' + snippet + ' ' + body + '</p>')
    if len(art) > 0:
        document += ('<p>' + art + '</p>')

    for i, line in enumerate(lines):
        document = re.sub(line, "<span style='color:" + colors[i] + "'><b>" + line + "<b>[" + labels[i] + "]</span>",
                          document)
    return document

def render_doc_summary(title='', summary='', lines=[], colors=['red', 'blue'], labels=[]):
    '''
    Renders news articles and colors substrings within lines

    '''
    document = '<b>'+title+'</b>'+'<br>' + summary
    #document = "<h3> <center>" + title + " </center></h3>" + summary

    for i, line in enumerate(lines):
        document = re.sub(line, "<span style='background-color:" + colors[i] + "'><b>" + line + "<b>[" + labels[i] + "]</span>",
                          document)
    return document

def create_fake_summary(text):
    return '.'.join(df['text'][0].split('.')[:2])

def index_to_time(df_new):
    def time_stamp(val):
        try:
            return pd.to_datetime(val,format='%d-%m-%Y')
        except:
            #try:
            #    return pd.to_datetime(val,format='%d-%m-%Y')
            print('error with ',val)
            return np.nan
    df_new['time'] = df_new['date'].apply(time_stamp)
    df_new = df_new.dropna()
    #df_new.index = pd.DatetimeIndex(df_new['time'])
    df_new['time_new'] = df_new["time"].apply( lambda df : datetime.datetime(year=df.year, month=df.month, day=df.day))
    df_new.set_index(df_new["time_new"],inplace=True)
    return df_new

def create_sentiment_score(df):
    def sent(val):
        return len(val.split(' '))

    df['sentiment_score'] = df['text'].apply(sent)
    return df


df = df.dropna()
df = index_to_time(df)

sentiment_data = create_sentiment_score(df)


##############################################################################
# Define plotting functions
##############################################################################

def num_articles_figure():
    '''
    Shows the number of articles per week
    :return:
    '''
    return go.Scatter(
        x=df['text'].resample('7D', how='count').index,
        y=df['text'].resample('7D', how='count'),
        line={"color": "rgb(53, 83, 255)"},
        mode="lines", #lines,markers
        name="500 Index Fund Inv"
    )

def sentiment_score_figure():
    return go.Scatter(
        x=df['sentiment_score'].resample('7D', how='count').index,
        y=df['sentiment_score'].resample('7D', how='count'),
        line={"color": "rgb(53, 83, 255)"},
        mode="lines", #lines,markers
        name="500 Index Fund Inv"
    )

##############################################################################
# Build layout
##############################################################################

app.layout = html.Div(children=[
    html.H1(children='News Analytics Dashboard'),

    html.Div(children='''
        Dash: A web application framework for Python.
    '''),

    html.Div(children='Select Industry', style={'float': 'left', 'clear': 'both', 'text-align': 'center'}),
    dcc.Dropdown(
        options=[
            {'label': 'Retail and Trade', 'value': 'RT'},
            {'label': 'Manufacturing and Wholesale', 'value': 'MW'},
        ],
        value='RT'
    ),
    html.Div(children='Select companies', style={'float': 'left', 'clear': 'both', 'text-align': 'center'}),
    dcc.Dropdown(
        options=[
            {'label': 'Coca Cola', 'value': 'CC'},
            {'label': 'Sony Ericson', 'value': 'SE'},
        ],
        value='',
        multi=True
    ),

    html.Div(children='Pick your date range', style={'float': 'left', 'clear': 'both', 'text-align': 'center'}),
    dcc.DatePickerRange(
        id='date-picker-range',
        start_date=dt(1997, 5, 3),
        end_date_placeholder_text='Select a date!'
    ),

    html.Div(children='Insights', style={'float': 'left', 'clear': 'both', 'text-align': 'center'}),
    dcc.Dropdown(
    id="insights_dropdown",
    options=[
        {'label': 'Number of articles', 'value': 'num_articles'},
        {'label': 'Sentiment score', 'value': 'sentiment'},
    ],
    value='num_articles'
    ),

    dcc.Graph(id="insights_graph"),



    html.Div(children=''' ***Document Summaries***
    ''',style={'float':'left','clear':'both','text-align': 'center'}),

    html.Div(id='ifr',
        className="six columns",style={'padding': 20,'margin': 5,'borderRadius': 5,
                   'border': 'thin lightgrey solid','overflowY': 'scroll','height': 500,'float':'left','clear':'both'}),

# ,style={'overflowY': 'scroll', 'height': 50},className="six columns"




    ## end of layout
    ])


##############################################################################
# Updates (callbacks)
##############################################################################

@app.callback(
    Output(component_id='insights_graph', component_property='figure'),
    [Input(component_id='insights_dropdown', component_property='value')])
def update_insight_figure(input_val):
    '''Shows the number of articles per week'''
    if input_val == 'num_articles':
        print('done')
        traces = [go.Scatter(
            x=df['text'].resample('7D', how='count').index,
            y=df['text'].resample('7D', how='count'),
            line={"color": "rgb(53, 83, 255)"},
        mode="lines", #lines,markers
        name="500 Index Fund Inv"
        )]
        layout = go.Layout(
            xaxis={'title': 'publication date'},
            yaxis={'title': '# of articles'}, #, 'range': [20, 90]},
            margin={'l': 40, 'b': 40, 't': 10, 'r': 10},
            #legend={'x': 0, 'y': 1},
            hovermode='closest'
        )
        return {'data':traces,'layout':layout}
    else:
        print('else')
        traces = [go.Scatter(
            x=sentiment_data['sentiment_score'].resample('1D', how='count').index,
            y=sentiment_data['sentiment_score'].resample('1D', how='count'),
            line={"color": "rgb(53, 83, 255)"},
            mode="lines",  # lines,markers
            name="500 Index Fund Inv"
        )]
        layout = go.Layout(
            xaxis={'title': 'publication date'},
            yaxis={'title': 'sentiment score    '}, #, 'range': [20, 90]},
            margin={'l': 40, 'b': 40, 't': 10, 'r': 10},
            #legend={'x': 0, 'y': 1},
            hovermode='closest'
        )
        return {'data':traces,'layout':layout}


@app.callback(
    dash.dependencies.Output('ifr', 'children'),
    [dash.dependencies.Input('insights_graph', 'hoverData')])
def update_article_summary(hoverData):
    print('hover data',hoverData)

    # subselection for day
    rs = df.resample('7D')
    print('try to find ',hoverData['points'][0]['x'])
    df_subset = rs.get_group(pd.to_datetime(hoverData['points'][0]['x']))

    return [html.Div(children=[
            html.Iframe(id='ifr'+str(i),srcDoc=render_doc_summary(title=df_subset['title'].iloc[i],summary=create_fake_summary(df_subset['text'].iloc[i]),lines=['as'],labels=['M&A']),
style = {'overflowY': 'scroll', 'height': 80,'width':500,'overflow': '-moz-scrollbars-vertical',
         'padding': 5, 'margin': 0, 'borderRadius': 0, 'border': 'thin lightgrey solid'}),
            # for ranking, document summary stats
            html.Div(children=['# Sent. :'+str(len(df_subset['text'].iloc[i].split('.'))),' ',df_subset['date'].iloc[i]],
                     style={'padding': 20, 'margin': 5, 'borderRadius': 5,'border': 'thin lightgrey solid',
                            'float': 'right', 'clear': 'both','width':70})
        ],className="twelve columns",style={'float':'left','clear':'both'})
       for i in range(len(df_subset))]

if __name__ == '__main__':
    app.run_server(debug=True)