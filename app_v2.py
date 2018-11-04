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
from dash.dependencies import State

import json

external_stylesheets = ['bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)


##############################################################################
# Define data sources
##############################################################################


#df = pd.read_csv('dataframe_tmp.csv',dtype={'title': str,'content':str})

df = pd.read_csv('jacobs_corpus_body_labeled.csv')

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
    '''
    Here we would call a more sophisticaed function... TextRank
    '''
    return '.'.join(text.split('.')[:2])

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
df = df.sort_index()

sentiment_data = create_sentiment_score(df)


# used by multiple functions
event_labels = ['Profit', 'Dividend', 'MergerAcquisition', 'SalesVolume', 'BuyRating', 'QuarterlyResults',
                'TargetPrice', 'ShareRepurchase', 'Turnover', 'Debt']




##############################################################################
# Build layout
##############################################################################

app.layout = html.Div(children=[
    html.H1(children='News Analytics Dashboard',style={'textAlign': 'center'}),

    dcc.Tabs(id="tabs", value='tab-1', children=[
        dcc.Tab(label='Articles', value='tab-1'),
        dcc.Tab(label='Article View', value='tab-2'),
    ]),


    # initial selection block
    html.Div(className='row',children=[
        html.Div(children='Select Industry'), #, style={'float': 'left', 'clear': 'both', 'text-align': 'center'}),
        html.Div([
        dcc.Dropdown(
            options=[
                {'label': 'Retail and Trade', 'value': 'RT'},
                {'label': 'Manufacturing and Wholesale', 'value': 'MW'},
            ],
            value='RT'
        )],className='six columns'),

        html.Div(children='Select companies'), #, style={'float': 'left', 'clear': 'both', 'text-align': 'center'}),
        html.Div([
        dcc.Dropdown(
            options=[
                {'label': 'Coca Cola', 'value': 'CC'},
                {'label': 'Sony Ericson', 'value': 'SE'},
                {'label':'All','value':'All'},
            ],
            value='All',
        multi=True
        )],className='six columns'),


        html.Div(children='Pick your date range'),#, style={'float': 'left', 'clear': 'both', 'text-align': 'center'}),
        html.Div([
        dcc.DatePickerRange(id='date-picker-range',
        start_date=df.index[0],
        end_date = df.index[-1]),
        ],className='six columns'),


        dcc.Checklist(
            options=[
                {'label': 'CA', 'value': 'CANEWS'},
                {'label': 'International', 'value': 'INTNEWS'}
            ],
            values=['CANEWS']
        )

        ],style={'padding': 20,'margin': 5,'borderRadius': 5,'border': 'thin lightgrey solid','width' : '400'}),#,style={'float':'left','clear':'both'}),


    # diagram + summary block
    html.Div(className="row",children=[

        # document summaries
        html.Div(className="six columns",children=[

            html.H4(children='Article Summaries',style={'textAlign': 'center'}),

            html.Div(id='ifr',
                className="six columns",style={'padding': 20,'margin': 5,'borderRadius': 5,
                   'border': 'thin lightgrey solid','overflowY': 'scroll','height': 800,'width':700,'float':'right','clear':'both'}),
        ]),

        # all diagrams
        html.Div(className="six columns",children=[

            # events
            html.Div(children=[
                html.Div(children='Event Detector'),
                dcc.Dropdown(
                    id="events_time_dropdown",
                    options=[
                        {'label': 'last day', 'value': 'day'},
                        {'label': 'last week', 'value': 'week'},
                        {'label': 'last month', 'value': 'month'},
                        {'label':'all','value':'all'}
                    ],
                    value='week'
                ),
                dcc.Graph(id="events_graph"),
            ], style={'padding': 20, 'margin': 5, 'borderRadius': 5, 'border': 'thin lightgrey solid',
                      'width': '40%', 'height': '20%'}),# 'float': 'left', 'clear': 'both'}),

            #insights diagrams
            html.Div(className='row',children=[
            html.Div(children='Insights'),#, style={'float': 'left', 'clear': 'both', 'text-align': 'center'}),
            dcc.Dropdown(
            id="insights_dropdown",
            options=[
                {'label': 'Number of articles', 'value': 'num_articles'},
                {'label': 'Sentiment score', 'value': 'sentiment'},
            ],
            value='num_articles'
            ),

            dcc.Graph(id="insights_graph")

            ],style={'padding': 20,'margin': 5,'borderRadius': 5,'border': 'thin lightgrey solid','display': 'inline-block',
                     'width': '40%','height': '20%'}),

        ]),


    ]),


    # full document view


    html.Div(className="twelve columns",children=[

        html.H4(children='Article Fulltext',style={'textAlign': 'center'}),

        html.Div(id='full_document',children=[

            html.Div(children=[

                html.Div(children=[

                    html.H6(children='Labels'),
                    dcc.Checklist(
                        id='articlechecklist',
                        options=[{'label': label, 'value': label} for label in event_labels],
                        values=[]
                    ),
                    html.Button('Update labels', id='articlechecklistbutton'),
                    html.Button('Previous article', id='prevarticlebutton'),
                    html.Button('Next article', id='nextarticlebutton'),
                ]),
                html.Div(children=[
                    html.Iframe(id='fullarticleiframe',
                                style={'overflowY': 'scroll', 'height': 400, 'width': 1000,
                                       'overflow': '-moz-scrollbars-vertical',
                                       'padding': 5, 'margin': 0, 'borderRadius': 0, 'border': 'thin lightgrey solid'
                                })
                ])

            ],className="twelve columns",style={'height': 200, 'width': 1300})

            ])
        ]),

    html.P(id='placeholder'),
    html.Div(id='intermediate-value', style = {'display':'none'}),
    html.P(id='updated_df', children=[],style={'display': 'none'})

    #html.P(id='intermediate-value')

## end of layout
    ])



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

def create_events_figure(df,xlimrange=None):



    traces = [go.Scatter(
        x=df[event][df[event]==1].resample('1D', how='count').index,
        y=df[event][df[event]==1].resample('1D', how='count'),
        #line={"color": "rgb(53, 83, 255)"},
        mode="markers",  # lines,markers
        #hoverinfo="text",
        name=event
    ) for event in event_labels]
    layout = go.Layout(
        xaxis={'title': 'publication date','range':xlimrange},
        yaxis={'title': '# of detections'},  # , 'range': [20, 90]},
        margin={'l': 40, 'b': 40, 't': 10, 'r': 10},
        # legend={'x': 0, 'y': 1},
        hovermode='closest'
    )
    return {'data': traces, 'layout': layout}


def figure_num_articles(xlimrange=None):

    print('done')
    traces = [go.Scatter(
        x=df['text'].resample('7D', how='count').index,
        y=df['text'].resample('7D', how='count'),
        line={"color": "rgb(53, 83, 255)"},
        mode="lines",  # lines,markers
        name="500 Index Fund Inv"
    )]
    layout = go.Layout(
        xaxis={'title': 'publication date','range':xlimrange},
        yaxis={'title': '# of articles'},  # , 'range': [20, 90]},
        margin={'l': 40, 'b': 40, 't': 10, 'r': 10},
        # legend={'x': 0, 'y': 1},
        hovermode='closest'
    )
    return {'data': traces, 'layout': layout}



##############################################################################
# Updates (callbacks)
##############################################################################



@app.callback(
    Output(component_id='insights_graph', component_property='figure'),
    [Input(component_id='insights_dropdown', component_property='value'),
     Input('date-picker-range', 'start_date'),
     Input('date-picker-range', 'end_date')])
def update_insight_figure(input_val,start_date,end_date):
    '''Shows the number of articles per week'''
    if input_val == 'num_articles':
        return figure_num_articles(xlimrange=[start_date,end_date])
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
            yaxis={'title': 'sentiment score'}, #, 'range': [20, 90]},
            margin={'l': 40, 'b': 40, 't': 10, 'r': 10},
            #legend={'x': 0, 'y': 1},
            hovermode='closest'
        )
        return {'data':traces,'layout':layout}


@app.callback(
    Output(component_id='events_graph', component_property='figure'),
    [Input(component_id='events_time_dropdown', component_property='value'),
     Input('date-picker-range', 'start_date'),
     Input('date-picker-range', 'end_date')])
def update_insight_figure(input_val,start_date,end_date):
    '''subselects events diagram'''

    if input_val == 'day':
        first_date = df.index[-1] - pd.Timedelta(days=1)
        return create_events_figure(df[first_date:])
        # extract the events from the last day
    elif input_val =='week':
        first_date = df.index[-1] - pd.Timedelta(days=7)
        return create_events_figure(df[first_date:])
    elif input_val =='month':
        first_date = df.index[-1] - pd.Timedelta(days=30)
        return create_events_figure(df[first_date:])
    elif input_val == 'all':
        return create_events_figure(df)
    return create_events_figure(df[first_date:],xlimrange=[start_date, end_date])


number_article_summary = 0

@app.callback(
    dash.dependencies.Output('ifr', 'children'),
    [Input('insights_graph', 'clickData'),
     Input('events_graph', 'clickData'),
     ])
def update_article_summary(clickData_insights,clickData_events):

    global number_article_summary

    print('hover data',clickData_events)

    # subselection for day


    # for event graph
    try:
        rs = df.resample('1D')
        print('try to find ',clickData_events['points'][0]['x'])

        event_type = event_labels[clickData_events['points'][0]['curveNumber']]
        print('found ',event_type)
        #subselection by time range
        df_subset = rs.get_group(pd.to_datetime(clickData_events['points'][0]['x']))
        #subselection by event type
        print('does htis work?')
        df_subset = df_subset[df_subset[event_type]==1]

        print('select subset')
    except:
        df_subset = df

    number_article_summary = len(df_subset)

    return [html.Div(children=[
            html.Iframe(id='ifr'+str(i),srcDoc=render_doc_summary(title=df_subset['title'].iloc[i],summary=create_fake_summary(df_subset['text'].iloc[i]),lines=['as'],labels=['M&A']),
style = {'overflowY': 'scroll', 'height': 80,'width':500,'overflow': '-moz-scrollbars-vertical',
         'padding': 5, 'margin': 0, 'borderRadius': 0, 'border': 'thin lightgrey solid'}),
            # for ranking, document summary stats
            html.Div(children=['# Sent. :'+str(len(df_subset['text'].iloc[i].split('.'))),' ',df_subset['date'].iloc[i],
                               html.Button('Read', id='button'+str(i))
                               ],
                     style={'padding': 20, 'margin': 5, 'borderRadius': 5,'border': 'thin lightgrey solid',
                            'float': 'right', 'clear': 'both','width':70})
        ],className="twelve columns",style={'float':'left','clear':'both'})
       for i in range(len(df_subset))]


def subselect_docs(df,clickData):
    rs = df.resample('1D')
    print('try to find ',clickData['points'][0]['x'])

    event_type = event_labels[clickData['points'][0]['curveNumber']]
    print('found ',event_type)
    #subselection by time range
    df_subset = rs.get_group(pd.to_datetime(clickData['points'][0]['x']))
    #subselection by event type
    print('does htis work?')
    df_subset = df_subset[df_subset[event_type]==1]
    return df_subset


# update the article text view
@app.callback(
    dash.dependencies.Output('intermediate-value','children'),
    [Input('events_graph','clickData'),
     Input('prevarticlebutton','n_clicks'),
     Input('nextarticlebutton','n_clicks'),
    Input('updated_df', 'children')])
def preprocess_events_article(clickData_events,n_clicks_prev,n_clicks_next,updated_df_json):
    '''

    '''

    # decide here to use global df variable or updated_df_Json
    if len(updated_df_json)>0:
        df_subset = pd.read_json(updated_df_json[0], orient='split')
        print('consider updated df in intermediate step')
    else:
        df_subset = df
    #print('updatejson',updated_df_json)

    #data = json.loads(json_data)
    #print(data)

    #print('data: ',data)
    #data1={}
    #for key,val in zip(data['index'],data['data']):
    #    data1[key]=val

    #print(data)

    print('next document')
    try:
        df_subset = subselect_docs(df_subset,clickData_events)
    except:
        # if no document has been selected
        print('problem, clickdata empty')
        df_subset = df_subset




    if n_clicks_prev is None:
        n_clicks_prev=0
    if n_clicks_next is None:
        n_clicks_next=0
    n_clicks = n_clicks_next-n_clicks_prev
    print(n_clicks_prev,n_clicks_next,n_clicks)
    len_data_entries = len(df_subset)
    if n_clicks>=len_data_entries:
        n_clicks = len_data_entries
    idx = n_clicks - 1
    return df_subset.iloc[idx].to_json(orient='split', date_format='iso')
    #else:
        # no click yet so use first item
     #   return df_subset.iloc[0].to_json(orient='split', date_format='iso')





# update the article text view
@app.callback(
    Output('fullarticleiframe', 'srcDoc'),
    [Input('intermediate-value', 'children')
     ])
def update_full_articleiframe(json_data):
    '''

    '''
    data = json.loads(json_data)
    #print('data: ',data)
    data1={}
    for key,val in zip(data['index'],data['data']):
        data1[key]=val

    return render_document(title=data1['title'],body=data1['text'])


# update the article labels
@app.callback(
    dash.dependencies.Output('articlechecklist', 'values'),
    [Input('intermediate-value', 'children')
     ])
def update_full_articleiframe(json_data):
    '''

    '''
    data = json.loads(json_data)
    #print('data: ',data)
    data1={}
    for key,val in zip(data['index'],data['data']):
        data1[key]=val

    active_labels=[]
    for label in event_labels:
        if data1[label] == 1:
            active_labels.append(label)
    print('found labels ',active_labels)

    return active_labels

# update dataframe with labels submitted
@app.callback(
    Output('updated_df', 'children'), #updated_df, placeholder
    [Input('articlechecklistbutton', 'n_clicks')],
     [State('intermediate-value', 'children'),
     State('articlechecklist','values')])
def update_article_label(n_clicks,json_data,checklistvals):
    print('executed!! ')
    print('number clicks:',n_clicks)
    print('checklinkvals',checklistvals)
    data = json.loads(json_data)
    data1={}
    for key,val in zip(data['index'],data['data']):
        data1[key]=val
    data = data1
    data_id = data['file_id']

    print('found id ',data_id)

    # update the labels

    df_mod = df.copy()

    df_mod[df_mod['file_id'] == data_id][event_labels] = 0 #set all 0 first
    #print('old',df[df['file_id'] == data_id][event_labels])
    for label in checklistvals:
        df_mod[df_mod['file_id']==data_id][label] = 1

    return [df_mod.to_json(orient='split', date_format='iso')]
    #print('new', df[df['file_id'] == data_id][event_labels])

    # find entry and update in data frame
    #return data







#@app.callback(
#    dash.dependencies.Output('full_document', 'children'),
#    [Input('events_graph', 'clickData')])



#def update_full_doc(args**):
#
#
#    return ender_document




if __name__ == '__main__':
    app.run_server(debug=True)