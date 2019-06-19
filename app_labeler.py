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
import os
from dash.dependencies import Output
from dash.dependencies import Input
from dash.dependencies import State

import json

external_stylesheets = ['bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)




##############################################################################
# Define preprocessor functions
##############################################################################

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


def render_document(title='', snippet='', body='', art='', id='',date='',publisher='', lines=[], colors=[], labels=[]):
    '''
    Renders news articles and colors substrings within lines

    '''
    document = ''
    line_br = "<br>"
    if len(title) > 0:
        document += ("<h3> <center>" + title + " </center></h3>" + line_br)
        document += line_br
    if len(id)>0:
        document += ('ID: '+id)
    if len(date)>0:
        if len(id) > 0:
            document+=', '
        document += (date)
    if len(publisher)>0:
        if len(id)>0 or len(date)>0:
            document+=', '
        document += publisher
    if len(id)>0 or len(date)>0 or len(pubisher)>0:
        document += line_br
    if len(body) > 0:
        document += ('<p>' + snippet + ' ' + body + '</p>')
    if len(art) > 0:
        document += ('<p>' + art + '</p>')

    for i, line in enumerate(lines):
        #document = re.sub(line, "<span style='color:" + colors[i] + "'><b>" + line + "</b>[" + labels[i] + "]</span>",
         #                 document)
        document = re.sub(line, "<span style='color:" + colors[i] + "'>" + line + "<b>[" + labels[i] + "]</b></span>",
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




##############################################################################
# Define data sources & properties of the sources
##############################################################################


# assumes ['title'],,['text'],'date','file_id' as input fields

input_file = 'jacobs_corpus_body_labeled.csv'
df = pd.read_csv(input_file)

df_pred=pd.read_csv('jacobs_corpus.csv')


df = df.dropna()
df = index_to_time(df)
df = df.sort_index()
# for modified labels
df_mod = df.copy()

# events as they appear in the dataframe columns
event_labels = ['Profit', 'Dividend', 'MergerAcquisition', 'SalesVolume', 'BuyRating', 'QuarterlyResults',
                'TargetPrice', 'ShareRepurchase', 'Turnover', 'Debt']

event_colors_tmp = ['r','b','g','y','m','k','gray','blueviolet','firebrick','bisque','orange']
event_colors_tmp = ['red','gray','yellow','blue','navy','silver','aqua','Purple','olive','lime']
event_colors={}
for label,color in zip(event_labels,event_colors_tmp):
    event_colors[label] = color


current_file_id = None

##############################################################################
# Build layout
##############################################################################

app.layout = html.Div(children=[
    html.H1(children='News Label Dashboard',style={'textAlign': 'center'}),


    html.Div(className='row',children=[
    html.Div(className='row', children=[dcc.Graph(id="numdocs_graph")], style={'float': 'right', 'clear': 'both',
                                                                               'padding': 20, 'margin': 5,
                                                                                        'borderRadius': 5,
                                                                                        'border': 'thin lightgrey solid'}),

    # initial selection block
    html.Div(className='row',children=[



        html.Div(children='Select file'),  # , style={'float': 'left', 'clear': 'both', 'text-align': 'center'}),
        html.Div([
            dcc.Dropdown(
                id='file_dropdown',
                options=[{'label':f,'value':f} for f in sorted([f for f in os.listdir('.') if (os.path.isfile(f) and '.csv' in f)],key=lambda x: os.path.getmtime(x))],
                value='jacobs_corpus_body_labeled.csv' # sorted([f for f in os.listdir('.') if os.path.isfile(f)],key=lambda x: os.path.getmtime(x))[-1]
            )]),#, className='six columns'),


        html.Div(children='Select Event Type'), #, style={'float': 'left', 'clear': 'both', 'text-align': 'center'}),
        html.Div([
        dcc.Dropdown(
            id='event_dropdown',
            options=[{'label':ev,'value':ev} for ev in event_labels],
            value='MergerAcquisition'
        )],className='six columns'),


        html.Div(children='Pick your date range'),#, style={'float': 'left', 'clear': 'both', 'text-align': 'center'}),
        html.Div([
        dcc.DatePickerRange(id='date-picker-range',
        start_date=df.index[0],
        end_date = df.index[-1]),
        ],className='six columns'),
        # show number of articles in the time range selected


    ],style={'padding': 20,'margin': 5,'borderRadius': 5,'border': 'thin lightgrey solid','width' : '600'}),
    ]),

    # full document view
    html.Div(className="twelve columns",children=[

        html.Div(id='full_document',children=[

            html.Div(children=[

                html.Div(children=[

                    html.H4(children='Labels'),
                    dcc.Checklist(
                        id='articlechecklist',
                        options=[{'label': label, 'value': label} for label in event_labels],
                        values=[]
                    ),
                    html.Button('Update labels', id='updatelabelsbutton'),
                    html.Button('Previous article', id='prevarticlebutton'),
                    html.Button('Next article', id='nextarticlebutton'),
                    html.Button('Save to file', id='savetofilebutton'),
                ]),
                html.Div(children=[
                    html.Iframe(id='fullarticleiframe',
                                style={'overflowY': 'scroll', 'height': 600, 'width': 1000,
                                       'overflow': '-moz-scrollbars-vertical',
                                       'padding': 5, 'margin': 0, 'borderRadius': 0, 'border': 'thin lightgrey solid'
                                })
                ])

            ],className="twelve columns",style={'padding': 20, 'margin': 5,
                                                                                        'borderRadius': 5,
                                                                                        'border': 'thin lightgrey solid'})

            ])
        ]),

    # full document view
    html.Div(className="twelve columns", children=[
        html.Button('Retrain and predict', id='retrainbutton'),
    ],style={'padding': 20, 'margin': 5,'borderRadius': 5,'border': 'thin lightgrey solid'}),


    html.P(id='placeholder'),
    html.Div(id='intermediate-value', style = {'display':'none'}),
    html.P(id='updated_df', children=[],style={'display': 'none'}),
    html.P(id='updated_df2', children=[],style={'display': 'none'}),
    html.P(id='updated_df3', children=[], style={'display': 'none'}),
    html.P(id='updated_df4', children=[], style={'display': 'none'}),
    html.P(id='updated_df5', children=[], style={'display': 'none'})




## end of layout
    ])



##############################################################################
# Define plotting functions
##############################################################################

def figure_num_articles(xlimrange=None,event_type=None):

    #select event

    df_sub = df[df[event_type]==1]

    print('done')
    traces = [go.Scatter(
        x=df_sub['text'].resample('7D', how='count').index,
        y=df_sub['text'].resample('7D', how='count'),
        line={"color": "rgb(53, 83, 255)"},
        mode="lines",  # lines,markers
        name="500 Index Fund Inv"
    )]
    layout = go.Layout(
        xaxis={'title': 'publication date','range':xlimrange},
        yaxis={'title': '# of articles'},  # , 'range': [20, 90]},
        margin={'l': 40, 'b': 40, 't': 10, 'r': 10},
        height= 170,
        width= 600,
        hovermode='closest'
    )
    return {'data': traces, 'layout': layout}

# uses df_pred to get tagged sentences and provide them with color
def get_tagged_sentences(file_id):
    global df_pred,event_colors
    import ast
    df_selection = df_pred[file_id == df_pred['file_id']]
    sent_tmp = df_selection['sentences'].values
    lab_tmp = df_selection['label'].apply(ast.literal_eval).values
    #print('found df selection ',df_selection.shape)
    colors=[]
    sentences=[]
    labels=[]
    for i,lab in enumerate(lab_tmp):
        #print('search ',lab,event_colors)
        #print(type(lab))
        if type(lab) == int:
            continue
        if lab[0] in event_colors:
            #print('####!!!!!!!!!!')
            colors.append(event_colors[lab[0]])
            sentences.append(sent_tmp[i])
            labels.append(lab[0])

    return sentences,colors,labels

##############################################################################
# Updates (callbacks)
##############################################################################

# update num docs figure based on parameters
@app.callback(
    Output(component_id='numdocs_graph', component_property='figure'),
     [Input('date-picker-range', 'start_date'),
     Input('date-picker-range', 'end_date'),
      Input('event_dropdown','value')])
def update_numdocs_figure(start_date,end_date,event_type):
    return figure_num_articles(xlimrange=[start_date,end_date],event_type=event_type)

# update the article text view
@app.callback(
    Output('fullarticleiframe', 'srcDoc'),
     [Input('prevarticlebutton','n_clicks'),
     Input('nextarticlebutton','n_clicks'),
      Input('event_dropdown', 'value')],
    [State('updatelabelsbutton','n_clicks')])
def update_shown_articletext(n_clicks_prev,n_clicks_next,event_type,n_clicks_updatelabels):

    global current_file_id,df_mod

    # check if need to use modified df
    if n_clicks_updatelabels is None:
        df_show = df
        print('click updates alreadY?')
    else:
        df_show = df_mod
        print('take modified dataset')

    if not event_type is None:
        df_show = df_show[df_show[event_type]==1.]

    # select df based on clicks
    if n_clicks_prev is None:
        n_clicks_prev=0
    if n_clicks_next is None:
        n_clicks_next=0
    n_clicks = n_clicks_next-n_clicks_prev
    print(n_clicks_prev,n_clicks_next,n_clicks)

    len_data_entries = len(df_show)
    if n_clicks>=len_data_entries:
        n_clicks = len_data_entries
    idx = n_clicks - 1

    current_file_id = df_show.iloc[idx]['file_id']
    print('set current_file_id',current_file_id)

    lines,colors,labels=get_tagged_sentences(current_file_id)
    print('lines1',lines,len(lines))
    print('colors1',colors,len(colors))

    return render_document(title=df_show.iloc[idx]['title'], body=df_show.iloc[idx]['text'],
                           id=str(current_file_id),date=df_show.iloc[idx]['date'],
                           lines=lines, colors=colors,labels=labels)


# show article labels for article considered
@app.callback(
    dash.dependencies.Output('articlechecklist', 'values'),
    [Input('fullarticleiframe','srcDoc')],
    #[Input('prevarticlebutton', 'n_clicks'),
     #Input('nextarticlebutton', 'n_clicks')],
    [State('updatelabelsbutton', 'n_clicks')])
def update_shown_labels(srcDocs,n_clicks_updatelabels):

    global current_file_id,df_mod
    print('enter update_shown_labels')

    # check if need to use modified df
    if n_clicks_updatelabels is None:
        print('click updates alreadY?')
        df_show = df
    else:
        print('take modified dataset')
        df_show = df_mod

    row = df_show.loc[df_show['file_id']==current_file_id]
    if row.empty:
        print('error row is empty for id ',current_file_id)
    print('check row ',row[event_labels].values, 'for id ',current_file_id
          )
    active_labels=[]
    for label in event_labels:
        if row[label].values[0] == 1.:
            active_labels.append(label)
    print('found labels ',active_labels)

    return active_labels

# update df-mod with labels submitted when update button is clicked
@app.callback(
    Output('updated_df', 'children'), #updated_df, placeholder
    [Input('updatelabelsbutton', 'n_clicks')],
     [State('articlechecklist','values')])
def update_article_label(n_clicks,checklistvals):
    global df_mod
    global current_file_id

    #print('executed!! ')
    #print('number clicks:',n_clicks)
    #print('checklinkvals',checklistvals)

    # modify df_mod
    df_mod[df_mod['file_id'] == current_file_id][event_labels] = 0  # set all 0 first
    for label in checklistvals:
        print('set label ',label)
        df_mod.loc[df_mod['file_id']==current_file_id,label] = 1
    print('labels ',checklistvals,' were updated for id ',current_file_id)


# save df_mod to file
@app.callback(
    Output('updated_df2', 'children'), #updated_df, placeholder
    [Input('savetofilebutton', 'n_clicks')])
def save_updated_labelsfile(n_clicks):
    global df_mod,input_file
    import time

    filename=input_file[:-3]+str(int(time.time()))+'.csv'
    print('save to ',filename)
    df_mod.to_csv(filename)


# update file list once df_mod is saved
@app.callback(
    Output('file_dropdown', 'options'),  # updated_df, placeholder
    [Input('updated_df2', 'children')])
def save_updated_labelsfile(vals):
    return [{'label':f,'value':f} for f in sorted([f for f in os.listdir('.') if (os.path.isfile(f) and '.csv' in f)],key=lambda x: os.path.getmtime(x))]
    #return [{'label': file, 'value': file} for file in [f for f in os.listdir('.') if (os.path.isfile(f) and 'csv' in f)]]


# read file selected in dropdown menu
@app.callback(
    Output('updated_df4', 'children'),  # updated_df, placeholder
    [Input('file_dropdown', 'value')])
def load_file(file_to_load):
    global df,df_mod
    print('read file into df: ',file_to_load)
    df = pd.read_csv(file_to_load)
    df = df.dropna()
    df = index_to_time(df)
    df = df.sort_index()
    # for modified labels
    df_mod = df.copy()

# read file selected in dropdown menu
#@app.callback(
#    Output('updated_df5', 'children'),  # updated_df, placeholder
#    [Input('retrainbutton', 'value'),
 #    Input('file_dropdown','file_to_load')])
#def retraining(value,filename):
    # start training based on filename and


##############################################################################
# Execute main
##############################################################################

if __name__ == '__main__':
    app.run_server(debug=True)