




app.layout = html.Div(children=[
    html.H1(children='News Analytics Dashboard',style={'textAlign': 'center'}),

    dcc.Dropdown(
        options=[
            {'label': 'Retail and Trade', 'value': 'RT'},
            {'label': 'Manufacturing and Wholesale', 'value': 'MW'},
        ],
        value='RT'
    ),

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

        html.Div(children='Select companiess'), #, style={'float': 'left', 'clear': 'both', 'text-align': 'center'}),
        html.Div([
        dcc.Dropdown(
            options=[
                {'label': 'Coca Cola', 'value': 'CC'},
                {'label': 'Sony Ericson', 'value': 'SE'},
            ],
            value='',
        multi=True
        )],className='six columns'),


        html.Div(children='Pick your date range'),#, style={'float': 'left', 'clear': 'both', 'text-align': 'center'}),
        html.Div([
        dcc.DatePickerRange(
            id='date-picker-range',
            start_date=dt(1997, 5, 3),
            end_date_placeholder_text='Select a date!'
        )],className='six columns'),

        ]),#,style={'float':'left','clear':'both'}),




if __name__ == '__main__':
    app.run_server(debug=True)