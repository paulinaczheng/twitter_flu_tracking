import dash_core_components as dcc
import dash_html_components as html
from twitter_package import *
import plotly.plotly as py
import plotly.graph_objs as go
import json
import dash
from dash.dependencies import Input, Output, State
import dash_table_experiments as dt
from twitter_package.charts import *
import base64

#load metrics sets
count_df = pd.read_csv('count_metrics.csv')
count_df = count_df.drop('Unnamed: 0', axis=1)
tfidf_df = pd.read_csv('tfidf_metrics.csv')
tfidf_df = tfidf_df.drop('Unnamed: 0', axis=1)
doc2vec_df = pd.read_csv('doc2vec_metrics.csv')
doc2vec_df.rename(columns={'Unnamed: 0': 'Training Method (n-gram)'}, inplace=True)

process_diagram = 'images/process_diagram.png'
encoded_process_image = base64.b64encode(open(process_diagram, 'rb').read())
sarima_diagram = 'images/sarima_process.png'
encoded_sarima_image = base64.b64encode(open(sarima_diagram, 'rb').read())

dataframes = {'COUNT_DF': count_df,
              'TFIDF_DF': tfidf_df,
              'DOC2VEC_DF': doc2vec_df
              }

def get_data_object(user_selection):
    return dataframes[user_selection]

app.layout = html.Div(style={'fontFamily': 'Sans-Serif'}, children=[
    html.H1('Tracking Flu Outbreaks with Twitter', style={'textAlign': 'center', 'margin': '48px 0', 'fontFamily': 'Sans-Serif'}),
    dcc.Tabs(id="tabs", children=[
        dcc.Tab(label='Framework Overview', children=[
            html.Div([
                html.H1('Project Process Overview'),
                dcc.Markdown('The project was defined by two phases: (1) training machine learning classification models to identify flu-related tweets and (2) conducting time-series analyses with identified tweets and CDC data.'),
                html.Div([html.Img(src='data:image/png;base64,{}'.format(encoded_process_image.decode()))])
                        ])
                        ]),
        dcc.Tab(label='Exploratory Data Analysis', children=[
            html.Div([
                    dcc.Graph(id='EDA', figure={'data': generate_eda_plot(),
                    'layout': go.Layout(xaxis={'title': 'Flu-Related'},
                                        yaxis={'title': 'Count'}
                                        )})
                        ])
                        ]),
        dcc.Tab(label='Natural Language Processing', children=[
            html.Div([
                # dcc.Dropdown(
                # id='select-vectorizer-metrics',
                # options=[{'label': 'Count Vectorization', 'value': 'count'},
                # {'label': 'TF-IDF Vectorization', "value": 'tfidf'},
                # {'label': 'Doc2Vec', 'value': 'doc2vec'},
                #         ],
                # placeholder="Select Vectorizer", value ='Vectorizer'),
                # dcc.RadioItems(
                # id='select-vectorizer-metrics',
                # options=[{'label': 'Count Vectorization', 'value': 'count'},
                #         {'label': 'TF-IDF Vectorization', "value": 'tfidf'},
                #         {'label': 'Doc2Vec', 'value': 'doc2vec'},
                #             ],
                #         value='count',
                #         labelStyle={'display': 'inline-block'}),
                # html.Div(id='vec-container'),
                html.H4('DataTable'),
                html.Label('Report type:', style={'font-weight': 'bold'}),
                dcc.Dropdown(
                    id='field-dropdown',
                    options=[{'label': df, 'value': df} for df in dataframes],
                    value='DF_COUNT',
                    clearable=False
                            ),
                dt.DataTable(
                    # Initialise the rows
                    rows=[{}],
                    row_selectable=True,
                    filterable=True,
                    sortable=True,
                    selected_row_indices=[],
                    id='table'
                            ),
                # html.Div(id='selected-indexes')
                        ])
                        ]),
        dcc.Tab(label='Feature Importance', children=[
            html.Div([
                    dcc.Graph(id='chisquare', figure={'data': generate_chisquare_plot(),
                    'layout': go.Layout(xaxis={'title': 'Chi-Square Value'},
                                                )})
                        ])
                        ]),
        dcc.Tab(label='Models Overview', children=[
                dcc.Dropdown(
                id='select-model',
                options=[{'label': 'Naive Bayes', 'value': 'nb'},
                {'label': 'Logistic Regression', 'value': 'log'},
                {'label': 'Random Forest Classifier', 'value': 'forest'},
                {'label': 'Gradient Boosted Trees', 'value': 'gradboost'},
                {'label': 'AdaBoost', 'value': 'adaboost'},
                {'label': 'Support Vector Machine', 'value': 'svm'}
                        ],
                placeholder="Select a Model", value ='Model'),
                html.Div(id='cm-container'),
                html.Div([dcc.Graph(id='roc',figure=generate_all_roc_curves())]),
                        ]),
        dcc.Tab(label='Time Series Analysis', children=[
            html.Div([
                html.H1('Time Series Analysis Overview'),
                html.Div([html.Img(src='data:image/png;base64,{}'.format(encoded_sarima_image.decode()))]),
                dcc.Dropdown(
                id='select-arima-metrics',
                options=[{'label': 'Visualizations', 'value': 'visual'},
                {'label': 'Stationarity', "value": 'stationarity'},
                {'label': 'ACF & PACF Plots', 'value': 'acf_pacf'},
                {'label': 'Diagnostic Plots', 'value': 'diagnostics'},
                {'label': 'Forecasting', 'value': 'forecasting'},
                        ],
                placeholder="Select Model Metrics", value ='Metric'),
                html.Div(id='ts-container'),
                        ])
                        ]),
        dcc.Tab(label='Conclusions', children=[
            html.Div([
                html.H1('Conclusions'),
                dcc.Markdown('* Logistic regression was the best-performing classifier, with TF-IDF vectorization (with trigrams) used to process the annotated tweets'),
                dcc.Markdown('* The SARIMA model that included both CDC & Twitter data did better at one-step ahead forecasting than the SARIMA model with just CDC data, using RMSE as a metric.'),
                dcc.Markdown('* This implies that flu-related tweets contribute to the SARIMA model in some way that improves the predictive ability of the SARIMA model'),
                html.H1('Limitations'),
                dcc.Markdown('* ARIMA models very dependent on data trends and characteristics, requiring frequent refitting of model parameters'),
                dcc.Markdown('* Poor ability to prospectively detect outbreaks'),
                html.H1('Next Steps'),
                dcc.Markdown('* Add more annotated tweets to train classifiers on'),
                dcc.Markdown('* Include Google Trends data in time-series analyses'),
                dcc.Markdown('* Use other time-series libraries/models for better forecasting (VARIMA, FBProphet)'),
                        ])
                        ]),
                        ])
                        ])

def check_model(model_name):
    if model_name=='log':
        return log
    elif model_name=='nb':
        return nb
    elif model_name=='forest':
        return forest
    elif model_name=='gradboost':
        return gradboost
    elif model_name=='adaboost':
        return adaboost
    elif model_name=='svm':
        return svm

@app.callback(Output(component_id = 'cm-container', component_property ='children'),
[Input(component_id = 'select-model',component_property = 'value')])
def generate_confusion_matrix(input_value):
    model = check_model(input_value)
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)
    cm = confusion_matrix(y_test, predictions)
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    trace = [go.Heatmap(x=['POS', 'NEG'], y=['POS', 'NEG'], z=cm)]
    return dcc.Graph(id ='heatmap', figure = go.Figure(data = trace))

# @app.callback(Output(component_id = 'vec-container', component_property ='children'),
# [Input(component_id = 'select-vectorizer-metrics',component_property = 'value')])
# def generate_vectorization_metrics(input_value):
#     if input_value=='count':
#         return 'Count'
#     elif input_value=='tfidf':
#         return 'TF-IDF'
#     elif input_value=='doc2vec':
#         return 'Doc2Vec'

@app.callback(Output(component_id = 'ts-container', component_property ='children'),
[Input(component_id = 'select-arima-metrics',component_property = 'value')])
def generate_vectorization_metrics(input_value):
    if input_value=='visual':
        return 'Visual'
    elif input_value=='stationarity':
        return 'Stationarity'
    elif input_value=='acf_pacf':
        return 'ACF/PACF plots'
    elif input_value=='diagnostics':
        return 'Diagnostics'
    elif input_value=='forecasting':
        return 'Forecasting'

@app.callback(Output('table', 'rows'), [Input('field-dropdown', 'value')])
def update_table(user_selection):
    df = get_data_object(user_selection)
    return df.to_dict('records')
