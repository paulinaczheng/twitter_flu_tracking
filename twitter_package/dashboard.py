import dash_core_components as dcc
import dash_html_components as html
from twitter_package import *
import plotly.plotly as py
import plotly.graph_objs as go
import numpy as np
import json
import dash
from dash.dependencies import Input, Output, State
from sklearn.externals import joblib
import pandas as pd
import plotly.figure_factory as ff
from twitter_package.charts import *

# nb = joblib.load('nb.pkl')
# log = joblib.load('log.pkl')
# forest = joblib.load('forest.pkl')
# gradboost = joblib.load('gradboost.pkl')
# adaboost = joblib.load('adaboost.pkl')
# svm = joblib.load('svm.pkl')
## load training and test sets

app.layout = html.Div(style={'fontFamily': 'Sans-Serif'}, children=[
    html.H1('Tracking Flu Outbreaks with Twitter', style={'textAlign': 'center', 'margin': '48px 0', 'fontFamily': 'Sans-Serif'}),
    dcc.Tabs(id="tabs", children=[
        dcc.Tab(label='Framework Overview', children=[
            html.Div([
                html.H1('Image of process diagram here')
                        ])
                        ]),
        dcc.Tab(label='Map Overview', children=[
            html.Div([
                html.H1('Generate map here')
                # dcc.Graph(id='map',figure=generate_map())
                        ])
                        ]),
        dcc.Tab(label='Vectorization Overview', children=[
            html.Div([
                html.H1('Vectorization metrics overview here')
                        ])
                        ]),
        dcc.Tab(label='Feature Importance', children=[
            html.Div([
                html.H1('Chi-square values and features here')
                        ])
                        ]),
        dcc.Tab(label='Models Overview', children=[
                dcc.Dropdown(
                id='select-model',
                options=[{'label': 'Logistic Regression', 'value': 'log'},
                {'label': 'Random Forest Classifier', 'value': 'forest'},
                {'label': 'Gradient Boosted Trees', 'value': 'gradboost'},
                {'label': 'AdaBoost', 'value': 'adaboost'},
                {'label': 'Support Vector Machine', 'value': 'svm'}
                        ],
                placeholder="Select a Model", value ='Model'),
                # html.Div(id='cm-container')
                        ]),
        dcc.Tab(label='Time Series Analysis', children=[
            html.Div([
                html.H1('SARIMAX metrics here')
                        ])
                        ]),
        dcc.Tab(label='Conclusions', children=[
            html.Div([
                html.H1('Conclusions here')
                        ])
                        ]),
                        ])
                        ])
