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

# nb = joblib.load('nb.pkl')
# log = joblib.load('log.pkl')
# forest = joblib.load('forest.pkl')
# gradboost = joblib.load('gradboost.pkl')
# adaboost = joblib.load('adaboost.pkl')
# svm = joblib.load('svm.pkl')
# mlp = open('mlp.json', 'r')
# mlp = mlp.read()
# mlp.close()
# mlp_model = model_from_json(mlp)
# mlp_model.load_weights("mlp.h5")

# def generate_map():
#     data = [go.Scattermapbox(lat=df['centroid_lat'],
#                             lon=df['centroid_long'],
#                             mode='markers',
#                             marker=dict(size=6),
#                             text=df['text'])
#                             ]
#     layout = go.Layout(width=1300, height=900,
#                     hovermode='closest',
#                     mapbox=dict(
#                     accesstoken=mapbox_access_token,
#                     bearing=0,
#                     center=dict(lat=39.8283,lon=-99.5795),
#                     pitch=0,
#                     zoom=3.5),)
#     return {'data': data, 'layout': layout}

app.layout = html.Div(style={'fontFamily': 'Sans-Serif'}, children=[
    html.H1('Tracking Flu Outbreaks with Twitter', style={'textAlign': 'center', 'margin': '48px 0', 'fontFamily': 'Sans-Serif'}),
    dcc.Tabs(id="tabs", children=[
        dcc.Tab(label='Framework Overview', children=[
            html.Div([
                html.H1('Image of diagram here')
                # dcc.Graph(id='map',figure=generate_map())
                        ])
                        ]),
        dcc.Tab(label='Models Overview', children=[
                dcc.Dropdown(
                id='select-model',
                options=[{'label': 'Logistic Regression', 'value': 'log'},
                {'label': 'Random Forest Classifier', 'value': 'forest'},
                {'label': 'Gradient Boosted Trees', 'value': 'gradboost'},
                {'label': 'AdaBoost', 'value': 'adaboost'},
                {'label': 'Support Vector Machine', 'value': 'svm'},
                {'label': 'Multilayer Perceptron', 'value': 'mlp'},
                        ],
                placeholder="Select a Model", value ='Model'),
                # html.Div(id='cm-container')
                        ]),
                        ])
                        ])
