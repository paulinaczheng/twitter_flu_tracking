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
from twitter_package.charts import *
import base64
import warnings
from nltk.corpus import stopwords
from nltk.tokenize import TweetTokenizer, word_tokenize
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import precision_score, accuracy_score, recall_score, f1_score, confusion_matrix
from sklearn.metrics import roc_curve, auc, roc_auc_score

classifiers = []

with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    nb = joblib.load('models/nb.pkl')
    classifiers.append(nb)
    log = joblib.load('models/log.pkl')
    classifiers.append(log)
    forest = joblib.load('models/forest.pkl')
    classifiers.append(forest)
    gradboost = joblib.load('models/gradboost.pkl')
    classifiers.append(gradboost)
    adaboost = joblib.load('models/adaboost.pkl')
    classifiers.append(adaboost)
    svm = joblib.load('models/svm.pkl')
    classifiers.append(svm)

#load training and test sets
test_data = pd.read_csv('train_test_data/test_data.csv', header=None)
test_data = test_data[1]
train_data = pd.read_csv('train_test_data/train_data.csv', header=None)
train_data = train_data[1]
y_test = pd.read_csv('train_test_data/y_test.csv', header=None)
y_test = y_test[1]
y_train = pd.read_csv('train_test_data/y_train.csv', header=None)
y_train = y_train[1]

# Load npz file containing image arrays
# x_train_npz = np.load("train_test_data/x_train.npz")
# x_train = x_train_npz['arr_0']
# x_test_npz = np.load("train_test_data/x_test.npz")
# x_test = x_test_npz['arr_0']
# y_train_npz = np.load("train_test_data/y_train.npz")
# y_train = y_train_npz['arr_0']
# y_test_npz = np.load("train_test_data/y_test.npz")
# y_test = y_test_npz['arr_0']

#vectorize data, TF-IDF with bigrams
def tokenize(tweet):
    tknzr = TweetTokenizer(strip_handles=True, reduce_len=True, preserve_case=False)
    return tknzr.tokenize(tweet)

tfidfvec2 = TfidfVectorizer(stop_words='english', tokenizer=tokenize, ngram_range=(1,2), max_features=20000)
x_train = tfidfvec2.fit_transform(train_data)
x_test = tfidfvec2.transform(test_data)

process_diagram = 'images/process_diagram.png'
encoded_process_image = base64.b64encode(open(process_diagram, 'rb').read())
sarima_diagram = 'images/sarima_process.png'
encoded_sarima_image = base64.b64encode(open(sarima_diagram, 'rb').read())

def generate_classifier_name(model):
    if model==log:
        return 'Logistic Regression'
    elif model==nb:
        return 'Naive Bayes'
    elif model==forest:
        return 'Random Forest'
    elif model==gradboost:
        return "Gradient Boost"
    elif model==adaboost:
        return "Adaboost"
    elif model==svm:
        return 'Support Vector Machine'

def generate_all_roc_curves():
    lw = 2
    data = []
    for classifier in classifiers:
#         print(classifier)
        classifier_name = generate_classifier_name(classifier)
        classifier.fit(x_test, y_test)
        if classifier==log:
            y_score = classifier.decision_function(x_test)
            fpr, tpr, thresholds = roc_curve(y_test, y_score)
#             print(y_score)
        else:
            y_score = classifier.predict_proba(x_test)
            fpr, tpr, thresholds = roc_curve(y_test, y_score[:,1])
        roc_auc = auc(fpr, tpr)
        trace = go.Scatter(x=fpr, y=tpr,
                           mode='lines',
#                            line=dict(width=lw, color=color),
                           name='{} (area = {})'.format(classifier_name, round(roc_auc,2)))
        data.append(trace)
    trace = go.Scatter(x=[0, 1], y=[0, 1],
               mode='lines',
               line=dict(width=lw, color='black', dash='dash'),
               name='Luck')
    data.append(trace)
    layout = go.Layout(title='Receiver Operating Characteristic (ROC) Curve',
                       xaxis=dict(title='False Positive Rate', showgrid=False,
                                  range=[-0.05, 1.05]),
                       yaxis=dict(title='True Positive Rate', showgrid=False,
                                  range=[-0.05, 1.05]))
    fig = go.Figure(data=data, layout=layout)
    return fig

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
        dcc.Tab(label='Vectorization Overview', children=[
            html.Div([
                dcc.Dropdown(
                id='select-vectorizer-metrics',
                options=[{'label': 'Count Vectorization', 'value': 'count'},
                {'label': 'TF-IDF Vectorization', "value": 'tfidf'},
                {'label': 'Doc2Vec', 'value': 'doc2vec'},
                        ],
                placeholder="Select Vectorizer", value ='Vectorizer'),
                        ])
                        ]),
        dcc.Tab(label='Natural Language Processing', children=[
            html.Div([
                html.H1('Chi-square values and features here'),
                html.H1('PCA plot here')
                        ])
                        ]),
        dcc.Tab(label='Exploratory Data Analysis', children=[
            html.Div([
                html.H1('Add EDA')
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
                html.H1('SARIMAX metrics here'),
                dcc.Dropdown(
                id='select-arima-metrics',
                options=[{'label': 'Visualizations', 'value': 'visual'},
                {'label': 'Stationarity', "value": 'stationarity'},
                {'label': 'ACF & PACF Plots', 'value': 'acf_pacf'},
                {'label': 'Diagnostic Plots', 'value': 'diagnostics'},
                {'label': 'Forecasting', 'value': 'forecasting'},
                        ],
                placeholder="Select Model Metrics", value ='Metric'),
                        ])
                        ]),
        dcc.Tab(label='Conclusions', children=[
            html.Div([
                html.H1('Conclusions'),
                html.H1('Limitations'),
                html.H1('Next Steps'),
                dcc.Markdown('* Include Google Trends data in time-series analyses'),
                dcc.Markdown('* Use other time-series models (VARIMA)'),
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
