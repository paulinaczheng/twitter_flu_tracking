import plotly.figure_factory as ff
import plotly.graph_objs as go
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from nltk.tokenize import TweetTokenizer, word_tokenize
from sklearn.feature_selection import chi2
import warnings
from sklearn.externals import joblib
from nltk.corpus import stopwords
from sklearn.metrics import precision_score, accuracy_score, recall_score, f1_score, confusion_matrix, roc_curve, auc, roc_auc_score
import dash_core_components as dcc
import dash_html_components as html
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller
import base64

process_diagram = 'images/process_diagram.png'
encoded_process_image = base64.b64encode(open(process_diagram, 'rb').read())
sarima_diagram = 'images/sarima_process.png'
encoded_sarima_image = base64.b64encode(open(sarima_diagram, 'rb').read())
acf_pacf_diagram = 'images/acf_pacf.png'
encoded_acf_pacf_image = base64.b64encode(open(acf_pacf_diagram, 'rb').read())
diagnostics_diagram = 'images/diagnostics.png'
encoded_diagnostics_image = base64.b64encode(open(diagnostics_diagram, 'rb').read())

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

#vectorize data, TF-IDF with bigrams
def tokenize(tweet):
    tknzr = TweetTokenizer(strip_handles=True, reduce_len=True, preserve_case=False)
    return tknzr.tokenize(tweet)

tfidfvec = TfidfVectorizer(stop_words='english', tokenizer=tokenize, ngram_range=(1,3), max_features=20000)
x_train = tfidfvec.fit_transform(train_data)
x_test = tfidfvec.transform(test_data)

df = pd.read_csv('annotated_counts.csv')
df = df.drop('Unnamed: 0', axis=1)

#import time-series data
def clean_df(df):
    df.reset_index(inplace=True) # Resets the index, makes factor a column
    df.drop('PERCENTAGE OF VISITS FOR INFLUENZA-LIKE-ILLNESS REPORTED BY SENTINEL PROVIDERS',axis=1,inplace=True) # drop factor from axis 1 and make changes permanent by inplace=True
    df.columns = df.iloc[0]
    df = df.iloc[1:]
    df["Date"] = pd.to_datetime(df.WEEK.astype(str)+
                              df.YEAR.astype(str).add('-1') ,format='%W%Y-%w')
    df.set_index(df['Date'], inplace=True)
    df['ILITOTAL'] = df['ILITOTAL'].astype('int64')
    return df

cdc_16 = pd.read_csv('FluViewPhase2Data/16_17.csv')
cdc_17 = pd.read_csv('FluViewPhase2Data/17_18.csv')

cdc_16 = clean_df(cdc_16)
cdc_16 = cdc_16.drop(['AGE 0-4', 'AGE 25-49', 'AGE 25-64', 'AGE 5-24',
               'AGE 50-64', 'AGE 65', 'NUM. OF PROVIDERS', 'YEAR','WEEK',
              '%UNWEIGHTED ILI', 'TOTAL PATIENTS'], axis=1)
cdc_17 = clean_df(cdc_17)
cdc_17 = cdc_17.drop(['REGION TYPE', 'REGION', 'AGE 0-4', 'AGE 25-49', 'AGE 25-64', 'AGE 5-24',
               'AGE 50-64', 'AGE 65', 'NUM. OF PROVIDERS', 'YEAR','WEEK', '% WEIGHTED ILI',
              '%UNWEIGHTED ILI'], axis=1)
cdc_df = pd.concat([cdc_16, cdc_17])

twitter_df = pd.read_csv('Parsed_tweets_2.csv', header=None)
twitter_df.drop([0], axis=1, inplace=True)
twitter_df.columns = ['original_date', 'tweet_id', 'status', 'text', 'week/year']
twitter_df_new = pd.DataFrame(twitter_df['week/year'].str.split('/',1).tolist(),
                                   columns = ['week','year'])
twitter_df = twitter_df.join(twitter_df_new, how='outer')
twitter_df["date"] = pd.to_datetime(twitter_df['week'].astype(str)+
                                           twitter_df['year'].astype(str).add('-1') ,format='%W%Y-%w')
twitter_df = twitter_df.groupby(['date']).size().reset_index(name='count')
twitter_df.set_index(twitter_df['date'], inplace=True)
twitter_df['count'] = twitter_df['count'].astype('int64')
twitter_df = twitter_df['2016-10-03':'2018-09-10']

fin_cdc_df = cdc_df.join(twitter_df, how='outer')
fin_cdc_df.at['2018-07-30', 'count'] = 7107
fin_cdc_df.at['2018-08-06', 'count'] = 7107

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
        classifier_name = generate_classifier_name(classifier)
        classifier.fit(x_test, y_test)
        if classifier==log:
            y_score = classifier.decision_function(x_test)
            fpr, tpr, thresholds = roc_curve(y_test, y_score)
        else:
            y_score = classifier.predict_proba(x_test)
            fpr, tpr, thresholds = roc_curve(y_test, y_score[:,1])
        roc_auc = auc(fpr, tpr)
        trace = go.Scatter(x=fpr, y=tpr,
                           mode='lines',
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

def generate_feature_importance():
    x_train = tfidfvec.fit_transform(train_data)
    x_test = tfidfvec.transform(test_data)
    chi2score = chi2(x_train, y_train)[0]
    wscores = list(zip(tfidfvec.get_feature_names(), chi2score))
    wchi2 = sorted(wscores, key=lambda x:x[1])
    topchi2 = list(zip(*wchi2[-20:]))
    x = range(len(topchi2[1]))
    labels = topchi2[0]
    return topchi2

def generate_chisquare_plot():
    topchi2 = generate_feature_importance()
    trace1 = go.Scatter(x=list(topchi2[1]),y=list(topchi2[0]))
    trace2 = go.Bar(x=list(topchi2[1]),y=list(topchi2[0]), orientation='h')
    return [trace1, trace2]

def generate_eda_plot():
   return [{'x': df['status'], 'y': df['counts'], 'type': 'bar'}]

def generate_visualizations():
    trace1 = go.Scatter(x=cdc_df['Date'], y=cdc_df['ILITOTAL'], name='CDC Visits')
    trace2 = go.Scatter(x=twitter_df['date'], y=twitter_df['count'], name='Flu-Related Tweets')
    data = [trace1, trace2]
    return dcc.Graph(id='ts-visual', figure={'data': data,
    'layout': go.Layout(xaxis={'title': 'Week'},
                        yaxis={'title': 'Count'}
                                )})

def smoothing_plots():
    cdc_df_new = cdc_df.drop(['Date'], axis=1)
    moving_avg = cdc_df_new.rolling(12).mean()
    moving_std = cdc_df_new.rolling(12).std()
    trace1 = go.Scatter(x=cdc_df['Date'], y=cdc_df['ILITOTAL'], name='Original')
    trace2 = go.Scatter(x=cdc_df['Date'], y=moving_avg['ILITOTAL'], name='Rolling Mean')
    trace3 = go.Scatter(x=cdc_df['Date'], y=moving_std['ILITOTAL'], name='Rolling STD')
    data = [trace1, trace2, trace3]
    return dcc.Graph(id='smooth-visual', figure={'data': data,
    'layout': go.Layout(xaxis={'title': 'Week'},
                                )})

def acf_pacf_plots():
    return html.Img(src='data:image/png;base64,{}'.format(encoded_acf_pacf_image.decode()))

def diagnostics_plots():
    return html.Img(src='data:image/png;base64,{}'.format(encoded_diagnostics_image.decode()))
