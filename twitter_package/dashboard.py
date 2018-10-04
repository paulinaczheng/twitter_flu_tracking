from twitter_package import *
import plotly.plotly as py
import plotly.graph_objs as go
import json
import dash
from dash.dependencies import Input, Output, State
import dash_table_experiments as dt
from twitter_package.charts import *

#load metrics sets
count_df = pd.read_csv('count_metrics.csv')
count_df = count_df.drop('Unnamed: 0', axis=1)
tfidf_df = pd.read_csv('tfidf_metrics.csv')
tfidf_df = tfidf_df.drop('Unnamed: 0', axis=1)
doc2vec_df = pd.read_csv('doc2vec_metrics.csv')
doc2vec_df.rename(columns={'Unnamed: 0': 'Training Method (n-gram)'}, inplace=True)

dataframes = {'Count Vectorization': count_df,
              'TF-IDF Vectorization': tfidf_df,
              'Doc2Vec': doc2vec_df
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
                html.Div([
                        html.Img(src='data:image/png;base64,{}'.format(encoded_process_image.decode()))
                        ])
                        ])
                        ]),
        dcc.Tab(label='Exploratory Data Analysis', children=[
            html.Div([
                    html.Img(src='data:image/png;base64,{}'.format(encoded_sample_image.decode())),
                    dcc.Graph(id='EDA', figure={'data': generate_eda_plot(),
                    'layout': go.Layout(xaxis={'title': 'Flu-Related'},
                                        yaxis={'title': 'Count'},
                                        title='Status Distribution of Annotated Tweets',
                                        )},
                                style={'margin-left': '100px'}
                                ),
                        ],
                        style={'display': 'flex', 'align-items': 'center'})
                        ]),
        dcc.Tab(label='Natural Language Processing', children=[
            html.Div([
                html.H1('Metrics Table'),
                html.Label('Report Type:', style={'font-weight': 'bold'}),
                dcc.Dropdown(
                    id='field-dropdown',
                    options=[{'label': df, 'value': df} for df in dataframes],
                    value='Count Vectorization',
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
                        ])
                        ]),
        dcc.Tab(label='Feature Importance', children=[
            html.Div([
                    dcc.Graph(id='chisquare', figure={'data': generate_chisquare_plot(),
                    'layout': go.Layout(xaxis={'title': 'Chi-Square Value'},
                                        showlegend=False,
                                                )},
                                                )
                        ])
                        ]),
        dcc.Tab(label='Doc2Vec: PCA', children=[
            html.Div([
                dcc.Markdown("*Explained variance ratio is 3.82%, 96.18% is lost by reducing the dimensionality*"),
                dcc.RadioItems(
                id='select-pca-visualization',
                options=[{'label': 'Line Plot', 'value': 'line'},
                        {'label': '3D Visualization', "value": '3d'},
                            ],
                        value='line',
                        labelStyle={'display': 'inline-block'}),
                html.Div(id='pca-container'),
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
                dcc.Markdown('***'),
                html.Div(id='cm-roc-container'),
                        ]),
        dcc.Tab(label='Time Series Analysis', children=[
            html.Div([
                html.H1('Time Series Analysis Overview'),
                html.Div([html.Img(src='data:image/png;base64,{}'.format(encoded_sarima_image.decode()))]),
                dcc.Markdown('***'),
                dcc.Dropdown(
                id='select-arima-metrics',
                options=[{'label': 'Visualizations', 'value': 'visual'},
                {'label': 'Stationarity', "value": 'stationarity'},
                {'label': 'ACF & PACF Plots', 'value': 'acf_pacf'},
                {'label': 'Diagnostic Plots', 'value': 'diagnostics'},
                {'label': 'Forecasting', 'value': 'forecasting'},
                        ],
                placeholder="Select Model Metrics", value ='Metric'),
                dcc.Markdown('***'),
                html.Div(id='ts-container'),
                        ])
                        ]),
        dcc.Tab(label='Conclusions', children=[
            html.Div([
                html.H1('Conclusions'),
                dcc.Markdown('* Use model to predict outbreaks'),
                dcc.Markdown('* Logistic regression was the best-performing classifier, with TF-IDF vectorization (with trigrams) used to process the annotated tweets'),
                dcc.Markdown('* The SARIMA model that included both CDC & Twitter data did better at one-step ahead forecasting than the SARIMA model with just CDC data, using RMSE as a metric (3651.55 vs. 3448.62)'),
                dcc.Markdown('* This implies that flu-related tweets contribute to the SARIMA model in some way that improves the predictive ability of the SARIMA model'),
                html.H1('Limitations'),
                dcc.Markdown('* ARIMA models very dependent on data trends and characteristics, requiring frequent refitting of model parameters'),
                dcc.Markdown('* Poor ability to prospectively detect outbreaks'),
                html.H1('Next Steps'),
                dcc.Markdown('* Add more annotated tweets to train classifiers on'),
                dcc.Markdown('* Incorporate deep learning (recurrent neural networks) to classify tweets'),
                dcc.Markdown('* Include Google Trends data in time-series analyses'),
                dcc.Markdown('* Use other time-series libraries/models for better forecasting (VARIMA, FBProphet)'),
                        ])
                        ]),
                        ])
                        ])

@app.callback(Output(component_id = 'pca-container', component_property ='children'),
[Input(component_id = 'select-pca-visualization',component_property = 'value')])
def generate_pca_plot(input_value):
    if input_value=='line':
        return generate_pca_line_plot()
    elif input_value=='3d':
        return doc2vec_3d_plot()

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

def generate_confusion_matrix(input_value):
    model = check_model(input_value)
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)
    cm = confusion_matrix(y_test, predictions)
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    trace = [go.Heatmap(x=['UNRELATED', 'RELATED'], y=['UNRELATED', 'RELATED'], z=cm,
                        colorscale='Viridis')]
    layout = go.Layout(title='Confusion Matrix',
                           xaxis=dict(title='Actual Class',
                                      range=[0, 0.5, 1]),
                           yaxis=dict(title='Predicted Class',
                                      range=[0, 0.5, 1]))
    return dcc.Graph(id ='heatmap', figure = go.Figure(data = trace, layout= layout), style={'display': 'inline-block'})

def generate_roc_curve(input_value):
    data = []
    lw=2
    model = check_model(input_value)
    model.fit(x_train, y_train)
    model_name = generate_classifier_name(model)
    if model==log:
        y_score = model.decision_function(x_test)
        fpr, tpr, thresholds = roc_curve(y_test, y_score)
    else:
        y_score = model.predict_proba(x_test)
        fpr, tpr, thresholds = roc_curve(y_test, y_score[:,1])
    roc_auc = auc(fpr, tpr)
    trace = go.Scatter(x=fpr, y=tpr,
                       mode='lines',
                           line=dict(width=lw, color='b'),
                       name='{} (area = {})'.format(model_name, round(roc_auc,2)))
    data.append(trace)
    trace = go.Scatter(x=[0, 1], y=[0, 1],
               mode='lines',
               line=dict(width=lw, color='black', dash='dash'),
               name='Chance')
    data.append(trace)
    layout = go.Layout(title='Receiver Operating Characteristic (ROC) Curve',
                       xaxis=dict(title='False Positive Rate', showgrid=False,
                                  range=[-0.05, 1.05]),
                       yaxis=dict(title='True Positive Rate', showgrid=False,
                                  range=[-0.05, 1.05]))
    fig = go.Figure(data=data, layout=layout)
    return dcc.Graph(id='roc', figure={'data': fig,
                    'layout': layout}, style={'display': 'inline-block'})


@app.callback(Output(component_id = 'cm-roc-container', component_property ='children'),
[Input(component_id = 'select-model',component_property = 'value')])
def generate_cm_roc_plot(input_value):
    return generate_confusion_matrix(input_value), generate_roc_curve(input_value)

@app.callback(Output(component_id = 'ts-container', component_property ='children'),
[Input(component_id = 'select-arima-metrics',component_property = 'value')])
def generate_vectorization_metrics(input_value):
    if input_value=='visual':
        return generate_visualizations()
    elif input_value=='stationarity':
        return smoothing_plots()
    elif input_value=='acf_pacf':
        return acf_pacf_plots()
    elif input_value=='diagnostics':
        return diagnostics_plots()
    elif input_value=='forecasting':
        return forecasting_plots()

@app.callback(Output('table', 'rows'), [Input('field-dropdown', 'value')])
def update_table(user_selection):
    df = get_data_object(user_selection)
    return df.to_dict('records')
