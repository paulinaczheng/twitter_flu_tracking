import plotly.figure_factory as ff
import plotly.graph_objs as go
import pandas as pd

df = pd.read_csv('annotated_counts.csv')
df = df.drop('Unnamed: 0', axis=1)

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

def generate_feature_importance():
    pass

def generate_eda_plot():
   return [{'x': df['status'], 'y': df['counts'], 'type': 'bar'}]

def bar_trace(artist):
   avg_features = avg_featurevalues_artist(artist)
   x = [feature for feature in avg_features.keys()]
   y = [value for value in avg_features.values()]
   y[4] = tempo_normalization(y[4])
   return {'x': x, 'y': y, 'type': 'bar', 'name': artist}

def all_bars():
   all_bars_list = []
   x = [artist.name for artist in Artist.query.all()]
   for artist in x:
       all_bars_list.append(bar_trace(artist))
   return all_bars_list
