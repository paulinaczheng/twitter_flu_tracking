import plotly.figure_factory as ff
import plotly.graph_objs as go
from sklearn.metrics import precision_score, accuracy_score, recall_score, f1_score, confusion_matrix
import pandas as pd

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

def generate_all_roc_curves():
    data = []
    classifiers = [nb, log, forest, gradboost, adaboost, svm]
    for classifier in classifiers:
        if classifer==log:
            y_score = classifier.decision_function(x_test)
            fpr, tpr, thresholds = roc_curve(labels, y_score)
        else:
            y_score = classifier.predict_proba(x_test)
            fpr, tpr, thresholds = roc_curve(labels, y_score[:,1])
        mean_tpr += interp(mean_fpr, fpr, tpr)
        mean_tpr[0] = 0.0
        roc_auc = auc(fpr, tpr)
        trace = go.Scatter(x=fpr, y=tpr,
                           mode='lines',
                           line=dict(width=lw, color=color),
                           name='ROC fold %d (area = %0.2f)' % (i, roc_auc))
        data.append(trace)
        i += 1
    trace = go.Scatter(x=[0, 1], y=[0, 1],
                       mode='lines',
                       line=dict(width=lw, color='black', dash='dash'),
                       name='Luck')
    data.append(trace)

    mean_tpr /= 6
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)

    trace = go.Scatter(x=mean_fpr, y=mean_tpr,
                       mode='lines',
                       line=dict(width=lw, color='green', dash='dash'),
                       name='Mean ROC (area = %0.2f)' % mean_auc)
    data.append(trace)

    layout = go.Layout(title='Receiver Operating Characteristic (ROC) Curve',
                       xaxis=dict(title='False Positive Rate', showgrid=False,
                                  range=[-0.05, 1.05]),
                       yaxis=dict(title='True Positive Rate', showgrid=False,
                                  range=[-0.05, 1.05]))
    return go.Figure(data=data, layout=layout)
