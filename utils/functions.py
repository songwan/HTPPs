from logging.config import valid_ident
from pathlib import Path
import base64
from tabnanny import verbose
import time
import itertools
import plotly.figure_factory as ff
import streamlit as st
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import pickle
import json
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from tensorflow import keras
from plotly.subplots import make_subplots
import plotly.graph_objs as go
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from models.utils import model_infos, model_urls
from keras.utils import np_utils

# from utils.ui import dataset_selector # for access to "current_data"

def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


def plot_history(history):
    fig, loss_ax = plt.subplots()

    #fig.set_figwidth(10)
    fig.set_figheight(3.5)

    acc_ax = loss_ax.twinx()

    loss_ax.plot(history.history['loss'], 'y', label='train loss')
    loss_ax.plot(history.history['val_loss'], 'r', label='val loss')
    loss_ax.set_xlabel('epoch')
    loss_ax.set_ylabel('loss')
    loss_ax.legend(loc='upper left')
    
    acc_ax.plot(history.history['mae'], 'b', label='train mae')
    acc_ax.plot(history.history['val_mae'], 'g', label='val mae')
    acc_ax.set_ylabel('mae')
    acc_ax.legend(loc='upper right')

    return fig

def plot_confusion_matrix(cm, target_names=None, cmap=None, normalize=True, labels=True, title='Confusion matrix'):

    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    fig = plt.figure(figsize=(10, 10))

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar() 

    thresh = cm.max() / 1.5 if normalize else cm.max() / 2

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names)
        plt.yticks(tick_marks, target_names)
    
    if labels:
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            if normalize:
                plt.text(j, i, "{:0.2f}".format(cm[i, j]),
                         horizontalalignment="center",
                         color="white" if cm[i, j] > thresh else "black")
            else:
                plt.text(j, i, "{:,}".format(cm[i, j]),
                         horizontalalignment="center",
                         color="white" if cm[i, j] > thresh else "black")
 
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    
    return fig

def plot_classification_and_metrics(
        y_train, y_test, metrics, y_train_pred, y_test_pred, labely
):
    conf_mat = confusion_matrix(y_test, y_test_pred)
    fig_conf = plot_confusion_matrix(conf_mat, target_names=labely.transpose()[0].tolist(), title="Confusion matrix")
    
    fig = make_subplots(
        rows=1,
        cols=2,
        specs=[[{"type": "indicator"}, {"type": "indicator"}]],
        #row_heights=[0.30],
    )

    fig.add_trace(
        go.Indicator(
            mode="gauge+number+delta",
            value=metrics["test_accuracy"],
            title={"text": f"Accuracy (test)"},
            domain={"x": [0, 1], "y": [0, 1]},
            gauge={"axis": {"range": [0, 1]}},
            delta={"reference": metrics["train_accuracy"]},
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Indicator(
            mode="gauge+number+delta",
            value=metrics["test_f1"],
            title={"text": f"F1 score (test)"},
            domain={"x": [0, 1], "y": [0, 1]},
            gauge={"axis": {"range": [0, 1]}},
            delta={"reference": metrics["train_f1"]},
        ),
        row=1,
        col=2,
    )

    fig.update_layout(
        height=150,
    )
    fig['layout'].update(margin=dict(l=20,r=20,b=20,t=50))
    
    return fig, fig_conf


# Plotting y vs. y_predicted scatterplot for visualizing prediction results
def plot_prediction_and_metrics(
        y_train, y_test, metrics, y_train_pred, y_test_pred
):

    fig = make_subplots(
        rows=2,
        cols=2,
        specs=[[{"colspan": 2}, None], [{"type": "indicator"}, {"type": "indicator"}]],
        subplot_titles=("Scatterplot", None, None),
        row_heights=[0.7, 0.30],
    )

    train_data = go.Scatter(
        x = y_train,
        y = y_train_pred,
        name="train data",
        mode="markers",
        showlegend=True,
        marker=dict(
            size=5,
            color='green',
            line=dict(color="black", width=2),
        ),
    )

    test_data = go.Scatter(
        x = y_test,
        y = y_test_pred,
        name="test data",
        mode="markers",
        showlegend=True,
        # marker_symbol="cross",
        visible="legendonly",
        marker=dict(
            size=5,
            color='tomato',
            line=dict(color="black", width=2),
        ),
    )

    fig.add_trace(train_data, row=1, col=1).add_trace(test_data).update_xaxes(title='Target').update_yaxes(title='Predicted')
    #.update_xaxes(range=[x_min, x_max], title='Target (test)').update_yaxes(range=[y_min, y_max], title='Predicted')

    fig.add_trace(
        go.Indicator(
            mode="gauge+number+delta",
            value=metrics["test_rsquare"],
            title={"text": f"R-squared (test)"},
            domain={"x": [0, 1], "y": [0, 1]},
            gauge={"axis": {"range": [0, 1]}},
            delta={"reference": metrics["train_rsquare"]},
        ),
        row=2,
        col=1,
    )

    fig.add_trace(
        go.Indicator(
            mode="gauge+number+delta",
            value=metrics["test_mse"],
            title={"text": f"MSE (test)"},
            domain={"x": [0, 1], "y": [0, 1]},
            gauge={"axis": {"range": [0, y_train.var()]}},
            delta={"reference": metrics["train_mse"]},
        ),
        row=2,
        col=2,
    )

    fig.update_layout(
        height=700,
    )

    return fig

# Normalize data
def norm(x, train_stats):
    return (x - train_stats['mean']) / train_stats['std']

def train_keras_model(model, x_train, y_train, x_test, y_test, epochs, validation_split, goal):
   
    t0 = time.time()

    train_stats = x_train.describe().transpose()
    normed_x_train = norm(x_train, train_stats)
    normed_x_test = norm(x_test, train_stats)

    # if goal in ('Classification'):

    # elif goal in ('Regression'):

    # Fit the model
    history = model.fit(
        normed_x_train, y_train,
        epochs=epochs, validation_split = validation_split, verbose=0) 
    # print(history.history)

    if goal == 'Regression':
        # Predict
        y_train_pred = model.predict(normed_x_train).flatten()
        y_test_pred =  model.predict(normed_x_test).flatten()

        train_rsquare =  np.round(np.square(np.corrcoef(y_train, y_train_pred)[0,1]), 3)
        train_mse =  np.round(np.square(np.subtract(y_train, y_train_pred)).mean(), 3)
        test_rsquare = np.round(np.square(np.corrcoef(y_test, y_test_pred)[0,1]), 3)
        test_mse = np.round(np.square(np.subtract(y_test, y_test_pred)).mean(), 3)

        metrics = {'train_rsquare':train_rsquare, 'train_mse':train_mse, 'test_rsquare':test_rsquare, 'test_mse':test_mse}

    elif goal == 'Classification': ###############################################################
        # Predict
        y_train_pred = model.predict(normed_x_train)
        y_test_pred =  model.predict(normed_x_test)
        
        y_train_pred = np.argmax(y_train_pred, axis=1) # softmax result -> classification
        y_test_pred = np.argmax(y_test_pred, axis=1)

        train_accuracy = np.round(accuracy_score(y_train, y_train_pred), 3)
        train_f1 = np.round(f1_score(y_train, y_train_pred, average="weighted"), 3)

        test_accuracy = np.round(accuracy_score(y_test, y_test_pred), 3)
        test_f1 = np.round(f1_score(y_test, y_test_pred, average="weighted"), 3)
        metrics = {'train_accuracy':train_accuracy, 'train_f1':train_f1, 'test_accuracy':test_accuracy, 'test_f1':test_f1 }

    duration = time.time() - t0

    model.save(f'tmp_result/model.h5', )
    return model, duration, y_train_pred, y_test_pred, history, metrics

def train_classification_model(model, x_train, y_train, x_test, y_test):
    t0 = time.time()
    model.fit(x_train, y_train)
    duration = time.time() - t0
    y_train_pred = model.predict(x_train)
    y_test_pred = model.predict(x_test)

    train_accuracy = np.round(accuracy_score(y_train, y_train_pred), 3)
    train_f1 = np.round(f1_score(y_train, y_train_pred, average="weighted"), 3)

    test_accuracy = np.round(accuracy_score(y_test, y_test_pred), 3)
    test_f1 = np.round(f1_score(y_test, y_test_pred, average="weighted"), 3)

    return model, train_accuracy, test_accuracy, train_f1, test_f1, duration, y_train_pred, y_test_pred

def train_regression_model(model, x_train, y_train, x_test, y_test):
    t0 = time.time()

    model.fit(x_train, y_train)

    y_train_pred = model.predict(x_train)
    y_test_pred = model.predict(x_test)

    train_rsquare = np.round(np.square(np.corrcoef(y_train, y_train_pred)[0, 1]), 3)
    train_mse = np.round(np.square(np.subtract(y_train, y_train_pred)).mean(), 3)
    test_rsquare = np.round(np.square(np.corrcoef(y_test, y_test_pred)[0, 1]), 3)
    test_mse = np.round(np.square(np.subtract(y_test, y_test_pred)).mean(), 3)

    duration = time.time() - t0

    pickle.dump(model, open('tmp_result/model.pkl', 'wb'))



    return model, train_rsquare, test_rsquare, train_mse, test_mse, duration, y_train_pred, y_test_pred


def img_to_bytes(img_path):
    img_bytes = Path(img_path).read_bytes()
    encoded = base64.b64encode(img_bytes).decode()
    return encoded


def get_model_tips(model_type):
    model_tips = model_infos[model_type]
    return model_tips


def get_model_url(model_type):
    model_url = model_urls[model_type]
    model_to_pkg = {'Keras Neural Network':'keras', 'SVR': 'scikit-learn', 'Linear Regression': 'scikit-learn', 'SVC': 'scikit-learn'}
    text = f"**Link to {model_to_pkg[model_type]} official documentation [here]({model_url}) 💻 **"
    return text


def set_sidebar_width(width):
    st.markdown(f'''
        <style>
            section[data-testid="stSidebar"] .css-ng1t4o {{width: {width}rem;}}
            section[data-testid="stSidebar"] .css-1d391kg {{width: {width}rem;}}
        </style>
    ''',unsafe_allow_html=True)
    
# def output_csv():
#     with open('tmp_result/params.txt', 'r') as f:
#         params = f.read()
#     with open('tmp_result/x.txt', 'r') as f:
#         x = f.read()
#     with open('tmp_result/y.txt', 'r') as f:
#         y = f.read()
#     with open('tmp_result/metrics.txt', 'r') as f:
#         metrics = f.read()
    
#     metrics = metrics.split(',')[0:4] # train rs, train mse, test sq, test mse
    
#     evaluation = pd.DataFrame({'params':params, 'y':y, 'x':x, 'rsq.train':metrics[0], 'mse.train':metrics[1], 'rsq.test':metrics[2], 'mse.test':metrics[3]}, index=[0])
#     evaluation.to_csv('tmp_result/evaluation_result.csv', sep=',', index=False)