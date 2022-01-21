from logging.config import valid_ident
from pathlib import Path
import base64
from tabnanny import verbose
import time
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.datasets import make_regression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow import keras
from plotly.subplots import make_subplots
import plotly.graph_objs as go

from models.utils import model_infos, model_urls
# from utils.ui import dataset_selector # for access to "current_data"

def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

@st.experimental_memo
def generate_data(dataset, uploaded_file=None):
    if dataset == "regress":
        n_samples = 200
        x_train, y_train = make_regression(n_samples=n_samples)
        x_test, y_test = make_regression(n_samples=n_samples)
        current_data = None

    elif dataset == "2016 DS":
        current_data = pd.read_csv('data/_output.csv')
        x_train = None
        x_test = None
        y_train = None
        y_test = None

    elif dataset == "upload":
        current_data = pd.read_csv(uploaded_file)
        #n_samples = current_data.shape[0]
        #cur_data = st.session_state.cur_data
        #x = cur_data.iloc[:, 0]
        #y = cur_data.iloc[:, 1:]
        #x_train, x_test, y_train, y_test = train_test_split(x, y,
        #                                                   test_size=0.2, random_state=1234)

    return x_train, y_train, x_test, y_test, current_data


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

    # st.text(f'{metrics["train_rsquare"]}, {metrics["test_rsquare"]}')
    # st.write('y_train_var', y_train.var())
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

def train_keras_model(model, x_train, y_train, x_test, y_test):
    t0 = time.time()
    
    # https://www.tensorflow.org/tutorials/keras/regression

    # Normalize data
    def norm(x):
        return (x - train_stats['mean']) / train_stats['std']

    train_stats = x_train.describe().transpose()
    normed_x_train = norm(x_train)
    normed_x_test = norm(x_test)

    epochs = 10 #################################################### change as a parameter

    # Fit the model
    history = model.fit(
        normed_x_train, y_train,
        epochs=epochs, validation_split = 0.2, verbose=0) #################################################### change as a parameter

    # Predict
    y_train_pred = model.predict(normed_x_train).flatten()
    y_test_pred =  model.predict(normed_x_test).flatten()

    # https://stackoverflow.com/questions/44843581/what-is-the-difference-between-model-fit-an-model-evaluate-in-keras
    # - model.fit(): for training the model with the given inputs
    # - model.evaluate(): for evaluating the already trained model using the validation (or test) data. Returns loss value and metrics values
    # - model.predict(): for actual prediction. It generates output predictions for the input samples
    # -------------------------------------------------------------------------------------

    train_rsquare =  np.round(np.square(np.corrcoef(y_train, y_train_pred)[0,1]), 3)
    train_mse =  np.round(np.square(np.subtract(y_train, y_train_pred)).mean(), 3)
    test_rsquare = np.round(np.square(np.corrcoef(y_test, y_test_pred)[0,1]), 3)
    test_mse = np.round(np.square(np.subtract(y_test, y_test_pred)).mean(), 3)

    duration = time.time() - t0


    return model, train_rsquare, test_rsquare, train_mse, test_mse, duration, y_train_pred, y_test_pred


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
    text = f"**Link to scikit-learn official documentation [here]({model_url}) 💻 **"
    return text


def set_sidebar_width(width):
    st.markdown(f'''
        <style>
            section[data-testid="stSidebar"] .css-ng1t4o {{width: {width}rem;}}
            section[data-testid="stSidebar"] .css-1d391kg {{width: {width}rem;}}
        </style>
    ''',unsafe_allow_html=True)
    
