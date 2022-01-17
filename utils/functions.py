from pathlib import Path
import base64
import time
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, r2_score
from sklearn.datasets import make_moons, make_regression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from plotly.subplots import make_subplots
import plotly.graph_objs as go

from models.utils import model_infos, model_urls
# from utils.ui import dataset_selector # for access to "current_data"

def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

@st.cache(
    suppress_st_warning=True,
    allow_output_mutation=True,
    show_spinner=True)
def generate_data(dataset, uploaded_file=None):
    if dataset == "moons":
        n_samples = 200
        x_train, y_train = make_moons(n_samples=n_samples)
        x_test, y_test = make_moons(n_samples=n_samples)

    elif dataset == "regress":
        n_samples = 200
        x_train, y_train = make_regression(n_samples=n_samples)
        x_test, y_test = make_regression(n_samples=n_samples)

    elif dataset == "2016 DS":
        current_data = pd.read_csv('data/_output.csv')
        n_samples = current_data.shape[0] 
        x = current_data.iloc[:,12:20] ########################################## Make options to choose X by user
        y = current_data['HT.x'] ################################################## Make options to choose Y by user
        x_train_pd, x_test_pd, y_train_pd, y_test_pd = train_test_split(x, y, test_size=0.2, random_state=1234)
        
        # convert to numpy array
        x_train = x_train_pd.to_numpy()
        x_test = x_test_pd.to_numpy()
        y_train = y_train_pd.to_numpy()
        y_test = y_test_pd.to_numpy()

    elif dataset == "upload":
        current_data = pd.read_csv(uploaded_file)
        #n_samples = current_data.shape[0]
        #cur_data = st.session_state.cur_data
        #x = cur_data.iloc[:, 0]
        #y = cur_data.iloc[:, 1:]
        #x_train, x_test, y_train, y_test = train_test_split(x, y,
        #                                                   test_size=0.2, random_state=1234)

    return x_train, y_train, x_test, y_test


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

    fig.add_trace(
        go.Indicator(
            mode="gauge+number+delta",
            value=metrics["test_mse"],
            title={"text": f"MSE (test)"},
            domain={"x": [0, 1], "y": [0, 1]},
            gauge={"axis": {"range": [0, 1]}},
            delta={"reference": metrics["train_mse"]},
        ),
        row=2,
        col=2,
    )

    fig.update_layout(
        height=700,
    )

    return fig

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

    return model, train_accuracy, train_f1, test_accuracy, test_f1, duration



def train_keras_model(model, x_train, y_train, x_test, y_test):
    t0 = time.time()
    y_train_pred = model.predict(x_train)[:,0]
    y_test_pred = model.predict(x_test)[:,0]
    duration = time.time() - t0
    
    train_rsquare = np.round(r2_score(y_train, y_train_pred), 3)
    train_mse = np.round(np.square(np.subtract(y_train, y_train_pred)).mean(), 3)

    test_rsquare = np.round(r2_score(y_test, y_test_pred), 3)
    test_mse = np.round(np.square(np.subtract(y_test, y_test_pred)).mean(), 3)

    st.text(f'{y_train_pred.shape}, {y_test_pred.shape}, {y_train.shape}, {y_test.shape}') ####################################
    st.text(f'{train_rsquare},{test_rsquare}') ####################################

    return model, train_rsquare, test_rsquare, train_mse, test_mse, duration, y_train_pred, y_test_pred


def train_regression_model(model, x_train, y_train, x_test, y_test):
    t0 = time.time()
    model.fit(x_train, y_train)
    duration = time.time() - t0
    y_train_pred = model.predict(x_train)
    y_test_pred = model.predict(x_test)
    
    # https://scikit-learn.org/dev/modules/generated/sklearn.metrics.r2_score.html
    train_rsquare = np.round(r2_score(y_train, y_train_pred), 3)
    train_mse = np.round(np.square(np.subtract(y_train, y_train_pred)).mean(), 3)

    test_rsquare = np.round(r2_score(y_test, y_test_pred), 3)
    test_mse = np.round(np.square(np.subtract(y_test, y_test_pred)).mean(), 3)

    st.text(f'train_reg_model:{model},{train_rsquare},{test_rsquare}')

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


def add_polynomial_features(x_train, x_test, degree):
    for d in range(2, degree + 1):
        x_train = np.concatenate(
            (
                x_train,
                x_train[:, 0].reshape(-1, 1) ** d,
                x_train[:, 1].reshape(-1, 1) ** d,
            ),
            axis=1,
        )
        x_test = np.concatenate(
            (
                x_test,
                x_test[:, 0].reshape(-1, 1) ** d,
                x_test[:, 1].reshape(-1, 1) ** d,
            ),
            axis=1,
        )
    return x_train, x_test
