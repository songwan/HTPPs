from pathlib import Path
import base64
import time
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, r2_score
from sklearn.datasets import make_moons, make_circles, make_blobs, make_regression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from plotly.subplots import make_subplots
import plotly.graph_objs as go

from models.utils import model_infos, model_urls


def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


@st.cache(
    suppress_st_warning=True,
    allow_output_mutation=True,
    show_spinner=True)
def generate_data(dataset, n_samples, n_classes, uploaded_file=None):
    if dataset == "moons":
        x_train, y_train = make_moons(n_samples=n_samples)
        x_test, y_test = make_moons(n_samples=n_samples)
    elif dataset == "circles":
        x_train, y_train = make_circles(n_samples=n_samples)
        x_test, y_test = make_circles(n_samples=n_samples)
    elif dataset == "regress":
        x_train, y_train = make_regression(n_samples=n_samples)
        x_test, y_test = make_regression(n_samples=n_samples)
    elif dataset == "blobs":
        x_train, y_train = make_blobs(
            n_features=2,
            n_samples=n_samples,
            centers=n_classes,
            cluster_std=1, 
            random_state=42,
        )
        x_test, y_test = make_blobs(
            n_features=2,
            n_samples=n_samples // 2,
            centers=2,
            cluster_std=1,
            random_state=42,
        )

        scaler = StandardScaler()
        x_train = scaler.fit_transform(x_train)

        x_test = scaler.transform(x_test)
    elif dataset == "upload":
        current_data = pd.read_csv(uploaded_file)
        # n_samples = current_data.shape[0]
        # cur_data = st.session_state.cur_data
        # x = cur_data.iloc[:, 0]
        # y = cur_data.iloc[:, 1:]
        # x_train, x_test, y_train, y_test = train_test_split(x, y,
        #                                                    test_size=0.2, random_state=1234)

    return x_train, y_train, x_test, y_test


# Plotting decision boundary is relevant for classification tasks

def plot_decision_boundary_and_metrics(
        model, x_train, y_train, x_test, y_test, metrics, model_type
):
    d = x_train.shape[1]

    x_min, x_max = x_train[:, 0].min() - 1, x_train[:, 0].max() + 1
    y_min, y_max = x_train[:, 1].min() - 1, x_train[:, 1].max() + 1

    h = 0.02

    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    y_ = np.arange(y_min, y_max, h)

    model_input = [(xx.ravel() ** p, yy.ravel() ** p) for p in range(1, d // 2 + 1)]
    aux = []
    for c in model_input:
        aux.append(c[0])
        aux.append(c[1])

    Z = model.predict(np.concatenate([v.reshape(-1, 1) for v in aux], axis=1))

    Z = Z.reshape(xx.shape)

    fig = make_subplots(
        rows=2,
        cols=2,
        specs=[[{"colspan": 2}, None], [{"type": "indicator"}, {"type": "indicator"}]],
        subplot_titles=("Decision Boundary", None, None),
        row_heights=[0.7, 0.30],
    )

    heatmap = go.Heatmap(
        x=xx[0],
        y=y_,
        z=Z,
        colorscale=["tomato", "rgb(27,158,119)"],
        showscale=False,
    )

    train_data = go.Scatter(
        x=x_train[:, 0],
        y=x_train[:, 1],
        name="train data",
        mode="markers",
        showlegend=True,
        marker=dict(
            size=10,
            color=y_train,
            colorscale=["tomato", "green"],
            line=dict(color="black", width=2),
        ),
    )

    test_data = go.Scatter(
        x=x_test[:, 0],
        y=x_test[:, 1],
        name="test data",
        mode="markers",
        showlegend=True,
        marker_symbol="cross",
        visible="legendonly",
        marker=dict(
            size=10,
            color=y_test,
            colorscale=["tomato", "green"],
            line=dict(color="black", width=2),
        ),
    )

    fig.add_trace(heatmap, row=1, col=1, ).add_trace(train_data).add_trace(
        test_data
    ).update_xaxes(range=[x_min, x_max], title="x1").update_yaxes(
        range=[y_min, y_max], title="x2"
    )
    
    # Classification Models
    if model_type in ('Neural Network'):

        fig.add_trace(
            go.Indicator(
                mode="gauge+number+delta",
                value=metrics['test_accuracy'],
                title={"text": f"Accuracy (test)"},
                domain={"x": [0, 1], "y": [0, 1]},
                gauge={"axis": {"range": [0, 1]}},
                delta={"reference": metrics['train_accuracy']},
            ),
            row=2,
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
            row=2,
            col=2,
        )

    # Regression Models
    elif model_type in ("SVR"):
        fig.add_trace(
            go.Indicator(
                mode="gauge+number+delta",
                value=metrics['test_rsquare'],
                title={"text": f"R-squared (test)"},
                domain={"x": [0, 1], "y": [0, 1]},
                gauge={"axis": {"range": [0, 1]}},
                delta={"reference": metrics['train_rsquare']},
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


def train_regression_model(model, x_train, y_train, x_test, y_test):
    t0 = time.time()

    model.fit(x_train, y_train)
    duration = time.time() - t0
    y_train_pred = model.predict(x_train)
    y_test_pred = model.predict(x_test)

    train_rsquare = np.round(r2_score(y_train, y_train_pred), 3)
    train_mse = np.round(np.square(np.subtract(y_train, y_train_pred)).mean(), 3)

    test_rsquare = np.round(r2_score(y_test, y_test_pred), 3)
    test_mse = np.round(np.square(np.subtract(y_test, y_test_pred)).mean(), 3)

    return model, train_rsquare, test_rsquare, train_mse, test_mse, duration


#def train_model(model, x_train, y_train, x_test, y_test):
#    t0 = time.time()
#
#    model.fit(x_train, y_train)
#    duration = time.time() - t0
#    y_train_pred = model.predict(x_train)
#    y_test_pred = model.predict(x_test)
#
#    train_accuracy = np.round(accuracy_score(y_train, y_train_pred), 3)
#    train_f1 = np.round(f1_score(y_train, y_train_pred, average="weighted"), 3)
#
#    test_accuracy = np.round(accuracy_score(y_test, y_test_pred), 3)
#    test_f1 = np.round(f1_score(y_test, y_test_pred, average="weighted"), 3)
#
#    return model, train_accuracy, train_f1, test_accuracy, test_f1, duration


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
