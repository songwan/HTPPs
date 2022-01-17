import numpy as np
import streamlit as st
from utils.functions import (
    add_polynomial_features,
    generate_data,
    get_model_tips,
    get_model_url,
    plot_prediction_and_metrics,
    train_classification_model,
    train_regression_model,
)

from utils.ui import (
    dataset_selector,
    footer,
    #generate_snippet,
    polynomial_degree_selector,
    introduction,
    model_selector,
)

st.set_page_config(
    page_title="Predict Phenotype from HTP indices", layout="wide", page_icon="./images/flask.png"
)


def sidebar_controllers():
    dataset = dataset_selector() # loads global variable current_data 
    model_type, model = model_selector() 
    #if dataset == "upload":
    x_train, y_train, x_test, y_test = generate_data(dataset)
    st.sidebar.header("Feature engineering")
    degree = polynomial_degree_selector()
    footer()

    return (
        dataset,
        model_type,
        model,
        x_train,
        y_train,
        x_test,
        y_test,
        degree,
    )


def body(
    x_train, x_test, y_train, y_test, degree, model, model_type # noise may be interesting, but less important to the users
):
    introduction()
    col1, col2 = st.columns((1, 1))

    with col1:
        plot_placeholder = st.empty()

    with col2:
        duration_placeholder = st.empty()
        model_url_placeholder = st.empty()
        code_header_placeholder = st.empty()
        snippet_placeholder = st.empty()
        tips_header_placeholder = st.empty()
        tips_placeholder = st.empty()

    x_train, x_test = add_polynomial_features(x_train, x_test, degree)
    model_url = get_model_url(model_type)

    # Classification Models -> Accuracy, F1 score
    if model_type in ('Neural Network'):
        (
            model,
            train_accuracy,
            train_f1,
            test_accuracy,
            test_f1,
            duration,
        ) = train_classification_model(model, x_train, y_train, x_test, y_test)

        metrics = {
            "train_accuracy": train_accuracy,
            "train_f1": train_f1,
            "test_accuracy": test_accuracy,
            "test_f1": test_f1,
        }
    
    # Regression Models -> R-squared, MSE
    elif model_type in ('SVR', 'Keras Neural Network'):
        (   
            model, 
            train_rsquare, 
            test_rsquare, 
            train_mse, 
            test_mse, 
            duration,
            y_train_pred,
            y_test_pred,
        ) = train_regression_model(model, x_train, y_train, x_test, y_test)

        metrics = {
            "train_rsquare": train_rsquare,
            "train_mse": train_mse,
            "test_rsquare": test_rsquare,
            "test_mse": test_mse,
        }

    #snippet = generate_snippet(
    #    model, model_type, dataset, degree
    #)
# ----------------------------------------------------------------------------------------------------

    model_tips = get_model_tips(model_type)

    fig = plot_prediction_and_metrics(
        y_train, y_test, metrics, y_train_pred, y_test_pred
    )

    plot_placeholder.plotly_chart(fig, True)
    duration_placeholder.warning(f"Training took {duration:.3f} seconds")
    model_url_placeholder.markdown(model_url)
    # code_header_placeholder.header("**Retrain the same model in Python**")
    # snippet_placeholder.code(snippet)
    tips_header_placeholder.header(f"**Tips on the {model_type} 💡 **")
    tips_placeholder.info(model_tips)


if __name__ == "__main__":
    (
        dataset,
        model_type,
        model,
        x_train,
        y_train,
        x_test,
        y_test,
        degree,
    ) = sidebar_controllers()
    body(
        x_train,
        x_test,
        y_train,
        y_test,
        degree,
        model,
        model_type,
    )