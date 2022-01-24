import numpy as np
import streamlit as st
from utils.functions import (
    # add_polynomial_features,
    # generate_data,
    get_model_tips,
    get_model_url,
    plot_prediction_and_metrics,
    #train_classification_model,
    train_keras_model,
    train_regression_model,
)

from utils.ui import (
    dataset_selector,
    footer,
    #generate_snippet,
    # polynomial_degree_selector,
    introduction,
    model_selector,
    parameter_selector,
    column_selector,
)

st.set_page_config(
    page_title="Predict Phenotype from HTP indices", layout="wide", page_icon="./images/flask.png"
)

def sidebar_controllers():


    current_data = dataset_selector() # loads global variable current_data 
    #if dataset == "upload":
    # x_train, y_train, x_test, y_test, current_data = generate_data(current_data)    
    
    # -----------------------------------------------------------------------------------------------------------------
    x_train, y_train, x_test, y_test, input_shape = column_selector(current_data)

    # if dataset == '2016 DS':
    #     x_train, y_train, x_test, y_test, input_shape = column_selector(current_data)
        
    # else: ################# delete later
    #     input_shape = x_train.shape[1]
    # # -----------------------------------------------------------------------------------------------------------------

    # model_type, model = model_selector(input_shape=input_shape) 
    model_type = model_selector()
    model = parameter_selector(model_type, input_shape=input_shape)
    
    # st.sidebar.header("Feature engineering")
    # degree = polynomial_degree_selector()
    footer()

    return (
        #dataset,
        model_type,
        model,
        x_train,
        y_train,
        x_test,
        y_test,
        #degree,
    )


def body(
    x_train, x_test, y_train, y_test, model, model_type # noise may be interesting, but less important to the users
):
    # introduction()

    st.subheader(f'{model_type} Results')
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

    # x_train, x_test = add_polynomial_features(x_train, x_test, degree)
    model_url = get_model_url(model_type)
    
    # Regression Models -> R-squared, MSE
    if model_type in ('SVR'):
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

    # Keras NN -> R-squared, MSE
    elif model_type in ('Keras Neural Network'):
        (   
            model, 
            train_rsquare, 
            test_rsquare, 
            train_mse, 
            test_mse, 
            duration,
            y_train_pred,
            y_test_pred,
        ) = train_keras_model(model, x_train, y_train, x_test, y_test)

        metrics = {
            "train_rsquare": train_rsquare, 
            "train_mse": train_mse, 
            "test_rsquare": test_rsquare, 
            "test_mse": test_mse, 
        }

    model_tips = get_model_tips(model_type)

    fig = plot_prediction_and_metrics(
        y_train, y_test, metrics, y_train_pred, y_test_pred
    )

    plot_placeholder.plotly_chart(fig, True)
    duration_placeholder.warning(f"Training took {duration:.3f} seconds")
    model_url_placeholder.markdown(model_url)
    # code_header_placeholder.header("**Retrain the same model in Python**")
    # snippet_placeholder.code(snippet)
    tips_header_placeholder.subheader(f"**Tips on the {model_type} 💡 **")
    tips_placeholder.info(model_tips)



def header():
    introduction()

if __name__ == "__main__":
    header()
    (
        #dataset,
        model_type,
        model,
        x_train,
        y_train,
        x_test,
        y_test,
        # degree,
    ) = sidebar_controllers()
    body(
        x_train,
        x_test,
        y_train,
        y_test,
        # degree,
        model,
        model_type,
    )