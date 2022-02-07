import numpy as np
import streamlit as st
import pandas as pd
from utils.functions import (
    # add_polynomial_features,
    # generate_data,
    get_model_tips,
    get_model_url,
    plot_prediction_and_metrics,
    plot_classification_and_metrics,
    plot_history,
    train_classification_model,
    train_keras_model,
    train_regression_model,
    output_csv,
    set_sidebar_width,
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
    zip_dir,
    createFolder,
    download_result,
)

st.set_page_config(
    page_title="Predict Phenotype from HTP indices", layout="wide", page_icon="./images/flask.png"
)

def update_expand_counter():
    st.session_state.expand_clicked = not(st.session_state.expand_clicked)

def expend_sidebar():
    
    with st.sidebar.form("expand_form"):    
        if 'expand_clicked' not in st.session_state:
            st.session_state.expand_clicked = False

        st.form_submit_button('Expand/shrink sidebar', on_click=update_expand_counter)

    if st.session_state.expand_clicked:
        set_sidebar_width(60)
    else:
        set_sidebar_width(21)

def sidebar_controllers():

    createFolder('tmp_result') # temporary folder to save result

    current_data = dataset_selector() # loads global variable current_data 

    x_train, y_train, x_test, y_test, input_shape, goal, nclasses = column_selector(current_data)


    model_type = model_selector(goal)

    if model_type in ('Keras Neural Network'):
        expend_sidebar()

    validation_split, epochs, model = parameter_selector(model_type, goal, nclasses, input_shape=input_shape)

    footer()    

    return (
        #dataset,
        model_type,
        model,
        x_train,
        y_train,
        x_test,
        y_test,
        epochs,
        validation_split,
        goal,
        #degree,
    )


def body(
    x_train, x_test, y_train, y_test, model, model_type, epochs, validation_split, goal # noise may be interesting, but less important to the users
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
        history_placeholder = st.empty()

    # x_train, x_test = add_polynomial_features(x_train, x_test, degree)
    model_url = get_model_url(model_type)
    
    # 1) Rregression Models -> R-squared, MSE
    if model_type in ('Linear Regression', 'SVR'):
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


    # 2) Keras NN -> R-squared, MSE
    elif model_type in ('Keras Neural Network'):
        (   
            model, 
            duration,
            y_train_pred,
            y_test_pred,
            history,
            metrics,
        ) = train_keras_model(model, x_train, y_train, x_test, y_test, epochs, validation_split, goal)
            
        # metrics = {
        #     "train_rsquare": train_rsquare, 
        #     "train_mse": train_mse, 
        #     "test_rsquare": test_rsquare, 
        #     "test_mse": test_mse, 
        # }

    # 3) Classification Models -> f1, accuracy
    elif model_type in ('SVC'):
        (   
            model, 
            train_accuracy, 
            test_accuracy, 
            train_f1, 
            test_f1, 
            duration, 
            y_train_pred, 
            y_test_pred

        ) = train_classification_model(model, x_train, y_train, x_test, y_test)

        metrics = {
            "train_accuracy": train_accuracy, 
            "train_f1": train_f1, 
            "test_accuracy": test_accuracy, 
            "test_f1": test_f1, 
        }

    with open('tmp_result/metrics.txt', 'w') as f:
        for item in metrics.values(): # train rs, train mse, test sq, test mse
            f.write(f'{item},')

    model_tips = get_model_tips(model_type)

    if model_type in ('SVR', 'Linear Regression'):
        fig = plot_prediction_and_metrics(y_train, y_test, metrics, y_train_pred, y_test_pred)

    elif model_type in ('Keras Neural Network'):

        if goal=='Prediction':
            fig = plot_prediction_and_metrics(y_train, y_test, metrics, y_train_pred, y_test_pred)
        elif goal=='Classification':
            fig = plot_classification_and_metrics(y_train, y_test, metrics, y_train_pred, y_test_pred)

        fig_history = plot_history(history)
        history_placeholder.pyplot(fig_history, True)

    elif model_type in ('SVC'):
        fig = plot_classification_and_metrics(y_train, y_test, metrics, y_train_pred, y_test_pred)

    plot_placeholder.plotly_chart(fig, True)
    duration_placeholder.warning(f"Training took {duration:.3f} seconds")
    model_url_placeholder.markdown(model_url)
    # code_header_placeholder.header("**Retrain the same model in Python**")
    # snippet_placeholder.code(snippet)
    tips_header_placeholder.subheader(f"**Tips on the {model_type} 💡 **")
    tips_placeholder.info(model_tips)
    

    # -------------------------------- save files and zip
    predicted_values = pd.concat([y_test.reset_index(drop=False, inplace=False), pd.DataFrame(y_test_pred)], axis=1, ignore_index=True)
    predicted_values.to_csv('tmp_result/predicted_values.csv', header=['index', 'y_test', 'pred'], index=False)
    output_csv()
    zip_dir(f'result/myfile', 'tmp_result') ##################################################
    download_result()

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
        epochs,
        validation_split,
        goal,
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
        epochs,
        validation_split,
        goal,
    )