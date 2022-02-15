from xmlrpc.client import FastUnmarshaller
import numpy as np
import streamlit as st
import pandas as pd
from pandas.api.types import is_numeric_dtype
import pickle
import keras
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
    # output_csv,
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
    column_display,
    # zip_dir,
    createFolder,
    # download_result,
)

st.set_page_config(
    page_title="Predict Phenotype from HTP indices", layout="wide", page_icon="./images/flask.png"
)


def resize_sidebar(width):
    st.sidebar.markdown(f'''
    <style>
        section[data-testid="stSidebar"] .css-zbg2rx {{width: {width}rem;}}
    </style>
''',unsafe_allow_html=True)

def update_run_counter(goal, y_test):
    if "run_counter" not in st.session_state:
        st.session_state.run_counter = 0

    st.session_state.run_counter +=1
    
    if (goal=='Classification') and (is_numeric_dtype(y_test)):
        st.warning(f'Do you really want to do a classification? Y (test dataset) is numeric with {len(set(y_test.tolist()))} unique values.')

def update_sidebar_size():
    if "size_counter" not in st.session_state:
        st.session_state.size_counter = False

    st.session_state.size_counter = not(st.session_state.size_counter)
    st.sidebar.write(st.session_state.size_counter)

resize_sidebar(22)

def sidebar_controllers():
    
    st.sidebar.info('''
    This deployed version of app is only for testing purposes (1GB memory limit). For big computations, please [download local version](https://github.com/songwan/HTPPs).
    ''')

    with st.sidebar.expander("Configure a dataset", True):
        current_data = dataset_selector() # loads global variable current_data 

    with st.sidebar.form('Options form'):

        x_train, y_train, x_test, y_test, input_shape, goal, nclasses, labely, var_names, ohe_info, x = column_selector(current_data)

        model_type = model_selector(goal)

        validation_split, epochs, model, json_param = parameter_selector(model_type, goal, nclasses, input_shape=input_shape)

        if model_type in ('Keras Neural Network'):
            st.form_submit_button('Expand/shrink sidebar', on_click=update_sidebar_size)
        st.form_submit_button('Update parameters', on_click=update_run_counter(goal, y_test))

        # footer()    
    
    if "size_counter" not in st.session_state:
            st.session_state.size_counter = False

    if model_type in ('Keras Neural Network'):
            
        if st.session_state.size_counter:
            resize_sidebar(55)
        else:
            resize_sidebar(22)

    with st.sidebar.expander('Train model', True):
        st.info('Please make sure that parameters were set as intended befor running the model')
        run_body = st.checkbox('Run the model', False)

    results = {'json_param':json_param, 'name_y':var_names['y'], 'name_x':var_names['x'].tolist(), 'ohe_info':ohe_info}

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
        labely,
        results,
        current_data,
        x,
        run_body,
    )


def body(
    x_train, x_test, y_train, y_test, model, model_type, epochs, validation_split, goal, labely, results, current_data, x, run_body # noise may be interesting, but less important to the users
):
    column_display(current_data, x)

    # if st.session_state.show_result:
    if run_body:
        # introduction()

        st.subheader(f'{model_type} Results')
        col1, col2 = st.columns((1, 1))
        with col1:
            confusion_placeholder = st.empty()
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
        
        st.sidebar.write('Download result')
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

            st.sidebar.download_button(label = 'Download model (pkl or h5)', data = pickle.dumps(model), file_name = 'model.pkl')
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

            model.save('model.h5')
            with open('model.h5', 'rb') as fp:
                st.sidebar.download_button(label = 'Download model (pkl or h5)', data = fp, file_name = 'model.h5')

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

            st.sidebar.download_button(label = 'Download model (pkl or h5)', data = pickle.dumps(model), file_name = 'model.pkl')

        # with open('tmp_result/metrics.txt', 'w') as f:
        #     for item in metrics.values(): # train rs, train mse, test sq, test mse
        #         f.write(f'{item},')

        model_tips = get_model_tips(model_type)

        if model_type in ('SVR', 'Linear Regression'):
            fig = plot_prediction_and_metrics(y_train, y_test, metrics, y_train_pred, y_test_pred)

        elif model_type in ('Keras Neural Network'):

            if goal=='Regression':   
                fig = plot_prediction_and_metrics(y_train, y_test, metrics, y_train_pred, y_test_pred)

            elif goal=='Classification':  
                fig, fig_conf = plot_classification_and_metrics(y_train, y_test, metrics, y_train_pred, y_test_pred, labely)
                confusion_placeholder.pyplot(fig_conf, True)

            fig_history = plot_history(history)
            history_placeholder.pyplot(fig_history, True)

        elif model_type in ('SVC'):
            fig, fig_conf = plot_classification_and_metrics(y_train, y_test, metrics, y_train_pred, y_test_pred, labely)
            confusion_placeholder.pyplot(fig_conf, True)

        plot_placeholder.plotly_chart(fig, True)
        duration_placeholder.warning(f"Training took {duration:.3f} seconds")
        model_url_placeholder.markdown(model_url)
        # code_header_placeholder.header("**Retrain the same model in Python**")
        # snippet_placeholder.code(snippet)
        tips_header_placeholder.subheader(f"**Tips on the {model_type} 💡 **")
        tips_placeholder.info(model_tips)
        
        predicted_values = pd.concat([y_test.reset_index(drop=False, inplace=False), pd.DataFrame(y_test_pred)], axis=1, ignore_index=True)
        predicted_values.columns = ['index', 'y_test', 'pred']
        

        # save result
        lebely3 = None
        if labely is not None:
            labely2 = labely.transpose()
            labely2.columns = ['ylevels']
            lebely3 = str(labely2.to_dict()['ylevels'])

        dict_result = {
            'param': results['json_param'],
            'name_x': results['name_x'],
            'name_y': results['name_y'],
            'test_idx': predicted_values['index'].astype(str).str.cat(sep=','),
            'test_y': predicted_values['y_test'].astype(str).str.cat(sep=','),
            'test_pred': predicted_values['pred'].astype(str).str.cat(sep=','),
            'encoding_y': lebely3
        }
        
        # st.write(metrics)
        df_result = pd.concat([pd.DataFrame([dict_result]), pd.DataFrame([metrics])], axis=1)
        df_result = df_result.to_csv(index=False).encode('utf-8')
        st.sidebar.download_button('Download model information (csv)', data=df_result, file_name='result.csv')

    else:
        st.subheader('Results')
        st.warning("To display the result, select  the checkbox")

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
        labely,
        results,
        current_data,
        x,
        run_body,
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
        labely,
        results,
        current_data,
        x,
        run_body,
    )
