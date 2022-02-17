from xmlrpc.client import FastUnmarshaller
import streamlit as st
import pandas as pd
import pickle

from utils.functions import (
    get_model_tips,
    get_model_url,
    plot_prediction_and_metrics,
    plot_classification_and_metrics,
    plot_history,
    train_classification_model,
    train_keras_model,
    train_regression_model,
)

from utils.ui import (
    dataset_selector,
    introduction,
    model_selector,
    parameter_selector,
    column_selector,
    column_display,
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


def update_run_counter(goal, labely):
    if "run_counter" not in st.session_state:
        st.session_state.run_counter = 0

    st.session_state.run_counter +=1
    
    if labely is not None:
        nclasses = len(labely.transpose())
    else:
        nclasses = None

    if (goal=='Classification') and (nclasses > 20):
        st.warning(f'Do you really want a classification? The number of class is {nclasses}.')


def update_sidebar_size():
    if "size_counter" not in st.session_state:
        st.session_state.size_counter = False
    st.session_state.size_counter = not(st.session_state.size_counter)

resize_sidebar(22)

def sidebar_controllers():
    
    st.sidebar.info('''
    This deployed version of app is only for testing purposes (1GB memory limit). 
    For big computations, please [download local version](https://github.com/songwan/HTPPs).
    ''')

    with st.sidebar.expander("1. Configure a dataset", True):
        current_data = dataset_selector() # loads global variable current_data 

    with st.sidebar.expander('2. Select X and Y', True):
        x_train, y_train, x_test, y_test, input_shape, goal, nclasses, labely, var_names, ohe_info, x, train_test_ratio = column_selector(current_data)

    with st.sidebar.expander('3. Select Model', True):
        model_type = model_selector(goal)

    with st.sidebar.expander('4. Select parameters', True):
        validation_split, epochs, batch_size, model, json_param = parameter_selector(model_type, goal, nclasses, input_shape=input_shape)

    with st.sidebar.form('Expand/shrink sidebar'):
        if model_type in ('Keras Neural Network'):
            st.form_submit_button('Expand/shrink sidebar', on_click=update_sidebar_size)


    if "size_counter" not in st.session_state:
            st.session_state.size_counter = False

    if model_type in ('Keras Neural Network'):
            
        if st.session_state.size_counter:
            resize_sidebar(55)
        else:
            resize_sidebar(22)

    # with st.sidebar.expander('5. Train model', True):
    #     st.info('Please make sure that parameters were set as intended befor running the model')
    #     run_body = st.checkbox('Run the model (always rerun)', False)

    with st.sidebar.form('Train model'):
        st.write('5. Train model')
        st.info('Please make sure that parameters were set as intended befor running the model')
        run_body = st.form_submit_button('Run the model')

    results = {'json_param':json_param, 'name_y':var_names['y'], 'name_x':var_names['x'].tolist(), 'ohe_info':ohe_info, 'test_data_ratio': train_test_ratio}

    return (
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
        batch_size,
        ohe_info,
    )


def body(
    x_train, x_test, y_train, y_test, model, model_type, epochs, validation_split, goal, labely, results, current_data, x, run_body, batch_size, ohe_info # noise may be interesting, but less important to the users
):
    y_enc_origin, y_enc_new, labely = column_display(current_data, x, ohe_info, labely, goal)

    # if new encoding was applied, update y_train, y_test
    if y_enc_new is not None:
        y_enc_dict = dict(zip(y_enc_origin, y_enc_new))
        y_train = y_train.replace(y_enc_dict)
        y_test = y_test.replace(y_enc_dict)

    # if st.session_state.show_result:
    if run_body:
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
            ) = train_keras_model(model, x_train, y_train, x_test, y_test, epochs, validation_split, batch_size, goal)

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

        model_tips = get_model_tips(model_type)

        if model_type in ('SVR', 'Linear Regression'):
            fig = plot_prediction_and_metrics(y_train, y_test, metrics, y_train_pred, y_test_pred)

        elif model_type in ('Keras Neural Network'):

            if goal=='Regression':   
                fig = plot_prediction_and_metrics(y_train, y_test, metrics, y_train_pred, y_test_pred)

            elif goal=='Classification':  
                fig, fig_conf = plot_classification_and_metrics(y_train, y_test, metrics, y_train_pred, y_test_pred, labely)
                confusion_placeholder.pyplot(fig_conf, True)

            fig_history = plot_history(history, goal)
            history_placeholder.pyplot(fig_history, True)

        elif model_type in ('SVC'):
            fig, fig_conf = plot_classification_and_metrics(y_train, y_test, metrics, y_train_pred, y_test_pred, labely)
            confusion_placeholder.pyplot(fig_conf, True)

        plot_placeholder.plotly_chart(fig, True)
        duration_placeholder.warning(f"Training took {duration:.3f} seconds")
        model_url_placeholder.markdown(model_url)
        tips_header_placeholder.subheader(f"Tips on the {model_type} 💡")
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
            'test_data_ratio': results['test_data_ratio'],
            'test_idx': predicted_values['index'].astype(str).str.cat(sep=','),
            'test_y': predicted_values['y_test'].astype(str).str.cat(sep=','),
            'test_pred': predicted_values['pred'].astype(str).str.cat(sep=','),
            'encoding_y': lebely3
        }
        
        df_result = pd.concat([pd.DataFrame([dict_result]), pd.DataFrame([metrics])], axis=1)
        df_result = df_result.to_csv(index=False).encode('utf-8')
        st.sidebar.download_button('Download model information (csv)', data=df_result, file_name='result.csv')

    else:
        st.subheader('Results')
        st.info("To display the result, select  the checkbox")


def header():
    introduction()


if __name__ == "__main__":
    header()
    (
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
        batch_size,
        ohe_info,
    ) = sidebar_controllers()
    body(
        x_train,
        x_test,
        y_train,
        y_test,
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
        batch_size,
        ohe_info,
    )
