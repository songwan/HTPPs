# import python libraries 
import numpy as np
import pandas as pd
from pandas._libs.missing import NA
import streamlit as st

# import user-defined parameter selectors
from models.SVR import svr_param_selector
from models.NeuralNetwork import nn_param_selector
from models.kerasNN import knn_param_selector

# import user-defined functions
from models.utils import model_imports
from utils.functions import img_to_bytes



def introduction():
    st.title("** Machine Learning Models for HTP phenotype prediction **")
    st.subheader(
        """
        Using this website you can play with machine learning models to predict phenotypes from indices and assess the accuracy of predictions
        """
    )

    st.markdown(
        """
    - 🗂️ Choose/upload a dataset
    - ⚙️  Pick a model and set its hyper-parameters
    - 📉 Train it and check its performance metrics  on train and test data
    - 🩺 Diagnose possible overfitting and experiment with other settings
    -----
    """
    )

# current_data: pd.DataFrame = None
def dataset_selector():
    dataset_container = st.sidebar.expander("Configure a dataset", True)
    global current_data

    with dataset_container:
        uploaded_file = st.file_uploader("Upload CSV file", key='data_uploader')

        if uploaded_file is not None:
            # current_data = pd.read_csv(uploaded_file)
            # n_samples = current_data.shape[0]
            # # From  versions above 1 use session state
            # if 'cur_data' not in st.session_state:
            #    st.session_state.cur_data = current_data
            dataset = "upload"
                        
        st.write("#### Or, choose a pre-loaded dataset")
        dataset = st.selectbox("Choose a dataset", options=("2016 DS", "regress", "moon"))

        #if dataset == "2016 DS":
           #current_data = pd.read_csv('data/_output.csv')
           #n_samples = current_data.shape[0]

        #else:
        #    n_samples = 200 # number of samples for regress, moons data = 200 (will be deleted/replaced later)

        #if uploaded_file is not None:
            # current_data = pd.read_csv(uploaded_file)
            # n_samples = current_data.shape[0]
            #dataset = "upload"

    return dataset


def model_selector():
    model_training_container = st.sidebar.expander("Train a model", True)
    with model_training_container:
        model_type = st.selectbox(
            "Choose a model",
            (
                "Neural Network", #
                "Keras Neural Network",
                "SVR",
            ),
        )

        if model_type == "Neural Network":
            model = nn_param_selector()

        elif model_type == "Keras Neural Network":
            model = knn_param_selector()

        elif model_type == "SVR":
            model = svr_param_selector()

    return model_type, model


#def generate_snippet(
#    model, model_type, dataset, degree
#):
#
#    model_text_rep = repr(model)
#    model_import = model_imports[model_type]
#
#    if degree > 1:
#        feature_engineering = f"""
#    >>> for d in range(2, {degree+1}):
#    >>>     x_train = np.concatenate((x_train, x_train[:, 0] ** d, x_train[:, 1] ** d))
#    >>>     x_test= np.concatenate((x_test, x_test[:, 0] ** d, x_test[:, 1] ** d))
#    """
#
#    if dataset == "moons":
#        dataset_import = "from sklearn.datasets import make_moons"
#        train_data_def = (
#            f"x_train, y_train = make_moons(n_samples={200})"
#        )
#        test_data_def = f"x_test, y_test = make_moons(n_samples={200 // 2})"
#    elif dataset == "regress":
#        dataset_import = "from sklearn.datasets import make_regression"
#        train_data_def = f"x_train, y_train = make_regression(n_samples={200})"
#        test_data_def = f"x_test, y_test = make_regression(n_samples={200 // 2})"
#    elif dataset == '2016 DS':
#        dataset_import = "from sklearn.datasets import make_regression" ##################################################
#        train_data_def = f"x_train, y_train = make_regression(n_samples={200})" ##################################################
#        test_data_def = f"x_test, y_test = make_regression(n_samples={200 // 2})" ##################################################
#
#
#    snippet = f"""
#    >>> {dataset_import}
#    >>> {model_import}
#    >>> from sklearn.metrics import accuracy_score, f1_score
#
#    >>> {train_data_def}
#    >>> {test_data_def}
#    {feature_engineering if degree > 1 else ''}    
#    >>> model = {model_text_rep}
#    >>> model.fit(x_train, y_train)
#    
#    >>> y_train_pred = model.predict(x_train)
#    >>> y_test_pred = model.predict(x_test)
#    >>> train_accuracy = accuracy_score(y_train, y_train_pred)
#    >>> test_accuracy = accuracy_score(y_test, y_test_pred)
#    """
#    return snippet


def polynomial_degree_selector():
    return st.sidebar.number_input("Highest polynomial degree", 1, 10, 1, 1)


def footer():
    st.sidebar.markdown("---")
    st.sidebar.markdown(
        """
        [<img src='data:image/png;base64,{}' class='img-fluid' width=25 height=25>](https://github.com/ahmedbesbes/playground) <small> Based on Playground 0.1.0 </small>""".format(
            img_to_bytes("./images/github.png")
        ),
        unsafe_allow_html=True,
    )
