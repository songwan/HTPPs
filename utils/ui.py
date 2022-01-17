# import python libraries 
import numpy as np
import pandas as pd
from pandas._libs.missing import NA
import streamlit as st

# import user-defined parameter selectors
from models.SVR import svr_param_selector
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
                "Keras Neural Network",
                "SVR",
            ),
        )
        if model_type == "Keras Neural Network":
            model = knn_param_selector()

        elif model_type == "SVR":
            model = svr_param_selector()

    return model_type, model

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
