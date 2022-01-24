# import python libraries 
import numpy as np
import pandas as pd
from pandas._libs.missing import NA
from sqlalchemy import null
import streamlit as st
import copy
import matplotlib.pyplot as plt

# import user-defined parameter selectors
from models.Regression import regression_param_selector
from models.SVR import svr_param_selector
from models.kerasNN import knn_param_selector
from sklearn.model_selection import train_test_split

# import user-defined functions
from models.utils import model_imports
from utils.functions import img_to_bytes

@st.experimental_memo
def read_csv(path):
    return pd.read_csv(path)

def introduction():
    st.title("** Machine Learning Models for HTP phenotype prediction **")
    st.subheader("Predict phenotypes from indices and assess the accuracy of predictions")
    st.image('./images/field.tif')
    st.markdown(
        """
    - 🗂️ Choose/upload a dataset
    - ⚙️  Pick a phenotype to predict, predictors, model and set its hyper-parameters
    - 📉 Train it and check its performance metrics  on train and test data
    - 🩺 Diagnose possible overfitting and experiment with other settings
    -----
    """
    )

# current_data: pd.DataFrame = None
def dataset_selector():
    dataset_container = st.sidebar.expander("Configure a dataset", True)
    global current_data
    current_data = None

    with dataset_container:
        
        uploaded_file = st.file_uploader("Upload CSV file", key='data_uploader')

        if uploaded_file is not None:
            
            current_data = read_csv(uploaded_file)
            # current_data = read_csv(uploaded_file)
            # n_samples = current_data.shape[0]
            # # From  versions above 1 use session state
            # if 'cur_data' not in st.session_state:
            #    st.session_state.cur_data = current_data
            dataset = "upload"
        else:
            st.write("#### Or, choose a pre-loaded dataset")
            dataset = st.selectbox("Choose a dataset", options=("2016 DS","aa")) #########################################
            if dataset == "2016 DS":
                current_data = read_csv('data/_output.csv')

        #else:
        #    n_samples = 200 # number of samples for regress data = 200 (will be deleted/replaced later)

        #if uploaded_file is not None:
            # current_data = read_csv(uploaded_file)
            # n_samples = current_data.shape[0]
            #dataset = "upload"
    return current_data


# def model_selector(input_shape=None):
#     model_training_container = st.sidebar.expander("Train a model", True)
#     with model_training_container:
#         model_type = st.selectbox(
#             "Choose a model",
#             (
#                 "Keras Neural Network",
#                 "SVR",
#             ),
#         )
#         if model_type == "Keras Neural Network":
#             model = knn_param_selector(input_shape=input_shape)

#         elif model_type == "SVR":
#             model = svr_param_selector()

#     return model_type, model

def model_selector():
    model_training_container = st.sidebar.expander("Choose a model", True)
    with model_training_container:
        model_type = st.selectbox(
            "Models",
            (   
                "Linear Regression",
                "Keras Neural Network",
                "SVR",
            ),
        )

    return model_type


def parameter_selector(model_type, input_shape=None):
    if model_type == "Linear Regression":
        model = regression_param_selector()
    
    elif model_type == "Keras Neural Network":
        model = knn_param_selector(input_shape=input_shape)

    elif model_type == "SVR":
        model = svr_param_selector()
    
    return model

# def polynomial_degree_selector():
#     return st.sidebar.number_input("Highest polynomial degree", 1, 10, 1, 1)


def footer():
    st.sidebar.markdown("---")
    st.sidebar.markdown(
        """
        [<img src='data:image/png;base64,{}' class='img-fluid' width=25 height=25>](https://github.com/ahmedbesbes/playground) <small> Based on Playground 0.1.0 </small>""".format(
            img_to_bytes("./images/github.png")
        ),
        unsafe_allow_html=True,
    )


# -----------------------------------------------------------------------------------------------------------------
def column_selector(current_data):

    st.subheader("Choose phenotype to predict, and predictors")

    st.dataframe(data=current_data.head())

    col_names = list(current_data.columns)

    col1, col2 = st.columns((1, 4))
    with col1:
        yy = st.selectbox(label='Phenotype to predict (Y)', options=col_names, index=10)
    with col2:
        xx = st.text_input(label='Predictors (X)', value=f'{col_names[8]}, {col_names[9]}, {col_names[12]}:{col_names[14]}')
    
    st.markdown(
        """
    - Pick a variable from the dataset
    - Y: select a numeric variable for prediction (e.g. `HT.x`)
    - X: use `,` for single selection, and `:` for sequence selection
        - For example, `PC1, PC2, R.x:B.x` selects PC1, PC2, and all variables between R.x and B.x 
    - Selected predictors (X)
    """
    )

    # Parse the selected predictors text input data
    xx = list(map(str.strip, str.split(xx, sep=',')))
    xx_idx = list() # index of selected predictors
    for element in xx:
        if element.find(':') == -1:
            xx_idx.append(col_names.index(element))
        else:
            element = list(map(str.strip, str.split(element, sep=':')))
            idx = [col_names.index(ii) for ii in element]
            xx_idx += (list(range(idx[0], idx[1]+1)))        

    yy_idx = col_names.index(yy)

    # remove rows with NA
    idx = copy.deepcopy(xx_idx)
    idx.append(yy_idx) # the last column is y
    current_data = current_data.iloc[:,idx].dropna()

    # split y and x
    y = current_data.iloc[:,-1] # the last column
    x = current_data.iloc[:,:-1] # all columns except the last one

    st.dataframe(x)
    st.markdown("""
        - Rows with NA's are removed
    -----
    """
    )

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1234) ############################### test train split

    input_shape = x.shape[1] # number of variables 

    return x_train, y_train, x_test, y_test, input_shape

# -----------------------------------------------------------------------------------------------------------------

