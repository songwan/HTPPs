# import python libraries 
from pickle import TRUE
import numpy as np
import pandas as pd
from pandas._libs.missing import NA
from sqlalchemy import null
import streamlit as st
import copy
import matplotlib.pyplot as plt
import zipfile
import os
import shutil

# import user-defined parameter selectors
from models.Regression import regression_param_selector
from models.SVR import svr_param_selector
from models.SVC import svc_param_selector
from models.kerasNN import knn_param_selector
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

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

def dataset_selector():
    dataset_container = st.sidebar.expander("Configure a dataset", True)
    global current_data
    current_data = None

    with dataset_container:
        
        uploaded_file = st.file_uploader("Upload CSV file", key='data_uploader')

        if uploaded_file is not None:
            
            current_data = read_csv(uploaded_file)
            dataset = "upload"
        else:
            st.write("#### Or, choose a pre-loaded dataset")
            dataset = st.selectbox("Choose a dataset", options=("2016 DS","iris")) #########################################
            if dataset == "2016 DS":
                current_data = read_csv('data/2016DS_merged.csv')
            elif dataset == 'iris':
                current_data = read_csv('data/iris.csv')

    return current_data


def model_selector(goal):
    model_training_container = st.sidebar.expander("Choose a model", True)
    with model_training_container:
        models = {'Prediction':['Linear Regression', 'Keras Neural Network', 'SVR'], 'Classification':['SVC', 'Keras Neural Network']}
        model_type = st.selectbox("Models", models[goal])
    return model_type

def update_run_counter():
    st.session_state.run_counter +=1


def parameter_selector(model_type, goal, nclasses, input_shape=None):

    with st.sidebar.expander('Set parameters', True):
        with st.sidebar.form("run_form2"):
            st.form_submit_button('Apply', on_click=update_run_counter)
        
            if "run_counter" not in st.session_state:
                st.session_state.run_counter = 0

            epochs = None
            validation_split = None
            
            if model_type == "Linear Regression":
                model = regression_param_selector()

            elif model_type == "Keras Neural Network":
                validation_split, epochs, model = knn_param_selector(goal, nclasses, input_shape=input_shape)
            
            elif model_type == "SVR":
                model = svr_param_selector()

            elif model_type == 'SVC':
                model = svc_param_selector()

            return validation_split, epochs, model

def footer():
    st.sidebar.markdown("---")
    st.sidebar.markdown(
        """
        [<img src='data:image/png;base64,{}' class='img-fluid' width=25 height=25>](https://github.com/songwan/HTPPs) <small> Based on Playground 0.1.0 </small>""".format(
            img_to_bytes("./images/github.png")
        ),
        unsafe_allow_html=True,
    )

# -----------------------------------------------------------------------------------------------------------------
def is_categorical(df, colname):
    # if the column is string, then assume it to be categorical
    if df[colname].dtype=='object': 
        return True

    # else assume it numerical
    else:
        return False

def labelencoder(y):
    label_encoder = LabelEncoder()
    cat_y = label_encoder.fit_transform(y)
    cat_y = pd.Series(cat_y)
    return cat_y


def onehot_encoder(df):
    # One-hot encoding if there is a charictar variable in X
    for var in df:
        if is_categorical(df, var):
            enc = OneHotEncoder(handle_unknown='ignore')
            var_ohe = pd.DataFrame(enc.fit_transform(pd.DataFrame(df[var])).toarray())
            colname_ohe = enc.get_feature_names_out([var])
            var_ohe.columns = colname_ohe
            df = pd.merge(df, var_ohe, left_index=True, right_index=True, how='left')
            df = df.drop(var, axis=1)
    return df        


def update_submit_counter():
    st.session_state.submit_counter +=1


def column_selector(current_data):
    
    with st.form("run_form"):
        st.form_submit_button('Submit', on_click=update_submit_counter)
        
        if "submit_counter" not in st.session_state:
            st.session_state.submit_counter = 0

        
        st.subheader("Choose phenotype to predict, and predictors")

        st.dataframe(data=current_data.head())

        col_names = list(current_data.columns)

        col1, col2, col3 = st.columns((1, 3, 1))

        with col1:
            if current_data.shape[1] < 10: # for iris sample
                yy = st.selectbox(label='Phenotype to predict (Y)', options=col_names, index=0)
            else:
                yy = st.selectbox(label='Phenotype to predict (Y)', options=col_names, index=73) # 'HT2.x'

        with col2:
            if current_data.shape[1] < 10: # for iris sample
                xx = st.text_input(label='Predictors (X)', value=f'{col_names[1]}, {col_names[2]}:{col_names[3]}')
            else:
                xx = st.text_input(label='Predictors (X)', value=f'{col_names[74]}, {col_names[75]}, {col_names[77]}:{col_names[95]}')

        with col3:
            if is_categorical(current_data, yy):
                goal = st.radio('Goal', ['Classification'], index=0) 
            else:
                goal = st.radio('Goal', ['Prediction','Classification'], index=0)

        st.markdown(
            """
        - Pick a variable from the dataset
        - Y: select a numeric variable for prediction (e.g. `HT.x`)
        - X: use `,` for single selection, and `:` for sequence selection
            - For example, `PC1, PC2, R.x:B.x` selects PC1, PC2, and all variables between R.x and B.x 
        - Selected predictors (X)
            - Automatically applied one-hot encoding for character/categorical inputs
            - Rows with NA's are removed
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

        # if Y is text -> convert text to numeric by labelencoder
        if is_categorical(current_data, yy):
            y = labelencoder(y)

        # if Y is numeric & goal == 'Classification' -> convert it to factor

        # One-hot encoding if there is a charictar variable in X
        x = onehot_encoder(x)
        st.dataframe(x.head(1)) # Show one-hot encoded x

        # For saving results
        with open('tmp_result/y.txt', 'w') as f:
            f.write(f'"{yy}"')

        with open('tmp_result/x.txt', 'w') as f:
            for item in x.columns:
                f.write(f'"{item}";')

        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1234) ############################### test train split
        input_shape = x.shape[1] # number of variables 

        nclasses = len(set(y))

        return x_train, y_train, x_test, y_test, input_shape, goal, nclasses

# -----------------------------------------------------------------------------------------------------------------
def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print ('Error: Creating directory. ' +  directory)

# First need to save result to zip format
def zip_dir(zip_name, dir_path):
    zip_path = os.path.join(os.path.abspath(os.path.join(dir_path, os.pardir)), zip_name + '.zip')
    new_zips = zipfile.ZipFile(zip_path, 'w')
    dir_path = dir_path + '/'
 
    for root, directory, files in os.walk(dir_path):
        for file in files: # ['model.h5','predicted_values.csv', 'evaluation_result.csv']: 
            if (file == 'model.h5') or (file == 'model.pkl') or (file == 'predicted_values.csv') or (file == 'evaluation_result.csv'):
                path = os.path.join(root, file)
                new_zips.write(path, arcname=os.path.relpath(os.path.join(root, file), dir_path), compress_type=zipfile.ZIP_DEFLATED)

            else:
                next

    new_zips.close()
 
    shutil.rmtree(dir_path) # remove all files including tmp_result


def download_result():
    with open("result/myfile.zip", "rb") as fp:
        btn = st.download_button(
            label="Download model results (.zip)",
            data=fp,
            file_name=f"myfile-out.zip",
            mime="application/zip"
        )