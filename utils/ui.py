# import python libraries 
from pickle import TRUE
import pandas as pd
from pandas._libs.missing import NA
from pandas.api.types import is_numeric_dtype
import streamlit as st
import copy
import matplotlib.pyplot as plt
import os

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
    st.title("Machine Learning Models for HTP phenotype prediction")
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

    global current_data
    current_data = None

    uploaded_file = st.file_uploader("Upload CSV file", key='data_uploader')

    if uploaded_file is not None:
        
        current_data = read_csv(uploaded_file)
        dataset = "upload"
    else:
        # st.write("#### Or, choose a pre-loaded dataset")
        # dataset = st.selectbox("Choose a dataset", options=("IRRI (2016)","iris")) #########################################
        st.write("#### Or, explore with a pre-loaded dataset")
        dataset = "IRRI (2016)"

        if dataset == "IRRI (2016)":
            current_data = read_csv('data/2016DS_merged.csv')
        elif dataset == 'iris':
            current_data = read_csv('data/iris.csv')
 
    return current_data


def model_selector(goal):

    models = {'Regression':['Linear Regression', 'Keras Neural Network', 'SVR'], 'Classification':['SVC', 'Keras Neural Network']}
    model_type = st.selectbox("Models", models[goal])

    return model_type


def parameter_selector(model_type, goal, nclasses, input_shape=None):

    epochs = None
    validation_split = None
    batch_size = None

    if model_type == "Linear Regression":
        model, json_param = regression_param_selector()

    elif model_type == "Keras Neural Network":
        validation_split, epochs, batch_size, model, json_param = knn_param_selector(goal, nclasses, input_shape=input_shape)
    
    elif model_type == "SVR":
        model, json_param = svr_param_selector()

    elif model_type == 'SVC':
        model, json_param = svc_param_selector()

    return validation_split, epochs, batch_size, model, json_param


def footer():
    st.sidebar.markdown("---")
    st.sidebar.markdown(
        """
        [<img src='data:image/png;base64,{}' class='img-fluid' width=25 height=25>](https://github.com/songwan/HTPPs) <small> Based on Playground 0.1.0 </small>""".format(
            img_to_bytes("./images/github.png")
        ),
        unsafe_allow_html=True,
    )


def is_categorical(df, colname):
    # if the column is string, then assume it to be categorical
    if df[colname].dtype=='object': 
        return True

    # else assume it numerical
    else:
        return False


def labelencoder(y, yy):
    label_encoder = LabelEncoder()
    cat_y = label_encoder.fit_transform(y)
    cat_y = pd.Series(cat_y)
    labely = pd.DataFrame(label_encoder.classes_).transpose()
    return cat_y, labely


def onehot_encoder(df):
    # One-hot encoding if there is a charictar variable in X
    ohe_info = None
    ohe_info_concat = pd.DataFrame()
    for var in df:
        if is_categorical(df, var):
            enc = OneHotEncoder(handle_unknown='ignore')
            var_ohe = pd.DataFrame(enc.fit_transform(pd.DataFrame(df[var])).toarray())
            var_ohe.index = df.index
            
            colname_ohe = enc.get_feature_names_out([var])
            var_ohe.columns = colname_ohe
            ohe_info = var_ohe.drop_duplicates(ignore_index=True)
            ohe_info_concat = pd.concat([ohe_info_concat, ohe_info], axis=1)

            df = pd.merge(df, var_ohe, left_index=True, right_index=True, how='left')
            df = df.drop(var, axis=1)
            
    ohe_info = ohe_info_concat
    return df, ohe_info    


def column_display(current_data, x, ohe_info, labely, goal):
    y_enc_new = None
    y_enc_origin = None

    st.subheader('Dataset')
    df_dtype = current_data.dtypes.astype(str)    
    df_to_display = pd.DataFrame(current_data.head().to_numpy(), columns = [current_data.columns, df_dtype]) # df with dtypes 
    st.dataframe(df_to_display)
    
    if labely is not None:
        st.write("- Y encoding")        
        st.dataframe(labely)
        
        y_enc_origin = labely.columns.tolist()

        if goal == 'Regression':
            st.warning('If you want regression on character Y, please make sure that the numerical encoding is as intended.')
            with st.expander("Expand to set your own Y encoding (optional)", expanded=False):
                st.write("- This feature is useful when you are using ordinal Y")
                labely_n = labely.shape[1]
                labely_usr = eval(st.text_input("Enter your own encoding", "6,2,4,9,11,...,10"))
                labely_usr_n = len(labely_usr)

                if (Ellipsis not in labely_usr):
                    if labely_usr_n==labely_n:
                        labely.columns = labely_usr
                        st.write(labely)
                        y_enc_new = labely.columns.tolist()
                    else:
                        st.error(f'The number of labels to encode is {labely_n} but have {labely_usr_n} encodings')
                # else: 
                    # st.info(f'Please provide your own encoding to apply.')

    col_names = list(current_data.columns)

    if ohe_info.shape[0]!=0:
        with st.expander('Expand to see the encoding details for categorical X (optional)', False):
            st.dataframe(ohe_info)

    st.markdown(
        """
    - Selected predictors (X)
        - Rows with NA's are removed
    """
    )
    st.dataframe(x.head()) # Show one-hot encoded x

    st.markdown("---")

    return y_enc_origin, y_enc_new, labely


def column_selector(current_data):

    df_dtype = current_data.dtypes.astype(str)    
    col_names = list(current_data.columns)

    if current_data.shape[1] < 10: # for iris sample
        yy = st.selectbox(label='Phenotype to predict (Y)', options=col_names, index=0)
    else:
        yy = st.selectbox(label='Phenotype to predict (Y)', options=col_names, index=73) # 'HT2.x'
        
    if current_data.shape[1] < 10: # for iris sample
        xx = st.text_input(label='Predictors (X)', value=f'{col_names[1]}, {col_names[2]}:{col_names[3]}')
    else:
        xx = st.text_input(label='Predictors (X)', value=f'{col_names[74]}, {col_names[75]}, {col_names[77]}:{col_names[95]}')
    
    st.info(
    """
    - Use ',' for single selection
    - Use ':' for sequence selection
    - Note that including a variable used as Y in X will produce an error
    """
    )

    if is_categorical(current_data, yy):
        goal = st.radio('Goal', ['Regression','Classification'], index=0)
    else:
        goal = st.radio('Goal', ['Regression','Classification'], index=0)

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

    labely = None

    # if (Y is numeric) & (goal is Classification) -> convert Y to categorical -> apply multilabel
    if (is_numeric_dtype(y)) and (goal=='Classification'):
        y = y.astype(object)
        y, labely = labelencoder(y, yy)

    # if Y is text -> convert text to numeric by labelencoder
    if is_categorical(current_data, yy):
        y, labely = labelencoder(y, yy)

    # if Y is numeric & goal == 'Classification' -> convert it to factor
    # One-hot encoding if there is a charictar variable in X
    x, ohe_info = onehot_encoder(x)

    # # For saving results
    var_names = {'y':yy, 'x':x.columns}

    train_test_ratio = st.number_input("Test data ratio", 0.1, 1.0, 0.2, 0.1)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=train_test_ratio, random_state=1234) ############################### test train split
    input_shape = x.shape[1] # number of variables 

    nclasses = len(set(y))
    return x_train, y_train, x_test, y_test, input_shape, goal, nclasses, labely, var_names, ohe_info, x, train_test_ratio


def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print ('Error: Creating directory. ' +  directory)
