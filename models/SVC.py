import streamlit as st
import json
from sklearn.svm import SVC

def svc_param_selector():

    C = st.number_input("C", 0.01, 2.0, 1.0, 0.01)
    kernel = st.selectbox("kernel", ("rbf", "linear", "poly", "sigmoid")) 
    degree = st.number_input('degree (for polynomial kernel)', 1, 10, 3, 1) # default degree = 3
    if kernel == 'poly':
        params = {"C": C, "kernel": kernel, 'degree': degree}
    else:
        params = {"C": C, "kernel": kernel}

    model = SVC(**params)
    json_params = json.dumps(params)
    # with open('tmp_result/params.txt', 'w') as f:
    #     f.write(json_params)

    return model, json_params