from sklearn.linear_model import LinearRegression
import streamlit as st
import json

def regression_param_selector():

    fit_intercept = st.selectbox("fit_intercept", ['True', 'False'], 0)
    params = {"fit_intercept": fit_intercept}
    model = LinearRegression(**params)

    json_params = json.dumps(params)
    with open('tmp_result/params.txt', 'w') as f:
        f.write(json_params)

    return model
