from sklearn.linear_model import LinearRegression
import streamlit as st
import json

def update_regression_counter():
    st.session_state.reg_counter +=1


def regression_param_selector():
    
    with st.sidebar.expander("Set parameters", True): 
        st.info('Click buttons to apply')

        with st.form("reg_form"):

            st.form_submit_button('Set parameters', on_click=update_regression_counter)

            if "reg_counter" not in st.session_state:
                st.session_state.reg_counter = 0

            fit_intercept = st.selectbox("fit_intercept", ['True', 'False'], 0)
            params = {"fit_intercept": fit_intercept}
            model = LinearRegression(**params)

            json_params = json.dumps(params)
            with open('tmp_result/params.txt', 'w') as f:
                f.write(json_params)
        return model