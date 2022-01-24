import streamlit as st
from sklearn.svm import SVR

def update_svr_counter():
    st.session_state.svr_counter +=1


def svr_param_selector():
    
    with st.sidebar.expander("Set parameters", True): 
        st.info('Click buttons to apply')

        with st.form("svr_form"):

            st.form_submit_button('Set parameters', on_click=update_svr_counter)

            if "svr_counter" not in st.session_state:
                st.session_state.svr_counter = 0

            C = st.number_input("C", 0.01, 2.0, 1.0, 0.01)
            kernel = st.selectbox("kernel", ("rbf", "linear", "poly", "sigmoid")) 
            params = {"C": C, "kernel": kernel}
            model = SVR(**params)


        return model