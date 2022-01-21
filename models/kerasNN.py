from grpc import ssl_server_certificate_configuration
import streamlit as st
from keras.models import Sequential
from keras.layers import Dense
from utils.functions import set_sidebar_width

def layers_layout(number_layers, input_shape):
    layer_sizes = []
    nrows = (number_layers-1)//5 + 1 
    ncols = 5
    count = 0

    st.write('Set the number of neurons')

    for row in range(nrows):
        cols = st.columns(ncols)
        for i in range(ncols):
            with cols[i]:
                count +=1

                if count==1:
                    n_neurons = st.number_input(f"input layer", 1, 200, 100, 25)

                elif count==number_layers:
                    n_neurons = st.number_input(f"output layer", 1, 200, 1, 10)

                else:
                   n_neurons = st.number_input(f"layer {count}", 1, 200, 100, 25)

                layer_sizes.append(n_neurons)

                if count == number_layers: 
                    return layer_sizes

def activation_layout(number_layers):
    activation_list = ("relu","sigmoid","softmax","softplus","softsign","tanh","selu","elu","exponential", "linear")
    layer_activations = []
    nrows = (number_layers-1)//5 + 1 
    ncols = 5
    count = 0

    st.write('Set the activation function')

    for row in range(nrows):
        cols = st.columns(ncols)
        for i in range(ncols):
            with cols[i]:
                count +=1

                if count==1:
                    n_activation = st.selectbox(f"input layer", activation_list)

                elif count==number_layers:
                    n_activation = st.selectbox(f"output layer", activation_list)

                else:
                    n_activation = st.selectbox(f"layer {count}", activation_list)

                # n_activation = st.selectbox(f"layer {count}", activation_list)

                layer_activations.append(n_activation)
                if count == number_layers: 
                    return layer_activations

def update_expand_counter():
    st.session_state.expand_clicked = not(st.session_state.expand_clicked)

def update_nn_counter():
    st.session_state.counter +=1

def update_compiler_counter():
    st.session_state.compiler_counter +=1


def knn_param_selector(input_shape=None):

    layer_sizes = []
    layer_activations = []

    with st.sidebar.expander("Set parameters", True): 
        st.info('Click buttons to apply')

        with st.form("layer_form"):
    
            if 'expand_clicked' not in st.session_state:
                st.session_state.expand_clicked = False

            if "counter" not in st.session_state:
                st.session_state.counter = 0

            st.form_submit_button('Expand Sidebar', on_click=update_expand_counter)

            number_layers = st.number_input("Number of layers", 2, 20, 2)
            st.form_submit_button("Set layer parameters", on_click=update_nn_counter)

            if st.session_state.counter>0:

                layer_sizes = layers_layout(number_layers, input_shape=input_shape)
                layer_activations = activation_layout(number_layers)

        with st.form("compiler_layer_form"):
            if "compiler_counter" not in st.session_state:
                st.session_state.compiler_counter = 0

            st.form_submit_button("Set compiler parameters", on_click=update_compiler_counter)
            optimizer = st.selectbox("Optimizer", ('sgd', 'rmsprop', 'adagrad', 'adadelta', 'adam', 'adamax', 'nadam'))
            loss = st.selectbox("Loss", ('mean_squared_error', 'mean_absolute_error', 'categorical_crossentropy', 'binary_crossentropy'))
            nn_metrics = st.selectbox("Metrics", ('mae', 'acc'), key="keras metrics")

    if st.session_state.expand_clicked:
        set_sidebar_width(60)
    else:
        set_sidebar_width(21)

    layer_sizes = tuple(layer_sizes)
    layer_activations = tuple(layer_activations)

    #### Build Keras model
    model = Sequential()
    
    if(len(layer_sizes)==0): # if parameters were not set (default setting)
        number_layers = 1
        model.add(Dense(units=5, activation='relu', input_shape=(input_shape,)))
        
    else: # if parameters were set from the sidebar
        model.add(Dense(units=layer_sizes[0], activation=layer_activations[0], input_shape=(input_shape,))) # input layer

        if number_layers > 1:
            for i in range(1, (number_layers-1)):
                model.add(Dense(units=layer_sizes[i], activation=layer_activations[i])) # hidden layers

        model.add(Dense(units=layer_sizes[-1], activation=layer_activations[-1])) # output layer
        # model.add(Dense(units=1)) # output layer

    model.compile(loss=loss, optimizer=optimizer, metrics=nn_metrics)

    return model
