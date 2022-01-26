# Initialization at every layer? https://datascience.stackexchange.com/questions/45500/neural-network-initialization-every-layer
# What is kernel initializer? https://datascience.stackexchange.com/questions/37378/what-are-kernel-initializers-and-what-is-their-significance

from grpc import ssl_server_certificate_configuration
import streamlit as st
import tensorflow as tf
import json
from keras.models import Sequential
from keras.layers import Dense
# from keras import initializers
#from keras import optimizers
from utils.functions import set_sidebar_width

def layers_layout(number_layers):
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
                    n_neurons = st.number_input(f"input layer", 1, 200, 100, 25) # first layer

                elif count==number_layers:
                    n_neurons = st.number_input(f"output layer", 1, 200, 1, 10) # last layer has only 1 neuron by default

                else:
                   n_neurons = st.number_input(f"layer {count}", 1, 200, 100, 25) # middle layers

                layer_sizes.append(n_neurons)

                if count == number_layers: 
                    return layer_sizes

def activation_layout(number_layers):
    activation_list = (None, "relu","sigmoid","softmax","softplus","softsign","tanh","selu","elu","exponential", "linear")
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
                    n_activation = st.selectbox(f"input layer", activation_list, index=1) # first layer

                elif count==number_layers:
                    n_activation = st.selectbox(f"output layer", activation_list, index=0) # last layer activation = None

                else:
                    n_activation = st.selectbox(f"layer {count}", activation_list, index=1) # middle layers

                # n_activation = st.selectbox(f"layer {count}", activation_list)

                layer_activations.append(n_activation)
                if count == number_layers: 
                    return layer_activations

def kernel_init_layout(number_layers):
    kernel_init_list = ('VarianceScaling', 'TruncatedNormal','RandomUniform', 'RandomNormal', 'glorot_uniform', 'glorot_normal', 
    'he_uniform', 'he_normal', 'lecun_normal', 'lecun_uniform', 'Identity', 'Orthogonal', 'Constant', 'Ones', 'Zeros')
    layer_kernelinits = []
    nrows = (number_layers-1)//5 + 1 
    ncols = 5
    count = 0

    st.write('Set the kernel initializer (glorot_uniform default)')

    for row in range(nrows):
        cols = st.columns(ncols)
        for i in range(ncols):
            with cols[i]:
                count +=1

                if count==1:
                    n_activation = st.selectbox(f"input layer", kernel_init_list, 4) # default: glorot_uniform

                elif count==number_layers:
                    n_activation = st.selectbox(f"output layer", kernel_init_list, 4)

                else:
                    n_activation = st.selectbox(f"layer {count}", kernel_init_list, 4)

                layer_kernelinits.append(n_activation)
                if count == number_layers: 
                    return layer_kernelinits

def update_expand_counter():
    st.session_state.expand_clicked = not(st.session_state.expand_clicked)

def update_nn_counter():
    st.session_state.counter +=1

def update_compiler_counter():
    st.session_state.compiler_counter +=1


def knn_param_selector(input_shape=None):

    layer_sizes = []
    layer_activations = []
    layer_kernelinits = []

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

                layer_sizes = layers_layout(number_layers)
                layer_activations = activation_layout(number_layers)
                layer_kernelinits = kernel_init_layout(number_layers)

        with st.form("compiler_layer_form"):
            if "compiler_counter" not in st.session_state:
                st.session_state.compiler_counter = 0

            st.form_submit_button("Set other parameters", on_click=update_compiler_counter)
            optimizer_selector = st.selectbox("Optimizer", ('SGD','RMSprop','Adagrad','Adadelta','Adam','Adamax','Nadam'))
            lr_default = {'SGD':0.01, 'RMSprop':0.001, 'Adagrad':0.001, 'Adadelta':0.001, 'Adam':0.001, 'Adamax':0.001, 'Nadam':0.001}
            lr = st.number_input(f'Learning rate ({optimizer_selector} default: {lr_default[optimizer_selector]})', 0.0, 1.0, 0.01)
            optimizer = eval(f'tf.keras.optimizers.{optimizer_selector}(learning_rate={lr})')
            loss = st.selectbox("Loss", ('mean_squared_error', 'mean_absolute_error', 'categorical_crossentropy', 'binary_crossentropy'))
            nn_metrics = st.selectbox("Metrics", ('mae', 'acc'), key="keras metrics")
            epochs = st.number_input('Epochs', 1, 100, 10, 10)
            validation_split = st.number_input('Validation split ratio', 0.0, 1.0, 0.2, 0.1)

    if st.session_state.expand_clicked:
        set_sidebar_width(60)
    else:
        set_sidebar_width(21)


    layer_sizes = tuple(layer_sizes)
    layer_activations = tuple(layer_activations)
    layer_kernelinits = tuple(layer_kernelinits)
    
    #### Build Keras model
    model = Sequential()
    
    if(len(layer_sizes)==0): # if parameters were not set (default setting) #################################################################### Change later
        number_layers = 2
        model.add(Dense(units=5, activation='relu', input_shape=(input_shape,)))
        model.add(Dense(units=1))
        
    else: # if parameters were set from the sidebar 
        model.add(Dense(units=layer_sizes[0], activation=layer_activations[0], input_shape=(input_shape,), kernel_initializer=layer_kernelinits[0])) # input layer

        if number_layers > 1:
            for i in range(1, (number_layers-1)):
                model.add(Dense(units=layer_sizes[i], activation=layer_activations[i], kernel_initializer=layer_kernelinits[i])) # hidden layers

        model.add(Dense(units=layer_sizes[-1], activation=layer_activations[-1], kernel_initializer=layer_kernelinits[-1])) # output layer

    model.compile(loss=loss, optimizer=optimizer, metrics=nn_metrics)

    params = {'n_layers': number_layers, 'units':layer_sizes, 'activation': layer_activations, 
        'kernel_initializer':layer_kernelinits, 'optimizer':optimizer_selector, 'learning_rate':lr, 'loss':loss, 'metrics':nn_metrics, 'epochs':epochs, 'validation_split':validation_split}
    
    json_params = json.dumps(params)
    with open('tmp_result/params.txt', 'w') as f:
        f.write(json_params)

    return validation_split, epochs, model
