from grpc import ssl_server_certificate_configuration
from sklearn.metrics import classification_report
import streamlit as st
import tensorflow as tf
import json
from keras.models import Sequential
from keras.layers import Dense

def layers_layout(goal, nclasses, number_layers):

    last_n_neuron = {'Regression':1, 'Classification':nclasses}

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
                    n_neurons = st.number_input(f"input layer", 1, 1000000, 100, 25) # first layer

                elif count==number_layers:
                    n_neurons = st.number_input(f"output layer (automatic)", last_n_neuron[goal], last_n_neuron[goal], last_n_neuron[goal], 0) # last layer has only 1 neuron by default
                else:
                   n_neurons = st.number_input(f"layer {count}", 1, 1000000, 100, 25) # middle layers

                layer_sizes.append(n_neurons)

                if count == number_layers: 
                    return layer_sizes

def activation_layout(goal, nclasses, number_layers):
    
    if (nclasses==2) and (goal=='Classification'): 
        goal = 'Binary'
    
    last_activation = {'Regression':[None, 'relu', 'linear'], 'Classification':['softmax'], 'Binary': ['sigmoid']}#:{'binary':['sigmoid'], 'multiple':['softmax']}}
    # https://towardsdatascience.com/deep-learning-which-loss-and-activation-functions-should-i-use-ac02f1c56aa8
    # softmax: multi-class classification
    # sigmoid: binary classification

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
                    n_activation = st.selectbox(f"output layer", last_activation[goal], index=0) # last layer activation = None

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

def knn_param_selector(goal, nclasses, input_shape=None):
    
    if (nclasses==2) and (goal=='Classification'): 
        goal = 'Binary'

    loss_by_goal = {'Regression':['mean_squared_error', 'mean_absolute_error'], 'Classification':['sparse_categorical_crossentropy', 'categorical_crossentropy'], 'Binary':['binary_crossentropy']}
    
    layer_sizes = []
    layer_activations = []
    layer_kernelinits = []
    number_layers = st.number_input("Number of layers", 2, 20, 2)
 
    layer_sizes = layers_layout(goal, nclasses, number_layers)
    layer_activations = activation_layout(goal, nclasses, number_layers)
    layer_kernelinits = kernel_init_layout(number_layers)

    optimizer_selector = st.selectbox("Optimizer", ('SGD','RMSprop','Adagrad','Adadelta','Adam','Adamax','Nadam'), index=4)
    lr_default = {'SGD':0.01, 'RMSprop':0.001, 'Adagrad':0.001, 'Adadelta':0.001, 'Adam':0.001, 'Adamax':0.001, 'Nadam':0.001}
    lr = st.number_input(f'Learning rate ({optimizer_selector} default: {lr_default[optimizer_selector]})', 0.0, 1.0, 0.01)
    optimizer = eval(f'tf.keras.optimizers.{optimizer_selector}(learning_rate={lr})')
    loss = st.selectbox("Loss", loss_by_goal[goal])
    nn_metrics = st.selectbox("Metrics", ('mae', 'acc'), key="keras metrics")
    epochs = st.number_input('Epochs', 1, 1000, 10, 10)
    validation_split = st.number_input('Validation split ratio', 0.0, 1.0, 0.2, 0.1)

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
    # with open('tmp_result/params.txt', 'w') as f:
    #     f.write(json_params)

    return validation_split, epochs, model, json_params
