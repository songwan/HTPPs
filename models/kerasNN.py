import streamlit as st
from keras.models import Sequential
from keras.layers import Dense

# number of Dense layers : model.add(Dense()\)
# number of neurons at layer 1, 2, ...: Dense(units=n)
# activation function at layer 1, 2, ...: Dense(.., activation='...') # relu, softmax, ...
# optimizer: model.compile(optimizer='sdg')
# loss: model.compile(loss='categorical_crossentropy')
# metrics: model.compile(metrics=['accuracy'])

def knn_param_selector():
    number_hidden_layers = st.number_input("Number of hidden layers", 1, 5, 1)

    hidden_layer_sizes = []
    hidden_layer_activations = []

    for i in range(number_hidden_layers):
        n_neurons = st.number_input(
            f"Number of neurons at layer {i+1}", 2, 200, 100, 25
        )

        n_activation = st.selectbox(f"Activation function at layer {i+1}", ("relu","sigmoid","softmax","softplus","softsign","tanh","selu","elu","exponential", "linear"))

        hidden_layer_sizes.append(n_neurons)
        hidden_layer_activations.append(n_activation)

    hidden_layer_sizes = tuple(hidden_layer_sizes)
    hidden_layer_activation = tuple(hidden_layer_activations)

    optimizer = st.selectbox("Optimizer", ('sgd', 'rmsprop', 'adagrad', 'adadelta', 'adam', 'adamax', 'nadam'))
    loss = st.selectbox("Loss", ('mean_squared_error', 'mean_absolute_error', 'categorical_crossentropy', 'binary_crossentropy'))
    nn_metrics = st.selectbox("Metrics", ('mae', 'acc'), key="keras metrics")
    #params = {"hidden_layer_sizes": hidden_layer_sizes, "hidden_layer_activations":hidden_layer_activations}
    #model = MLPClassifier(**params)
    #return model
    
    model = Sequential()
    model.add(Dense(units=hidden_layer_sizes[0], activation=hidden_layer_activations[0], input_shape=(5,)))
    
    if number_hidden_layers > 1: # 3
        for i in range(1, number_hidden_layers): # 0 1 2 -> 1, 2 
            model.add(Dense(units=hidden_layer_sizes[i], activation=hidden_layer_activations[i]))

    model.compile(loss=loss, optimizer=optimizer, metrics=nn_metrics)

    return model
    