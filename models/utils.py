model_imports = {
    "Neural Network": "from sklearn.neural_network import MLPClassifier",
    "Keras Neural Network": "keras.models import Sequential; from keras.layers import Dense",
    "SVR": "from sklearn.svm import SVR",
}


model_urls = {
    "Neural Network": "https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html",
    "Keras Neural Network": "https://keras.io/getting_started",
    "SVR": "https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html",
}


model_infos = {
    "Neural Network": """
        - Neural Networks have great representational power but overfit on small datasets if not properly regularized
        - They have many parameters that require tweaking
        - They are computationally intensive on large datasets
    """,
        "Keras Neural Network": """
        - Neural Networks have great representational power but overfit on small datasets if not properly regularized
        - They have many parameters that require tweaking
        - They are computationally intensive on large datasets
    """,
    "SVR": """
       - SVMs or SVRs are effective when the number of features is larger than the number of samples
       - They provide different type of kernel functions
       - They require careful normalization   
   """,
}
