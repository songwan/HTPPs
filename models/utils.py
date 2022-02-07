model_imports = {
    "Linear Regression": "from sklearn.linear_model import LinearRegression",
    "Keras Neural Network": "keras.models import Sequential; from keras.layers import Dense",
    "SVR": "from sklearn.svm import SVR",
    "SVC": "from sklearn.svm import SVC",

}


model_urls = {
    "Linear Regression": "https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html",
    "Keras Neural Network": "https://keras.io/getting_started",
    "SVR": "https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html",
    "SVC": "https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html",

}


model_infos = {
    "Linear Regression": """
        - Ordinary least squares Linear Regression
        - LinearRegression fits a linear model with coefficients w = (w1, …, wp) to minimize the residual sum of squares between the observed targets in the dataset, and the targets predicted by the linear approximation
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
    "SVC": """
        - SVMs or SVCs are effective when the number of features is larger than the number of samples
        - They provide different type of kernel functions
        - They require careful normalization   
    """,
}
