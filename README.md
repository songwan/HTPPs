# Machine Learning models for HTP phenotype prediction
## About the project
---
In this project, we developed an application to easily apply machine learning and deep learning models to build a HTP phenotyping regression/classification model. 

## Demo
---
https://share.streamlit.io/songwan/htpps/develop/app.py

<p align="center">
   <img src="./images/demo-htpp.gif">
</p>

## How does it work?
---
1. Upload your own dataset (.csv) or use pre-loaded dataset (2016 IRRI)
2. Select Phenotype to predict (Y), predictors (X), and a goal (regression/classification)
3. Select machine learning model and its parameters
   - For regression, you can choose linear regression, neural network, and SVR
   - For classification, you can choose neural network, and SVC
4. Click Run button, then the app automatically displays the following results:
   - Visualization outputs 
      - For regression, scatter plot of target vs predicted valeus
      - For classification, contingency matrix for the test data
      - For neural network, the history plot
   - Performance metrics (Accracy, F1 score, R-squared, MSE)
   - The time it took the model to train
   - Buttons for downloading model (.pkl/.h5) and model information (.csv)

## Run the app locally
---
Make sure you have pip installed with Python 3.

- install pipenv

```shell
pip install pipenv
```

- go inside the folder and install the dependencies

```shell
pipenv install
pipenv shell
```

- run the app

```shell
streamlit run app.py
```

## Structure of the code
---
- `app.py` : The main script to start the app
- `utils/`
  - `ui.py`: UI functions to display the different components of the app
  - `functions.py`: for data processing, training the model and building the plotly graphs
- `models/`: where each model's hyper-parameter selector is defined
