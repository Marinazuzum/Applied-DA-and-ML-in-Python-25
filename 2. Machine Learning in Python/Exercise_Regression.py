#Exercise: Regression
#Applied Data Analytics and Machine Learning in Python
#In this exercise, you will use the python library sklearn to implement regression using different
#algorithms and visualize the results. In addition, you need to use pipeline to assemble several
#steps of the regression workflow i.e. Preprocessor and Regressor.
#Further instructions for this task are given in Exercise_Regression_Template.py.
#After completing the task, you can try change the components in the pipeline to experience
#the impact of different processor combinations.

#Preprocessor: Transforms the input data (e.g., scaling, normalization). In my code, it's a MinMaxScaler.
#Regressor: The machine learning model that makes predictions. In my code, it varies (e.g., LinearRegression, SVR, RandomForestRegressor, KNeighborsRegressor).

from sklearn import datasets, linear_model, svm, neighbors, ensemble
from sklearn.metrics import mean_squared_error
import sklearn.pipeline
import matplotlib.pyplot as plt
import numpy as np
import joblib #Joblib is a set of tools to provide lightweight pipelining in Python.
#Загружать модели позже для предсказаний без необходимости повторного обучения.
import os
import time 
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV

N_SAMPLES = 1000
TEST_FRACTION = .2  #20% of the dataset will be used for testing, 
#and the remaining 80% will be used for training
OUTPUTDIRECTORY = os.getcwd() + '/' #Specifies the directory where output files (e.g., trained models) will be saved.

PARAMETERS_GRIDSEARCHCV = {'cv': 5, #Specifies the number of folds for cross-validation
                           'verbose': 1 #Controls the verbosity of the output during GridSearchCV

                           }


def create_dataset(samples=N_SAMPLES, test_fraction=TEST_FRACTION):
    # Input:
    #   - samples: int, which specifies the number of samples in the dataset
    #   - test_fraction: float, specifies the fraction of datapoints which are used for testing
    # Return:
    #   - x_train: array, X values for training (dim: (X, 1))
    #   - x_test: array, X values for testing (dim: (X, 1))
    
    #   - y_train: array, Y values for training (dim: (X, ))
    #   - y_test: array, Y values for testing (dim: (X, ))
    
    #   - x_predict: array, X values for prediction (dim: (X, 1))
    # Function:
    #   - use datasets.make_regression() to generate a dataset with samples of datapoints; 
    #       Use the following parameters:
    #       - n_features=1
    #       - n_targets=1
    #       - noise=5
    #   - split the dataset in two parts according to test_fraction

    #   - generate X values for prediction; they should be in the same interval as the X values of the generated dataset
    #     Generate as many points for prediction as there are samples in the dataset

    ### Enter your code here ###
    # Generate dataset function
    x, y = datasets.make_regression(n_samples=samples, n_features=1, n_targets=1, noise=5)
    
    # Split dataset
    split_idx = int(samples * (1 - test_fraction)) #20% of the dataset will be used for testing
    x_train, x_test = x[:split_idx], x[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    # Generate prediction points
    x_predict = np.linspace(x.min(), x.max(), samples).reshape(-1, 1) #Why Reshape to (-1, 1)?
#Most machine learning models in scikit-learn expect the input features (X) to be a 2D array, 
# even if there is only one feature.
#The shape of x_predict must be (n_samples, n_features).
# In this case, n_features = 1, so reshaping ensures the array has the correct shape.

    ### End of your code ###

    return x_train, x_test, y_train, y_test, x_predict


def train_lin_reg(dataset_in, filename=(OUTPUTDIRECTORY + 'lin_reg')):
    # Input:
    #   - dataset_in: list containing x_train, x_test, y_train, y_test, x_predict
    #   - filename: string
    # Return:
    #   - none
    # Function:
    #   - create 2 dicts with hyperparameters for data transformation and the regression model (already done)
    #     ('X__Y' references to a specific parameter Y of X)
    #     (arrays or lists create a grid of possible combinations which afterwards are tested;
    #      here: 2 different feature ranges)
    #   - combine the 2 dicts to a new parameter grid called parameter_dict (already done)
    #   - implement a MinMaxScaler in preprocessor using sklearn.preprocessing
    #   - create a new LinearRegression() model
    #   - define the steps of the pipeline (already done)
    #   - create the pipeline with sklearn.pipeline.Pipeline()
    #   - optimize the model using sklearn.model_selection.GridSearchCV()
    #       - Arguments: pipe, param_grid, **PARAMETERS_GRIDSEARCHCV
    #       - GridSearchCV: GridSearch on param_grid using cv-fold Cross Validation
    #         (cv is specified in PARAMETERS_GRIDSEARCH)
    #   - fit the optimized model with the train dataset
    #   - save the fitted model in filename.joblib using the joblib library
    x_train, x_test, y_train, y_test, x_predict = dataset_in
    hyperparameters_data_transformation = {'transform_data__feature_range': [(-1, 1), (0, 1)],
                                           'transform_data__copy': [True],
                                           }

    hyperparameters_model = {}

    parameter_dict = {}
    parameter_dict.update(hyperparameters_data_transformation)
    parameter_dict.update(hyperparameters_model)

    preprocessor = MinMaxScaler()
    model = linear_model.LinearRegression()

    steps = [('transform_data', preprocessor), ('built_model', model)]
    pipe = sklearn.pipeline.Pipeline(steps)

    ### Enter your code here ###

    optimized_model = GridSearchCV(pipe, parameter_dict, **PARAMETERS_GRIDSEARCHCV)
    optimized_model.fit(x_train, y_train)

    joblib.dump(optimized_model, filename + '.joblib')

    ### End of your code ###

    return


def train_svr(dataset_in, filename=(OUTPUTDIRECTORY + 'SVR')):
    # Input:
    #   - dataset_in: list containing x_train, x_test, y_train, y_test, x_predict
    #   - filename: string
    # Return:
    #   - none
    # Function:
    #   - create 2 dicts with hyperparameters for data transformation and the regression model
    #     use the same dict for data transformation as in train_lin_reg()
    #     model parameters:
    #       - vary epsilon and C: use logspace from -5 to 5 to base 2 creating 10 samples
    #       - set tol to 1e-3
    #   - combine the 2 dicts to a new parameter grid called parameter_dict
    #   - implement a MinMaxScaler in preprocessor using sklearn.preprocessing
    #   - create a new SVR() model
    #   - define the steps of the pipeline
    #   - create the pipeline with sklearn.pipeline.Pipeline()
    #   - optimize the model using sklearn.model_selection.GridSearchCV()
    #       - Arguments: pipe, param_grid, **PARAMETERS_GRIDSEARCHCV
    #       - GridSearchCV: GridSearch on param_grid using cv-fold Cross Validation
    #         (cv is specified in PARAMETERS_GRIDSEARCH)
    #   - fit the optimized model with the train dataset
    #   - save the fitted model in filename.joblib using the joblib library

    ### Enter your code here ###
    x_train, x_test, y_train, y_test, x_predict = dataset_in

    hyperparameters_data_transformation = {'transform_data__feature_range': [(-1, 1), (0, 1)],
                                           'transform_data__copy': [True],
                                           }

    hyperparameters_model = {'built_model__epsilon': np.logspace(-5, 5, 10, base=2),
                             'built_model__C': np.logspace(-5, 5, 10, base=2),
                             'built_model__tol': [1e-3]
                             }

    parameter_dict = {}
    parameter_dict.update(hyperparameters_data_transformation)
    parameter_dict.update(hyperparameters_model)

    preprocessor = MinMaxScaler()
    model = svm.SVR()

    steps = [('transform_data', preprocessor), ('built_model', model)]
    pipe = sklearn.pipeline.Pipeline(steps)

    optimized_model = GridSearchCV(pipe, parameter_dict, **PARAMETERS_GRIDSEARCHCV)
    optimized_model.fit(x_train, y_train)
    
    joblib.dump(optimized_model, filename + '.joblib')

    ### End of your code ###

    return


def train_random_forest(dataset_in, filename=(OUTPUTDIRECTORY + 'Rand_for')):
    # Input:
    #   - dataset_in: list containing x_train, x_test, y_train, y_test, x_predict
    #   - filename: string
    # Return:
    #   - none
    # Function:
    #   - create 2 dicts with hyperparameters for data transformation and the regression model
    #     use the same dict for data transformation as in train_lin_reg()
    #     model parameters:
    #       - vary max_features between 'sqrt' and 'log2'
    #       - set n_estimators to 5000
    #   - combine the 2 dicts to a new parameter grid called parameter_dict
    #   - implement a MinMaxScaler in preprocessor using sklearn.preprocessing
    #   - create a new RandomForestRegressor() model
    #   - define the steps of the pipeline
    #   - create the pipeline with sklearn.pipeline.Pipeline()
    #   - optimize the model using sklearn.model_selection.GridSearchCV()
    #       - Arguments: pipe, param_grid, **PARAMETERS_GRIDSEARCHCV
    #       - GridSearchCV: GridSearch on param_grid using cv-fold Cross Validation
    #         (cv is specified in PARAMETERS_GRIDSEARCH)
    #   - fit the optimized model with the train dataset
    #   - save the fitted model in filename.joblib using the joblib library

    ### Enter your code here ###
    x_train, x_test, y_train, y_test, x_predict = dataset_in

    hyperparameters_data_transformation = {'transform_data__feature_range': [(-1, 1), (0, 1)],
                                           'transform_data__copy': [True],
                                           }

    hyperparameters_model = {'built_model__max_features': ['sqrt', 'log2'],
                             'built_model__n_estimators': [5000]
                             }

    parameter_dict = {}
    parameter_dict.update(hyperparameters_data_transformation)
    parameter_dict.update(hyperparameters_model)

    preprocessor = MinMaxScaler()
    model = ensemble.RandomForestRegressor()

    steps = [('transform_data', preprocessor), ('built_model', model)]
    pipe = sklearn.pipeline.Pipeline(steps)

    optimized_model = GridSearchCV(pipe, parameter_dict, **PARAMETERS_GRIDSEARCHCV)
    optimized_model.fit(x_train, y_train)

    joblib.dump(optimized_model, filename + '.joblib')

    ### End of your code ###

    return

def train_knn(dataset_in, filename=(OUTPUTDIRECTORY + 'KNN')):
    # Input:
    #   - dataset_in: list containing x_train, x_test, y_train, y_test, x_predict
    #   - filename: string
    # Return:
    #   - none
    # Function:
    #   - create 2 dicts with hyperparameters for data transformation and the regression model (already done)
    #     use the same dict for data transformation as in train_lin_reg()
    #     model parameters:
    #       - have a look at the documentation of KNeighborsRegression() and select 2 parameters to vary
    #   - combine the 2 dicts to a new parameter grid called parameter_dict (already done)
    #   - implement a MinMaxScaler in preprocessor using sklearn.preprocessing
    #   - create a new KNeighborsRegression() model
    #   - define the steps of the pipeline
    #   - create the pipeline with sklearn.pipeline.Pipeline()
    #   - optimize the model using sklearn.model_selection.GridSearchCV()
    #       - Arguments: pipe, param_grid, **PARAMETERS_GRIDSEARCHCV
    #       - GridSearchCV: GridSearch on param_grid using cv-fold Cross Validation
    #         (cv is specified in PARAMETERS_GRIDSEARCH)
    #   - fit the optimized model with the train dataset
    #   - save the fitted model in filename.joblib using the joblib library

    ### Enter your code here ###
    x_train, x_test, y_train, y_test, x_predict = dataset_in

    hyperparameters_data_transformation = {'transform_data__feature_range': [(-1, 1), (0, 1)],
                                           'transform_data__copy': [True],
                                           }

    hyperparameters_model = {'built_model__n_neighbors': [3, 5, 7],
                             'built_model__weights': ['uniform', 'distance']
                             }

    parameter_dict = {}
    parameter_dict.update(hyperparameters_data_transformation)
    parameter_dict.update(hyperparameters_model)

    preprocessor = MinMaxScaler()
    model = neighbors.KNeighborsRegressor()

    steps = [('transform_data', preprocessor), ('built_model', model)]
    pipe = sklearn.pipeline.Pipeline(steps)

    optimized_model = GridSearchCV(pipe, parameter_dict, **PARAMETERS_GRIDSEARCHCV)
    optimized_model.fit(x_train, y_train)

    joblib.dump(optimized_model, filename + '.joblib')

    ### End of your code ###

    return

def model_predict(dataset_in, filename_extension):
    # Input
    #   - dataset_in: list containing x_train, x_test, y_train, y_test, x_predict
    #   - filename_extension: string, specifing which file to open
    # Return:
    #   - mse: float, mean squared error (MSE)
    #   - y: array, predicted y values based on x_predict
    # Function:
    #   - load model from file
    #   - compute and return mse evaluating the prediction for x_test and y_test
    #   - compute and return the prediction for x_predict

    ### Enter your code here ###
    x_train, x_test, y_train, y_test, x_predict = dataset_in

    model = joblib.load(OUTPUTDIRECTORY + filename_extension + '.joblib')
    y_pred = model.predict(x_test)
    mse = mean_squared_error(y_test, y_pred)

    y = model.predict(x_predict)

    ### End of your code ###
    
    return mse, y


if __name__ == "__main__":
    # Create dataset
    dataset = create_dataset()

    # plot dataset
    plt.plot(dataset[0], dataset[2], '.', color='blue', ms=2)
    plt.plot(dataset[1], dataset[3], '.', color='blue', ms=2, label='Database')

    # train different models
    train_lin_reg(dataset)
    train_svr(dataset)
    train_random_forest(dataset)
    train_knn(dataset)

    models = [('lin_reg', 'Linear Regression'), ('SVR', 'SVR'), ('Rand_for', 'Random Forest'), ('KNN', 'KNN')]
    results = []

    print('MSE for different models:')
    for model_sel in models:
        results.append(model_predict(dataset, model_sel[0]))
        plt.plot(dataset[4], results[-1][1], label=model_sel[1])
        print('MSE for {0} = {1:.2f}'.format(model_sel[1], results[-1][0]))

    plt.legend(loc='upper left')
    plt.show()

    #Conclusion:
    #Cross-validation
    # Fitting 5 folds for each of 2 candidates, totalling 10 fits (GridSearchCV GridSearchCV automates the process of finding the best hyperparameters, 
    # which saves time and reduces the likelihood of errors.) 
    # Fitting 5 folds for each of 200 candidates, totalling 1000 fits
    # Fitting 5 folds for each of 4 candidates, totalling 20 fits
    # Fitting 5 folds for each of 12 candidates, totalling 60 fits
    
    # MSE for different models:

    # MSE for Linear Regression = 23.58
    # MSE for SRV = 36.24
    # MSE for Random Forest = 37.24
    # MSE for KNN = 33.65
    
    # SVR is the best-performing model for this dataset, but Linear Regression is also a strong contender.
    # Random Forest and KNN underperform and may require further tuning or a different approach.
    # Focus on improving SVR and Linear Regression, and consider whether the dataset size or complexity is limiting the performance of more complex models.
