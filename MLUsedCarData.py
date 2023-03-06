#Sean Dube
#ID 5059 Coursework 1 

#Step 1. Frame the Problem

#The task is to predict list price of US Used Cars from a subset of attributes. The data used is from the US Used Cars dataset on Kaggle, posted by Ananay Mital. This analysis will explore the structure of the data to identify a small number of attributes that could plausibly correlate well with price. Using ML methods, we will attempt to answer this question. Our decision on what method to use will based on derived models and predictions. The performance measure we will use to analyze these models is RMSE or Root Mean Square Error

#First, we load the necessary packages

import sys
import pandas
import numpy
import matplotlib.pyplot as plt
import seaborn as sn

#IMPORT DATA
cars = pandas.read_csv("/Users/seandube/Desktop/ST ANDREWS 2022-2023/ Courses Semester 2/ID5059/Coursework /used_cars_data_large_0.csv", dtype={"bed": "string", "dealer_zip": "string"})
cars.head(6)
# Clear the maximum number of columns to be displayed, so that all will be visible.
pandas.set_option('display.max_columns', None)


# STEP 1 Explore Data

#historgram of each attribute
cars.hist(bins=50, figsize=(10,10))
plt.show()

## create test set

from sklearn.model_selection import train_test_split
train_set, test_set = train_test_split(cars, test_size=0.2, random_state=314)

# check split 

len(test_set) / len (cars) # we get 0.2 so that works

cars = train_set.copy()
# explore data for real this time using train set 

## what the data looks like spatially
carsplot = cars.plot(kind = "scatter", x = "longitude", y = "latitude", figsize =(10, 7))
carsplot.set_aspect("equal")
plt.show()

#correlation matrix for all attributes
cormat = cars.corr()
sn.heatmap(cormat)
plt.show()

#check correlations with price
cormat["price"].sort_values(ascending=True) #check how attributes are correlated to price

# Mileage, Highway and City Fuel Economy, Owner Count, Horsepower, Year, and Engine Displacement appear to have the highest correlations with Price

# STEP 2 PREPARE DATA

# Separate Labels 

cars_labels = cars["price"].copy()
cars_labels.head()

cars = cars.drop(columns="price")


cars.info() # check for missing values. There are a lot, so I will only include columns for relevant variables based on the best correlations

# include relevant attributes 
 
newcars = cars.filter(items = ["mileage", "highway_fuel_economy", "owner_count", "city_fuel_economy",
  "engine_displacement", "year", "horsepower"])


## create pipeline to sort missing values in variables
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy="median")


from sklearn.pipeline import Pipeline
numerical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy="median"))
])


#create transformer 
from sklearn.compose import ColumnTransformer

numerical_attributes = ["mileage", "highway_fuel_economy", "owner_count", "city_fuel_economy",
                               "engine_displacement", "year", "horsepower"]

full_pipeline = ColumnTransformer([
    ("numerical", numerical_pipeline, numerical_attributes),
])

cars_prepared = full_pipeline.fit_transform(newcars)

#Lets take a glance

pandas.DataFrame(cars_prepared).info()


# STEP 3 Explore Models
# LINEAR REGRESSION MODEL

## fit Lease Square Linear Regression Model to the prepared training data
from sklearn.linear_model import LinearRegression

linear_regression = LinearRegression()
linear_regression.fit(cars_prepared, cars_labels)

# Use this model to generate some predictions for some test data
some_data = test_set.iloc[:5]
pandas.DataFrame(some_data)


some_labels = cars_labels.iloc[:5]
pandas.DataFrame(some_labels)

some_data_prepared = full_pipeline.transform(some_data)
pandas.DataFrame(some_data_prepared)

#make predictions
some_predictions = linear_regression.predict(some_data_prepared).round()
some_predictions
some_labels

# first test row the predicted price is about 20,655 whereas the actual value is 6,900. 

#calculate RMSE for our predictions over the big set
from sklearn.metrics import mean_squared_error

linear_regression_cars_predictions = linear_regression.predict(cars_prepared)
linear_regression_mse = mean_squared_error(cars_labels, linear_regression_cars_predictions)
linear_regression_rmse = numpy.sqrt(linear_regression_mse)
numpy.round(linear_regression_rmse)

# RMSE = 16089.0 

# This does not look very accurate: Error is high in proportion to price

numpy.round(linear_regression_rmse / cars_labels.median(), 2) #0.60 Not terrible, but lets try other models
# Lets try other models and see how they compare

# DECISION TREE MODEL

from sklearn.tree import DecisionTreeRegressor

tree_regressor = DecisionTreeRegressor(random_state=42)
tree_regressor.fit(cars_prepared, cars_labels)

tree_regressor_cars_predictions = tree_regressor.predict(cars_prepared)
tree_regressor_mse = mean_squared_error(cars_labels, tree_regressor_cars_predictions)
tree_regressor_rmse = numpy.sqrt(tree_regressor_mse)
tree_regressor_rmse

# RMSE = 4429.864...
# this is much lower compared to the linear regression RMSE

# Lets try Cross Validation

# CROSS VALIDATION

from sklearn.model_selection import cross_val_score

K = 10

tree_regressor_scores = cross_val_score(tree_regressor, cars_prepared, cars_labels,
                         scoring="neg_mean_squared_error", cv=K)
tree_regressor_rmse_scores = numpy.sqrt(-tree_regressor_scores)

# Retreive K scores

def display_scores(scores):
    print("Scores:", numpy.round(scores))
    print("Mean:", numpy.round(scores.mean()))
    print("Standard deviation:", numpy.round(scores.std()))

display_scores(tree_regressor_rmse_scores)
numpy.mean(tree_regressor_rmse_scores)

# compare to linear regression scores

linear_regression_scores = cross_val_score(linear_regression, cars_prepared, cars_labels,
                                           scoring="neg_mean_squared_error", cv=K)
linear_regression_rmse_scores = numpy.sqrt(-linear_regression_scores)
display_scores(linear_regression_rmse_scores)

numpy.mean(linear_regression_rmse_scores) # 15586.473

# Cross Validation Scores are better compared to the linear regressions scores. Cross Validation coudl be a move

# Lets try with K = 5 and see if there is any difference

from sklearn.model_selection import cross_val_score

K = 5

tree_regressor_scores = cross_val_score(tree_regressor, cars_prepared, cars_labels,
                         scoring="neg_mean_squared_error", cv=K)
tree_regressor_rmse_scores = numpy.sqrt(-tree_regressor_scores)

def display_scores(scores):
    print("Scores:", numpy.round(scores))
    print("Mean:", numpy.round(scores.mean()))
    print("Standard deviation:", numpy.round(scores.std()))

display_scores(tree_regressor_rmse_scores)

numpy.mean(tree_regressor_rmse_scores) #13441.528


# Not much improvement from the K = 10. Let's try the Random Forest Model

from sklearn.ensemble import RandomForestRegressor

forest_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
forest_regressor.fit(cars_prepared, cars_labels)

forest_regressor_cars_predictions = forest_regressor.predict(cars_prepared)
forest_regressor_mse = mean_squared_error(cars_labels, forest_regressor_cars_predictions)
forest_regressor_rmse = numpy.sqrt(forest_regressor_mse)
forest_regressor_rmse.round()

# RMSE = 6224.0, but also lets try it with Cross Validation

forest_regressor_scores = cross_val_score(forest_regressor, cars_prepared, cars_labels,
                                          scoring="neg_mean_squared_error", cv=K)
forest_regressor_rmse_scores = numpy.sqrt(-forest_regressor_scores)
display_scores(forest_regressor_rmse_scores)  #11695.0 

## Random Forest with Cross Validation appears to be the best performer so far

# FINE TUNE MODELS

# we continue with Random Forest and tune the hyperparamaters using GridSearch CV. 
from sklearn.model_selection import GridSearchCV

parameter_grid = [
    # Try 12 (3×4) combinations of hyperparameters:
    {'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]},

    # Then try 6 (2×3) combinations with bootstrap set as False:
    {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]},
]

forest_regressor = RandomForestRegressor(random_state=42)

# Train across 5 folds, giving a total of (12+6)*5=90 rounds of training.
grid_search = GridSearchCV(forest_regressor, parameter_grid, cv=5,
                           scoring='neg_mean_squared_error',
                           return_train_score=True)

grid_search.fit(cars_prepared, cars_labels)

## check best hyperparameter and RMSE from what we tested

grid_search.best_estimator_

# Best estimator was with 2 features and 30 estimators 

# now calculate RMSE for best estimator
grid_search_results = grid_search.cv_results_
for mean_score, params in zip(grid_search_results["mean_test_score"], grid_search_results["params"]):
    print(numpy.round(numpy.sqrt(-mean_score)), params)

#Set Final Model

final_model = grid_search.best_estimator_

#Run on Test Set

X_test = test_set.drop("price", axis=1)
y_test = test_set["price"].copy()

X_test_prepared = full_pipeline.transform(X_test)

final_predictions = final_model.predict(X_test_prepared)

final_mse = mean_squared_error(y_test, final_predictions)
final_rmse = numpy.sqrt(final_mse)
final_rmse.round()

## Final RMSE = 10,114.0 
## Best performing model  was Random Forest with Cross Validation with K = 5 and 30 estimators

















