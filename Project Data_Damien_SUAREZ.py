# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 09:36:01 2023

@author: damie
"""

##Import the necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
from sklearn.externals import joblib

def flowers():
    #First of all, we want to import the database
    db = pd.read_csv("/Users/damie/Downloads/IRIS_ Flower_Dataset.csv")
    
    x = db.drop("species",axis=1) #In x, xe keep everything except the information we want
    y = db["species"] #Here in y, we save the information about the specie of the flower

    #We then divide both our x and y in a part that will to train a model and then test itself on the rest of the data
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    #We create 200 trees of decision that will return a type of flower for a precise input
    # Define the RandomForestClassifier
    rf = RandomForestClassifier(n_estimators=200, random_state=42)

    # Fit the model
    rf.fit(x_train, y_train)

    # Get feature importances
    feature_importance = pd.DataFrame({'Feature': x_train.columns, 'Importance': rf.feature_importances_})
    feature_importance = feature_importance.sort_values(by='Importance', ascending=False)
    print(feature_importance)
    
    # Define the hyperparameters and their ranges to search
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 5, 10, 15],
        'min_samples_split': [2, 5, 10]
    }

    # Initialize GridSearchCV
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    
    grid_search.fit(x_train, y_train)
    
    best_model = grid_search.best_estimator_
    joblib.dump(best_model, 'random_forest_model.pkl')

    y_pred = rf.predict(x_test)#We test it in order to determinate the accuracy afterwards

    #We calculate the accuracy and return it to evaluate our model
    accuracy = accuracy_score(y_test, y_pred)
    print(f"The precision of the model is : {accuracy}") 
    
    # Lists to store training accuracy for each iteration
    train_accuracy = []

    for i in range(1, 201):  # Training 200 trees
        rf = RandomForestClassifier(n_estimators=i, random_state=42)
        rf.fit(x_train, y_train)
        y_pred = rf.predict(x_test)
        accuracy = accuracy_score(y_test, y_pred)
        train_accuracy.append(accuracy)

    # Final model evaluation
    final_rf = RandomForestClassifier(n_estimators=200, random_state=42)
    final_rf.fit(x_train, y_train)
    final_y_pred = final_rf.predict(x_test)

    # Confusion matrix
    cm = confusion_matrix(y_test, final_y_pred)
    print(f"Confusion Matrix:\n{cm}")

    # Classification report
    report = classification_report(y_test, final_y_pred)
    print(f"Classification Report:\n{report}")
    
    plot_feature_importance(feature_importance)
    plot_confusion_matrix(best_model, x_test, y_test)
    
    
def titanic():
    x = pd.read_csv("/Users/damie/Downloads/titanic/train.csv")
    x_test = pd.read_csv("/Users/damie/Downloads/titanic/test.csv")
    y_test = pd.read_csv("/Users/damie/Downloads/titanic/gender_submission.csv")
    columns_to_remove_caract_train = ["Name", "Ticket", "Cabin", "Embarked","PassengerId"]
    x = x.drop(columns_to_remove_caract_train, axis=1)
    x_test = x_test.drop(columns_to_remove_caract_train, axis=1)
    x['Sex'] = x['Sex'].map({'male': 0, 'female': 1})
    x_test['Sex'] = x_test['Sex'].map({'male': 0, 'female': 1})
    x = x.fillna(0)
    x_test = x_test.fillna(0)
    y_test = y_test.drop("PassengerId", axis=1)

    x_train = x.drop("Survived", axis=1)
    y_train = x["Survived"]

    # Define the RandomForestClassifier
    rf = RandomForestClassifier(n_estimators=200, random_state=42)

    # Fit the model
    rf.fit(x_train, y_train)

    # Get feature importances
    feature_importance = pd.DataFrame({'Feature': x_train.columns, 'Importance': rf.feature_importances_})
    feature_importance = feature_importance.sort_values(by='Importance', ascending=False)
    print(feature_importance)

    # Define the hyperparameters and their ranges to search
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 5, 10, 15],
        'min_samples_split': [2, 5, 10]
    }

    # Initialize GridSearchCV
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1)

    # Fit the data to find the best hyperparameters
    grid_search.fit(x_train, y_train)

    # Get the best parameters and the best model
    best_model = grid_search.best_estimator_
    joblib.dump(best_model, 'random_forest_model.pkl')

    y_pred = best_model.predict(x_test)

    # Calculate the accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"The precision of the model is : {accuracy}")
    
    plot_feature_importance(feature_importance)
    plot_confusion_matrix(best_model, x_test, y_test)
    print(f"The precision of the model is : {accuracy}") 
    
    
    
def churn_subscription():

    db = pd.read_csv("/Users/damie/Downloads/Telco_customer_churn_dataset/Copie de Telco_customer_churn.csv")
    db['Gender'] = db['Gender'].map({'Male': 0, 'Female': 1})
    db['Contract'] = db['Contract'].map({'Month-to-month': 0, 'One year': 1, 'Two year': 2})
    db['Payment Method'] = db['Payment Method'].map({'Bank transfer (automatic)':0,'Credit card (automatic)':1,'Electronic check':2,'Mailed check':3})
    columns_to_remove_caract_train=["CustomerID","Country","State","City","Churn Reason","Zip Code","Lat Long","Latitude","Longitude","Count"]
    # Convert 'Total Charges' column to numeric
    db['Total Charges'] = pd.to_numeric(db['Total Charges'], errors='coerce')
    df = db.drop(columns_to_remove_caract_train,axis=1)
    df = df.fillna(0)

    # Split into features and target variable
    X = df.drop('Churn Score', axis=1)
    y = df['Churn Score']

    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Define a grid of hyperparameters to search
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [5, 10, 15]
        }

    # Initialize GridSearchCV and fit it to your data
    grid_search = GridSearchCV(RandomForestRegressor(), param_grid, cv=5)
    grid_search.fit(X_train, y_train)

    # Get the best parameters and the best model
    best_params = grid_search.best_params_
    best_model = grid_search.best_estimator_

    # Evaluate the best model
    y_pred_best = best_model.predict(X_test)
    mse_best = mean_squared_error(y_test, y_pred_best)
    print(f"Best Mean Squared Error: {mse_best}")
    print(f"Best Parameters: {best_params}")
    
    r_squared = r2_score(y_test, y_pred_best)
    print(f"R-squared: {r_squared}")
    

def plot_feature_importance(feature_importance):
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=feature_importance, palette='viridis')
    plt.title('Feature Importance')
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.show()

def plot_confusion_matrix(model, x_test, y_test):
    plt.figure(figsize=(8, 6))
    sns.set(font_scale=1.2)
    plot = sns.heatmap(confusion_matrix(y_test, model.predict(x_test)), annot=True, cmap='Blues', fmt='g')
    plot.set_title('Confusion Matrix')
    plot.set_xlabel('Predicted Labels')
    plot.set_ylabel('True Labels')
    plt.show()
    
    
flowers()   
titanic()
