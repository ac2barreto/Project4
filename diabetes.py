# Import the modules
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import requests
import sqlite3
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import plot_tree
from sklearn import tree
from flask import Flask, request, jsonify, render_template

# Read the CSV file from the Resources folder into a Pandas DataFrame
data_path = Path('Resources/Healthcare-Diabetes.csv')
df1 = pd.read_csv(data_path) #df1 saved for testing
df=df1

# Create a SQLite database and a table with the appropriate schema
conn = sqlite3.connect('diabetes.db')
cursor = conn.cursor()

# Create table
cursor.execute('''
    CREATE TABLE IF NOT EXISTS movies (
        id INTEGER PRIMARY KEY,
        Pregnancies INTEGER,
        Glucose INTEGER,
        BloodPressure INTEGER,
        SkinThickness INTEGER,
        BMI FLOAT,
        DiabetesPedigreeFunction FLOAT,
        Age INTEGER,
        Outcome INTEGER
    );
''')

# Insert data into the table
df.to_sql('diabetes', conn, if_exists='replace', index=False)

# Close the connection
conn.close()

# Note: When cleaning the code - Data polishing and cleaning of out dataframe  
# was directly integrated into the SQL queries for clean and efficiency

# Create a SQLite database connection
conn = sqlite3.connect('diabetes.db')

# Execute the SQL queries and load the result into a DataFrame

# Separate the data into labels and features
# Separate the y variable, the labels
y_query = "SELECT Outcome FROM diabetes"

# Separate the X variable, the features
X_query = "SELECT Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age FROM diabetes"

y = pd.read_sql_query(y_query, conn)
X = pd.read_sql_query(X_query, conn)

# Close the connection
conn.close()

# Display the DataFrame
#test.head(3)

# Split the data using train_test_split
# Assign a random_state of 1 to the function
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

# Reshape y to a 1D array (n_samples, ) using ravel() to resolve the warning about data shape.
y_train = np.ravel(y_train)
y_test = np.ravel(y_test)

# Decision Trees
dt_model = DecisionTreeClassifier()
dt_model.fit(X_train, y_train)
dt_predictions = dt_model.predict(X_test)

#####################################################################
#####################################################################
#####################################################################
#####################################################################
#########                 FLASK SETUP          ######################
#####################################################################
#####################################################################
#####################################################################
#####################################################################

app = Flask(__name__)

@app.route('/')
def form():
    # Render the HTML form
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Use request.form to access form data
    data = request.form
    pred_df = pd.DataFrame({
        'Pregnancies': [data.get('Pregnancies')],
        'Glucose': [data.get('Glucose')],
        'BloodPressure': [data.get('BloodPressure')],
        'SkinThickness': [data.get('SkinThickness')],
        'Insulin': [data.get('Insulin')],
        'BMI': [data.get('BMI')],
        'DiabetesPedigreeFunction': [data.get('DiabetesPedigreeFunction')],
        'Age': [data.get('Age')]
    })
    prediction = dt_model.predict(pred_df)

        # Check the prediction value and set message accordingly
    if prediction[0] == 1:
        result = "According to our model, you might have Diabetes! Please check-in with a doc!"
    elif prediction[0] == 2:
        result = "According to our model, you might NOT have Diabetes! Please check-in with a doc anyways!"
    else:
        result = "unknown"  # Handle unexpected prediction values

  #  return jsonify({'Prediction': int(prediction[0])})
    return jsonify({'Prediction': result})

if __name__ == '__main__':
    app.run(debug=True, port=5000)