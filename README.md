# Project4


# Diabetes Prediction and Analysis

- This repository contains a comprehensive Python project aimed at predicting diabetes outcomes based on various features. Utilizing a dataset that includes measurements such as Glucose levels, Blood Pressure, BMI (Body Mass Index), and others, we implement and compare several machine learning models to assess their accuracy in predicting diabetes. 

## Project Structure

- Data Preprocessing: The initial stage involves reading the healthcare-related dataset from a CSV file into a Pandas DataFrame. This step is crucial for preparing the data for subsequent analysis and model training.

- Database Integration: We leverage SQLite to store the dataset, allowing for efficient data manipulation and querying. This approach simulates real-world scenarios where data is often fetched from databases.

- Exploratory Data Analysis (EDA): Through EDA, we visualize the distribution of outcomes (diabetes positive or negative) in the dataset using pie charts and inspect correlations between different features using heatmaps. This step is vital for understanding the data and guiding the feature selection process.

- Data Splitting: The dataset is split into training and test sets, ensuring that models are evaluated on unseen data to gauge their generalization capability.

- Feature Scaling: We apply standard scaling to the features to normalize the data, a critical step for many machine learning algorithms to perform optimally.

- Model Training and Evaluation: The core of the project involves training various machine learning models, including Support Vector Machines (SVM), K-Nearest Neighbors (KNN), Naive Bayes, Decision Trees, Random Forest, Gradient Boosting Machines (GBM), and Logistic Regression. Each model's performance is evaluated based on accuracy, and a detailed analysis is conducted for the Decision Tree model, including a confusion matrix and a classification report.

- Decision Tree Visualization: For a more intuitive understanding of how the Decision Tree model makes predictions, we generate and display a graphical representation of the tree.

- User Interaction for Predictions: The project includes an interactive segment where users can input their health metrics to receive a prediction regarding their diabetes status, demonstrating a practical application of the trained models.

- Exporting Results: Finally, the accuracies of the different models are summarized and exported to a CSV file, which can be used for further analysis or presentation purposes.

## Technologies Used

- Pandas & NumPy: For data manipulation and numerical operations.
- Matplotlib & Seaborn: For data visualization.
- SQLite: For database management.
- Scikit-learn: For implementing machine learning models and preprocessing.
- Pydotplus & IPython.display: For visualizing the decision tree.

## Dataset
- The dataset used in this project, Healthcare-Diabetes.csv, contains several health metrics that are indicative of whether an individual has diabetes. The features include Pregnancies, Glucose, Blood Pressure, Skin Thickness, Insulin, BMI, Diabetes Pedigree Function, Age, and the Outcome (0 for negative, 1 for positive). Dataset obtained from Kaggle: [LINK| https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database]

## Notes on Use
- Ensure you have the required libraries installed.
- Run the Python script to perform the analysis and model training. (There is a seperate .py script for this!)
- Input the data when prompted to receive a diabetes prediction. This is educational, check with a doctor no matter the outcome provided!!
- The repository contains data CSV, a python ypnb notebook with the whole code analysis, a .py script for any user to run, and various images and visual resources

Presentation: https://docs.google.com/presentation/d/1ZhL4FYGR7v_rv-YlsMroukcmpLiIzDNBknqVb6HGox0/edit?usp=sharing 


Authors:
Alana Castellano
Andrea Barreto
Francisco Diaz
Arezoo Houshi
Juan Diaz

