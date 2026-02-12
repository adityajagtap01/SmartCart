🛒 SmartCart – Customer Segmentation & Classification using Machine Learning
📌 Project Overview

SmartCart is a Machine Learning project focused on customer segmentation using clustering techniques and classification modeling. The goal of the project is to analyze customer behavior, identify meaningful segments, and build predictive models to classify customer responses based on purchasing patterns and demographic features.

This project demonstrates real-world data preprocessing, feature engineering, clustering, visualization, and supervised learning workflows.

🎯 Objectives

Perform customer segmentation using Clustering Algorithms

Analyze customer purchasing behavior

Build a classification model to predict customer response

Visualize clusters in 2D and 3D

Apply feature engineering for better model performance

📂 Dataset Features

The dataset includes customer-related features such as:

Income

Recency

NumDealsPurchases

NumWebPurchases

NumCatalogPurchases

NumStorePurchases

NumWebVisitsMonth

Complain

Age

Customer_Tenure_Days

TotalSpending

Total_Children

Education categories (One-hot encoded)

Living status (One-hot encoded)

Response (Target variable for classification)

🧠 Machine Learning Techniques Used
🔹 1. Data Preprocessing

Handling missing values

Converting datetime and timedelta features

Feature scaling (StandardScaler / MinMaxScaler)

One-hot encoding categorical variables

🔹 2. Clustering

K-Means Clustering

Elbow Method for optimal K selection

2D & 3D cluster visualization

🔹 3. Classification

Logistic Regression / Neural Network (TensorFlow)

Model training and evaluation

Accuracy & performance metrics

📊 Technologies & Libraries

Python 3.9

Pandas

NumPy

Matplotlib

Seaborn

Scikit-learn

TensorFlow (for neural network model)

Jupyter Notebook

📈 Project Workflow

Data Cleaning & Feature Engineering

Exploratory Data Analysis (EDA)

Feature Scaling

Clustering Implementation

Cluster Visualization

Classification Model Training

Model Evaluation

🚀 How to Run the Project
1️⃣ Clone the Repository
git clone https://github.com/your-username/smartcart.git
cd smartcart

2️⃣ Create Environment (Recommended)
conda create -n smartcart python=3.9
conda activate smartcart

3️⃣ Install Dependencies
pip install -r requirements.txt

4️⃣ Run Jupyter Notebook
jupyter notebook


Open the main notebook and run all cells.

📊 Sample Results

Customers segmented into distinct behavioral groups

High-spending vs low-spending clusters identified

Classification model achieved strong predictive performance

Visual representation of clusters in 3D space

🏆 Key Learnings

Importance of feature scaling in clustering

Handling timedelta and datetime features properly

Difference between unsupervised (clustering) and supervised (classification) learning

Model evaluation and debugging dtype errors

🔮 Future Improvements

Deploy model using Flask / FastAPI

Build interactive dashboard using Streamlit

Apply advanced clustering (DBSCAN / Hierarchical)

Hyperparameter tuning for improved accuracy
