Breast Cancer Classification

This project implements a Breast Cancer Classification System that classifies tumors into Malignant or Benign based on tumor features. The project uses Python, Pandas, NumPy, and scikit-learn with a Logistic Regression model.

Table of Contents

Project Overview

Dataset

Installation

Usage

Model Prediction Example

Technologies Used

License

Project Overview

Breast cancer is one of the most common cancers. Early detection is crucial for treatment and survival. This project builds a machine learning model that predicts whether a tumor is Malignant (cancerous) or Benign (non-cancerous) based on numeric features extracted from medical images.

The model uses Logistic Regression, a classification algorithm suitable for predicting binary outcomes.

Dataset

The project uses the Breast Cancer Wisconsin Dataset from scikit-learn.

Features:
The dataset contains 30 features for each tumor, including:

Radius, Texture, Perimeter, Area

Smoothness, Compactness, Concavity

Symmetry, Fractal Dimension, etc.

Target:

0 = Malignant

1 = Benign

Installation

Clone the repository and install the required libraries:

git clone https://github.com/yourusername/breast-cancer-classification.git
cd breast-cancer-classification
pip install -r requirements.txt


Requirements (requirements.txt):

numpy
pandas
scikit-learn

Usage

You can use the model to predict breast cancer type for a new tumor using its features.

import numpy as np
import pickle

# Load trained model
model = pickle.load(open('breast_cancer_model.pkl', 'rb'))

# Input data (example)
input_data = (13.54, 14.36, 87.46, 566.3, 0.09779, 0.08129, 0.06664, 0.04781,
              0.1885, 0.05766, 0.2699, 0.7886, 2.058, 23.56, 0.008462, 0.0146,
              0.02387, 0.01315, 0.0198, 0.0023, 15.11, 19.26, 99.7, 711.2, 0.144,
              0.1773, 0.239, 0.1288, 0.2977, 0.07259)

# Convert to numpy array and reshape
input_data_as_numpy_array = np.asarray(input_data).reshape(1, -1)

# Make prediction
prediction = model.predict(input_data_as_numpy_array)

# Print result
if prediction[0] == 0:
    print('The Breast Cancer is Malignant')
else:
    print('The Breast Cancer is Benign')

Model Prediction Example

Input Features:
(13.54, 14.36, 87.46, 566.3, 0.09779, 0.08129, 0.06664, 0.04781, …)

Output:

The Breast Cancer is Malignant

Technologies Used

Python 3

NumPy for numerical operations

Pandas for data manipulation

scikit-learn for machine learning

Logistic Regression for classification

License

This project is licensed under the MIT License.
