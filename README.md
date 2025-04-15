# Bank Employee Churn Prediction using Artificial Neural Network (ANN) - Learning Journey

This project was undertaken to learn and implement a machine learning model for predicting customer churn in a bank using an Artificial Neural Network (ANN). The goal was to understand the process of building a binary classification model with ANNs, from data preparation to deployment.

## Project Overview

In this project, we aimed to predict whether a customer will exit (churn) or not based on various features in the dataset. The approach involved using an Artificial Neural Network (ANN) built with TensorFlow and Keras. The project covered several key stages:

1.  **Data Preprocessing:** Preparing the raw data for the model by handling categorical features and scaling numerical features.
2.  **Model Building:** Designing and constructing the architecture of the ANN suitable for binary classification.
3.  **Model Training:** Training the ANN on the prepared data to learn the patterns that indicate customer churn.
4.  **Deployment:** The trained model has been deployed as an interactive web application using Streamlit.

## Visit the Deployed Application

You can visit the live deployed application using the following link:

[**https://churnprediction-vuxrmx7wy3k3rzksrk6yzi.streamlit.app/**]

Feel free to interact with the application by entering different customer features to see the churn predictions.

## What I Learned

Through this project, I gained valuable insights and practical experience in the following areas:

### 1. Data Preprocessing

* **Handling Categorical Data:** I learned how to convert categorical features like 'Gender' (using Label Encoding) and 'Geography' (using One-Hot Encoding) into numerical representations suitable for a neural network.
* **Feature Scaling:** I understood the importance of scaling numerical features (using `StandardScaler`) for neural network training.
* **Data Splitting:** I learned how to split the dataset into training, validation, and testing sets to train the model effectively and evaluate its generalization ability.
* **Handling Imbalanced Data (If Applicable):** [**Optional: If you dealt with imbalanced churn data, mention it here and the techniques you used, e.g., oversampling, undersampling, or using appropriate loss functions.**]

### 2. Model Building with ANN for Classification

* **Neural Network Architecture:** I gained experience in designing a sequential neural network with multiple dense layers, choosing appropriate activation functions (ReLU for hidden layers and sigmoid for the output layer in binary classification).
* **Output Layer for Binary Classification:** I learned how to use a sigmoid activation function in the output layer to obtain probabilities for the two classes (churned or not churned).
* **Input Shape:** I learned how to define the input layer of the ANN based on the number of features in the dataset.

### 3. Model Training

* **Choosing Optimizer and Loss Function:** I understood how to select an appropriate optimizer (Adam) and loss function (Binary Cross-Entropy) for a binary classification problem.
* **Metrics:** I learned how to track relevant metrics for classification (e.g., accuracy, precision, recall, F1-score) during the training process.
* **Callbacks:** I explored the use of callbacks like TensorBoard for visualizing training progress and EarlyStopping to prevent overfitting.

### 4. Deployment with Streamlit

* **Creating a User Interface:** I learned how to use the Streamlit library to create a user interface to input customer features and get churn predictions.
* **Loading Saved Models and Preprocessors:** I gained experience in saving and loading trained models (`.h5` files) and preprocessing objects (scalers and encoders using `pickle`) for deployment.
* **End-to-End Prediction:** I understood the flow of taking user inputs, preprocessing them, using the loaded model to predict the probability of churn, and displaying the prediction (e.g., as churn or no churn based on a threshold).

## Key Takeaways

* Building a classification model with an ANN involves a similar systematic process as regression, but with different choices for the output layer and loss function.
* Understanding and addressing potential issues like imbalanced data is important in classification tasks.
* Evaluating classification models requires using appropriate metrics beyond just accuracy.
* Streamlit provides a powerful way to deploy machine learning models for classification as interactive applications.

This project provided a valuable learning experience in applying artificial neural networks to a binary classification problem (customer churn prediction) and understanding the key steps involved in the machine learning pipeline for classification.
