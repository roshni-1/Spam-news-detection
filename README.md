# Spam News Detection Using Neural Network 

## Table of Contents
- [Introduction](#introduction)
- [Data Fetching](#data-fetching)
- [Data Preprocessing](#data-preprocessing)
- [Feature Engineering](#feature-engineering)
- [Model Training](#model-training)
- [Hyperparameter Tuning](#hyperparameter-tuning)
- [Model Evaluation](#model-evaluation)
- [Model Interpretation with LIME](#model-interpretation-with-lime)
- [Model Saving](#model-saving)
- [Conclusion](#conclusion)

## Introduction
This project aims to develop a machine learning model for detecting spam news articles using natural language processing (NLP) techniques. The workflow includes fetching news data from an API, preprocessing the data, training a model, evaluating its performance, and finally saving the model for future use.

## Data Fetching
1. **API Key and Endpoint**: The project utilizes the MediaStack API to fetch news articles. An API key is required to authenticate requests.
2. **Data Retrieval**: A function is implemented to fetch articles based on keywords, date range, and language. This function supports pagination to retrieve multiple articles efficiently.
3. **Data Storage**: The fetched articles are stored in a DataFrame and saved as a CSV file for further processing.

## Data Preprocessing
1. **Combining Columns**: The title and description of each article are combined into a single text column to provide more context for the model.
2. **Text Cleaning**: A cleaning function is applied to preprocess the text data, which includes:
   - Removing URLs
   - Converting text to lowercase
   - Tokenizing text into words
   - Removing stopwords (common words that add little meaning)

## Feature Engineering
1. **Creating Target Labels**: A set of spam-related keywords is defined. A function is used to label articles as spam or not spam based on the presence of these keywords in the title or description.
2. **Vectorization**: The cleaned text data is transformed into numerical features using TF-IDF (Term Frequency-Inverse Document Frequency) vectorization, which reflects the importance of words in the context of the dataset.

## Model Training
1. **Model Selection**: A Bidirectional LSTM (Long Short-Term Memory) model is chosen for the classification task due to its effectiveness in handling sequential data.
2. **Training the Model**: The model is trained on the training dataset, with evaluation performed on a validation dataset to monitor its performance.

## Hyperparameter Tuning
1. **Optimization**: Keras Tuner is employed to fine-tune the model's hyperparameters, such as the embedding dimension, number of LSTM units, and dropout rates, to enhance model performance.

## Model Evaluation
1. **Performance Metrics**: The model's performance is evaluated using accuracy, precision, recall, and F1-score. The results are compared against baseline models to assess improvement.
2. **Visualizations**: Various plots are created to visualize training and validation accuracy and loss, allowing for easy assessment of model performance over epochs.

## Model Interpretation with LIME
1. **Local Interpretability**: The LIME (Local Interpretable Model-agnostic Explanations) library is utilized to provide insights into model predictions. This helps in understanding which features influence the model's decisions for individual predictions.
2. **Example Explanations**: Explanations are generated for a subset of articles to demonstrate how the model identifies spam versus non-spam content.

## Model Saving
1. **Model Serialization**: The trained model is saved in the HDF5 format using Keras's built-in functionalities. This allows for easy loading and deployment of the model in the future.

## Conclusion
This project successfully demonstrates the process of building a spam news detection system using machine learning and natural language processing techniques. By following the outlined steps, one can replicate the workflow and adapt it for different datasets or classification tasks. The model is ready for deployment and can be integrated into applications requiring spam detection capabilities.
