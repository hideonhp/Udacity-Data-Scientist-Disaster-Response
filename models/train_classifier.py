import sys
import re
import pandas as pd
import numpy as np
import pickle
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline, FeatureUnion
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import classification_report, accuracy_score, f1_score

#download sector
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

def load_data(database_filepath):
    """
    Load data from a SQLite database.

    This function connects to a SQLite database specified by the `database_filepath`,
    retrieves the "messages" table, and extracts features and target variables for
    machine learning tasks. 

    Parameters:
    - database_filepath (str): The file path to the SQLite database.

    Returns:
    - X (pd.Series): A pandas Series containing the message text.
    - Y (pd.DataFrame): A pandas DataFrame containing the target variables (categories).
    - category_names (list): A list of category names corresponding to the target variables.
    """
    print(database_filepath)
    # Create a database engine to connect to the SQLite database
    engine = create_engine("sqlite:///" + database_filepath, pool_pre_ping=True)
    # Read the "messages" table into a DataFrame
    df = pd.read_sql_table("messages", con=engine)
    # Extract the message column as the feature set
    X = df["message"]
    # Extract all columns starting from the 5th column as target variables
    Y = df.iloc[:, 4:]
    # Get a list of category names from the DataFrame columns
    category_names = list(df.columns[4:])
    
    return X, Y, category_names


def tokenize(text):
    """
    Tokenize and preprocess the input text.

    This function takes a string of text as input, detects and replaces URLs with a 
    placeholder, normalizes the text by converting it to lowercase, removes punctuation, 
    and eliminates English stopwords. It then lemmatizes the remaining tokens to their 
    base form.

    Parameters:
    - text (str): The input text to be tokenized and preprocessed.

    Returns:
    - clean_tokens (list): A list of cleaned and lemmatized tokens.
    """
    # Regular expression to detect URLs in the text
    url_regex = "http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+"
    detected_urls = re.findall(url_regex, text)
    # Replace detected URLs with a placeholder
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")
        
    # Normalize text: convert to lowercase and remove non-alphanumeric characters
    words = nltk.word_tokenize(re.sub(r"[^a-zA-Z0-9]", " ", text.lower()))
    
    # Remove English stopwords from the list of words
    tokens = [word for word in words if word not in stopwords.words("english")]
    
    # Lemmatize and clean the tokens
    clean_tokens = []
    lemmatizer = WordNetLemmatizer()
    for token in tokens:
        clean_token = lemmatizer.lemmatize(token).strip()  # Lemmatize and strip whitespace
        clean_tokens.append(clean_token)
    
    return clean_tokens


def build_model():
    """
    Build a machine learning model pipeline using AdaBoost for multi-output classification.

    This function constructs a pipeline that includes text vectorization, 
    transformation using TF-IDF, and a multi-output classifier based on 
    the AdaBoost algorithm. It also sets up a grid search to optimize 
    the maximum number of features to consider during vectorization.

    Returns:
    - cv (GridSearchCV): An instance of GridSearchCV configured with the pipeline and parameters.
    """
    # Define the pipeline with vectorization, TF-IDF transformation, and multi-output classification
    pipeline_adaboost = Pipeline([
        ("vect", CountVectorizer(tokenizer=tokenize)),  # Text vectorization using the custom tokenizer
        ("tfidf", TfidfTransformer()),                   # TF-IDF transformation
        ("clf", MultiOutputClassifier(AdaBoostClassifier()))  # Multi-output classification with AdaBoost
    ])
    
    # Set parameters for grid search to optimize the maximum features for vectorization
    parameters = {"vect__max_features": [200, 500, 1000]}
    
    # Initialize grid search with the pipeline and parameter grid
    cv = GridSearchCV(pipeline_adaboost, param_grid=parameters)
    
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    """
    Evaluate the performance of a trained machine learning model.

    This function uses the given model to predict the labels for the test dataset (X_test)
    and generates a classification report for each category in Y_test. It also calculates 
    the accuracy for each category and computes the overall accuracy across all categories.

    Parameters:
    - model: The trained machine learning model to evaluate.
    - X_test (pd.Series): The input features for the test set.
    - Y_test (pd.DataFrame): The true labels for the test set (multi-output).
    - category_names (list): The names of the categories corresponding to Y_test.

    Returns:
    - float: The average accuracy across all categories.
    """
    # Initialize a list to store accuracy scores for each category
    scores = []
    counter = 0
    # Predict labels for the test set using the model
    y_pred = model.predict(X_test)
    # Iterate over each feature in Y_test to generate evaluation metrics
    for feature in Y_test:
        print("Feature - {}: {}".format(counter + 1, feature))  # Print the feature number and name
        # Print classification report for the current feature
        print(classification_report(Y_test[feature], y_pred[:, counter]))
        # Calculate accuracy for the current feature
        acc = accuracy_score(Y_test.iloc[:, counter], y_pred[:, counter])
        scores.append(acc)  # Append the accuracy to the scores list
        counter += 1  # Increment the counter for the next feature
    # Print and return the average accuracy across all features
    print("Total Accuracy: {}".format(np.mean(scores)))
    return np.mean(scores)


def save_model(model, model_filepath):
    """
    Save the trained machine learning model to a specified file.

    This function serializes the given model using pickle and saves it to the
    specified file path. The model can later be loaded for predictions or further evaluation.

    Parameters:
    - model: The trained machine learning model to be saved.
    - model_filepath (str): The file path where the model will be saved, including the file name.

    Returns:
    - None: The function does not return any value.
    """
    # Construct the file name from the provided file path
    pkl_filename = '{}'.format(model_filepath)
    # Open the file in binary write mode and save the model using pickle
    with open(pkl_filename, 'wb') as file:
        pickle.dump(model, file)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()