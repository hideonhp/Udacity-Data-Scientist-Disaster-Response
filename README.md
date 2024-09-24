Udacity - Data Science - Disaster Response Pipeline Project

![image](https://github.com/user-attachments/assets/8f1da038-08ff-4b75-af8b-689fa6052a20)

## Table of Contents
1. [Description](#description)
2. [Getting Started](#getting_started)
	1. [Dependencies](#dependencies)
	2. [Installing](#installation)
	3. [Executing Program](#execution)
  4. [Additional Material](#material)
  5. [Important Files](#importantfiles)  
3. [Authors](#authors)
4. [Acknowledgement](#acknowledgement)
5. [Screenshots](#screenshots)

<a name="descripton"></a>
## Description

This project is a component of Udacity and Figure Eight's Data Science Nanodegree Program. Pre-labeled tweets and communications from actual catastrophe occurrences are included in the collection. The goal of the research is to develop a real-time message classification model using natural language processing (NLP).

The following are the main sections that comprise this project:

1. Processing data and creating an ETL pipeline to clean up, extract, and store the data in a SQLite database
2. Create a pipeline for machine learning to train algorithms that can categorize text messages into different groups.
3. Launch a web application that displays real-time model results.

<a name="getting_started"></a>
## Getting Started

<a name="dependencies"></a>
### Dependencies
* Python 3.5+
* Machine Learning Libraries: NumPy, SciPy, Pandas, Sciki-Learn
* Natural Language Process Libraries: NLTK
* SQLlite Database Libraqries: SQLalchemy
* Model Loading and Saving Library: Pickle
* Web App and Data Visualization: Flask, Plotly

<a name="installation"></a>
### Installing
To clone the git repository:
```
git clone https://github.com/hideonhp/Udacity-AI-programming-with-python-nanodegree.git
```

<a name="execution"></a>
### Executing Program:
1. You can run the following commands in the project's directory to set up the database, train model and save the model.

    - To run ETL pipeline to clean data and store the processed data in the database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/disaster_response_db.db`
    - To run the ML pipeline that loads data from DB, trains classifier and saves the classifier as a pickle file
        `python models/train_classifier.py data/disaster_response_db.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

<a name="material"></a>
### Additional Material

You will discover two Jupyter notebooks in the **data** and **models** folders that will walk you through the process of understanding how the model operates step-by-step:
1. **ETL Preparation Notebook**: Discover every detail of the deployed ETL pipeline.
2. **ML Pipeline Preparation Notebook**: Check out the Machine Learning Pipeline created using Scikit-Learn and NLTK.
   
The **ML Pipeline Preparation Notebook** may be utilized to re-train the model or optimize it via a specific Grid Search section.

<a name="importantfiles"></a>
### Important Files
**app/templates/***: templates/html files for web application

**data/process_data.py**: Extract Train Load (ETL) process utilized for data cleansing, feature extraction, and data storage in a SQLite database.

**models/train_classifier.py**: A machine learning pipeline that ingests data, trains a model, and exports the learned model as a .pkl file for future utilization.

**run.py**: This file serves to initiate the Flask web application (Python) designed for the classification of disaster alerts.

<a name="authors"></a>
## Authors
* [Bùi Đức Tiến](https://github.com/hideonhp)

<a name="acknowledgement"></a>
## Acknowledgements
* [Udacity](https://www.udacity.com/) for providing an amazing Data Science Nanodegree Program
* [Figure Eight](https://www.figure-eight.com/) for providing the relevant dataset to train the model

<a name="screenshots"></a>
## Screenshots

1. This is an example of a message we might type to see how well the model worked.
  ![image](https://github.com/user-attachments/assets/3e576397-f013-4ddf-9246-31cc7322a569)
2. Upon selecting **Classify Message**, the message's associated categories are indicated in green.
  ![image](https://github.com/user-attachments/assets/b86632c4-d7e3-4adf-88dd-aee5442e9f8a)
3. Figure Eight provides several graphs concerning the training dataset that are displayed on the main page.
  ![image](https://github.com/user-attachments/assets/c409160f-1bc0-4b7b-bbb0-3358f6d154b8)
