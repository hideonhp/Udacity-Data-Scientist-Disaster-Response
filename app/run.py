import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from sklearn.externals import joblib
from sqlalchemy import create_engine


app = Flask(__name__)

def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('messages', engine)

# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    
    # Create a DataFrame containing only the category columns by dropping irrelevant columns
    df_categories = df.drop(["id", "message", "original", "genre"], axis=1)

    # Sum the values across each category column to get the total count of messages for each category
    category_counts = df_categories.sum(axis=0)

    # Extract the names of the category columns to use them in visualizations or analysis
    category_names = df_categories.columns
    
    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    # Create a list of graphs to be rendered on the webpage
    graphs = [
        {
            'data': [
                Bar(  # Create a bar chart for message genres
                    x=genre_names,  # X-axis represents the names of the genres
                    y=genre_counts   # Y-axis represents the counts of messages in each genre
                )
            ],
            'layout': {
                'title': 'Distribution of Message Genres',  # Title of the graph
                'yaxis': {
                    'title': "Count"  # Label for the Y-axis
                },
                'xaxis': {
                    'title': "Genre"  # Label for the X-axis
                }
            }
        }, 
        {
            'data': [
                Bar(  # Create a bar chart for message categories
                    x=category_names,  # X-axis represents the names of the categories
                    y=category_counts   # Y-axis represents the counts of messages in each category
                )
            ],
            'layout': {
                'title': 'Distribution of Message Categories',  # Title of the graph
                'yaxis': {
                    'title': "Count"  # Label for the Y-axis
                },
                'xaxis': {
                    'title': "Category"  # Label for the X-axis
                }
            }
        }
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3000, debug=True)


if __name__ == '__main__':
    main()