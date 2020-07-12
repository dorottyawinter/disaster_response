import json
import plotly
import pandas as pd

import nltk
nltk.download(['punkt', 'stopwords', 'wordnet', 'averaged_perceptron_tagger'])
from nltk import pos_tag
from nltk.corpus import stopwords

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
#import sklearn.external.joblib as extjoblib
import joblib
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
engine = create_engine('sqlite:///C:/Users/DorottyaWinter/Documents/git/disaster_response/data/DisasterResponse.db')
df = pd.read_sql_table('messages', engine)

# load model
model = joblib.load("C:/Users/DorottyaWinter/Documents/git/disaster_response/models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
        
    # calculate disaster message counts per genre and related status (related/not related)   
    genre_related = df[df['related']==1].groupby('genre').count()['message']
    genre_not_rel = df[df['related']==0].groupby('genre').count()['message']
    genre_names = list(genre_related.index)
    
    # calculate incidences per categories (rate of occurences)
    cat_incidence = df.drop(['id', 'message', 'original', 'genre'], axis=1).sum()/len(df)
    cat_incidence = cat_incidence.sort_values(ascending = False)
    cat_names = list(cat_incidence.index)
     

    # create visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_related,
                    name='related',
                    marker_color='#7940bf'
                ),
                Bar(
                    x=genre_names,
                    y=genre_not_rel,
                    name='not related',
                    marker_color='#af8cd9'
                )
            ],

            'layout': {
                'title': 'Distribution of disaster messages per genre and related status',
                'yaxis': {
                    'title': "count"
                },
                'xaxis': {
                    'title': "genre"
                },
                'barmode': 'group'
            }
        },
        {
            'data': [
                Bar(
                    x=cat_names,
                    y=cat_incidence,
                    marker_color='#141414'
                )
            ],

            'layout': {
                'title': 'Proportion of positive occurences per category',
                'yaxis': {
                    'title': "proportion"
                },
                'xaxis': {
                    'title': "category",
                    'tickangle': -30
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
    #app.run(host='0.0.0.0', port=3001, debug=True)
    app.run(host='localhost', port=3001, debug=True)


if __name__ == '__main__':
    main()