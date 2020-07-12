import sys
import re
import pickle
import pandas as pd
import numpy as np
from sqlalchemy import create_engine

import nltk
nltk.download(['punkt', 'stopwords', 'wordnet', 'averaged_perceptron_tagger'])

from nltk import pos_tag
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from sklearn.multioutput import MultiOutputClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
#from sklearn.naive_bayes import MultinomialNB

import warnings
warnings.filterwarnings('ignore')


def load_data(database_filepath):
    
    table_name='messages'
    
    # load data from database
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table(table_name, engine)
    
    # category_names
    category_names = df.columns[4:]
    #category_names = [col for col in df.columns if re.search(pattern='-(0|1)', string=col)]
    
    # drop records with missing labels
    df = df.dropna(subset=category_names, how='any')
    
    df = df[:100]
        
    # X, y
    X = df['message'] #.values
    y = df[category_names] #.values
    
    return X, y, category_names
    

def tokenize(text):
    
    def lemmatize_all(text):
    
        lemmatizer = WordNetLemmatizer()

        tokens_lemmed = []
        # create tokens from text
        for word, tag in pos_tag(word_tokenize(text)):
            # remove stop words from tokens
            if word not in stopwords.words('english'):

                # clean tokens
                if tag.startswith("NN"):
                    lemmed = lemmatizer.lemmatize(word, pos='n')
                elif tag.startswith('VB'):
                    lemmed = lemmatizer.lemmatize(word, pos='v')
                elif tag.startswith('JJ'):
                    lemmed = lemmatizer.lemmatize(word, pos='a')
                else:
                    lemmed = word

                tokens_lemmed.append(lemmed)
        
        return tokens_lemmed

    # normalize text
    text_normalized = re.sub(pattern=r'[^a-zA-Z0-9]', repl=' ', string=text.lower())
    
    # create tokens from text, remove stopwords from tokens, clean tokens
    text_cleaned = lemmatize_all(text_normalized)
    
    return text_cleaned


def build_model():
    
    # classifier
    #classifier_rf = RandomForestClassifier(n_estimators=200)
    #classifier_knn = KNeighborsClassifier()
    #classifier_ada = AdaBoostClassifier()
    #classifier_dt = DecisionTreeClassifier()
        
    # pipeline with vectorizer and classifier
    pipeline_rf = Pipeline([
        ('vect', TfidfVectorizer(tokenizer=tokenize)), # CountVectorizer + TfidfTransformer
        ('clf', MultiOutputClassifier(RandomForestClassifier(n_estimators=200, verbose=1)))
    ])
    
    # gridsearch
    gridsearch_params = {
        'clf__estimator__min_samples_leaf': [1, 10],
        'clf__estimator__max_features': ['auto', 'log2'],
        'vect__smooth_idf': [True]
    }
    gridsearch_rf = GridSearchCV(pipeline_rf, param_grid=gridsearch_params, n_jobs=-1)
                
    return gridsearch_rf
    
    
def evaluate_model(model, X_test, Y_test, category_names):
    
    def performance_report_detailed(y_true, y_pred):
    
        y_cols = list(y_true.columns) 
        report = classification_report(y_true, y_pred, target_names=y_cols, zero_division=0)

        print(report)
    
    def performance_report_summarised(y_true, y_pred, output_df=True):
    
        y_cols = list(y_true.columns) 
        report = classification_report(y_true, y_pred, target_names=y_cols, zero_division=0, output_dict=True)

        selected_keys = ['micro avg', 'macro avg', 'weighted avg', 'samples avg']
        report = {key: report[key] for key in selected_keys}
        if output_df:
            report = pd.DataFrame.from_dict(report, orient='index')
        
        return report
        
    # predict
    y_test_pred = model.predict(X_test)
        
    print('Test performance:\n\n{}'.format(performance_report_summarised(Y_test, y_test_pred)))


def save_model(model, model_filepath):
    
    with open(model_filepath, 'wb') as f:
        pickle.dump(model, f)

        
def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
        
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