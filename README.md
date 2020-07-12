# Disaster Response Web App

### Table of Contents

1. [Dependencies](#dependencies)
2. [Project Overview](#overview)
3. [Project Components](#components)
4. [Running Instructions](#instructions)
5. [Web app](#app)
6. [Warning](#warning)
7. [Licensing, Author, Acknowledgements](#licensing)

## Dependencies <a name="dependencies"></a>
Requires Python 3.7.6 and the following libraries:
* numpy (>= 1.19.0)
* pandas (>= 0.23.3)
* json (>= )
* SQLAlchemy (>= 1.2.18)
* nltk (>= 3.2.5)
* scikit-learn (>= 0.23.1)
* Flask (>= 0.12.4)
* plotly (>= 2.0.15)

## Project Overview <a name="overview"></a>
This is a web app where an emergency worker can input new messages during disaster events (earthquake, forest fire, etc.) and get classification results in several categories to facilitate the forwarding of these messages to the appropriate aid agencies. Besides that, the app also displays visualizations of the data.

The app uses a custom ETL pipeline and ML pipeline.

## Project Components <a name="components"></a>
There are three components of the project.

1. ETL Pipeline
In a Python script, `process_data.py` is the data cleaning pipeline that:
* Loads the `messages` and `categories` (labels) datasets
* Merges the two datasets
* Cleans the data
* Stores it in a SQLite database

2. ML Pipeline
In a Python script, `train_classifier.py` is the machine learning pipeline that:
* Loads data from the SQLite database
* Splits the dataset into training and test sets
* Builds a text processing and machine learning pipeline
* Trains and tunes a model using GridSearchCV
* Outputs results on the test set
* Exports the final model as a pickle file

3. Flask Web App
In a Python script, `app.py` is the Flask web app that:
* Uses the mentioned pipelines
* Displays visualizations of the data

## Running Instructions <a name="instructions"></a>
#### Run process_data.py
1. Save the `data` folder in the current working directory including `process_data.py`, `disaster_messages.csv` and `disaster_categories.csv`.
2. From the current working directory, run the following command:
`python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`

#### Run train_classifier.py
1. In the current working directory, create a folder called `models` and save `train_classifier.py` in it.
2. From the current working directory, run the following command:
`python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

#### Run the web app 
1. Save the app folder in the current working directory.
2. Run the following command in the app directory:
`python run.py`
3. Go to http://0.0.0.0:3001/

## Web app <a name="app"></a>
![app front](https://github.com/dorottyawinter/disaster_response/blob/master/app_front.jpg)
![app results](https://github.com/dorottyawinter/disaster_response/blob/master/app_results.jpg)

## Warning <a name="warning"></a>
In some categories, the proportion of positive occurrence is very low (<1%) in the training set. In these cases, even though the model performance is very high the results are not reliable.

## Licensing, Authors, Acknowledgements <a name="licensing"></a>
This app was completed as part of the [Udacity Data Scientist Nanodegree](https://udacity.com/nanodegrees/nd025). 
The dataset origins from [Figure Eight]. (www.figure-eight.com)