import sys
import re
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    
    # import datasets
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    
    # merge datasets
    df = messages.merge(right=categories, how='inner', on='id')

    return df


def clean_data(df):
    
    # create a dataframe of the 36 individual category columns
    categories = df['categories'].str.split(pat=';', expand=True)
    categories.columns = [re.sub(pattern='-(0|1)', repl='', string=col) for col in categories.loc[0,:]]

    # set each category value to numbers 0/1
    for column in categories:
        categories[column] = [int(value[-1]) for value in categories[column].astype(str)]
    
    # drop the original categories column from df
    df.drop(labels='categories', axis=1, inplace=True)
    
    # concatenate the original dataframe with the new categories dataframe
    df = pd.concat(objs=[df, categories], axis=1)
    
    # drop duplicate records from df
    df.drop_duplicates(inplace=True)
    
    return df


def save_data(df, database_filename):
    
    engine = create_engine('sqlite:///'+database_filename)
    df.to_sql(name='messages', con=engine, index=False, if_exists='replace')


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()