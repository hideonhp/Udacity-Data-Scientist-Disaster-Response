import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    """
    Load and merge messages and categories datasets.

    This function reads two CSV files: one containing message data and the other containing
    category data. It merges these datasets into a single DataFrame based on the common "id" column.

    Parameters:
    - messages_filepath (str): The file path to the messages CSV file.
    - categories_filepath (str): The file path to the categories CSV file.

    Returns:
    - df (pd.DataFrame): A merged DataFrame containing both message and category data.
    """
    # Read the messages data from the specified CSV file
    messages = pd.read_csv(messages_filepath)
    # Read the categories data from the specified CSV file
    categories = pd.read_csv(categories_filepath)

    # Merge the messages and categories DataFrames on the "id" column
    df = messages.merge(categories, how="outer", on="id")
    # Display the first few rows of the merged DataFrame for verification
    print(df.head())
    return df


def clean_data(df):
    """
    Clean and transform the categories data in the DataFrame.

    This function processes the 'categories' column in the input DataFrame by splitting
    it into separate category columns, converting the values to integers, and removing
    any duplicate rows. The cleaned categories are then merged back into the original DataFrame.

    Parameters:
    - df (pd.DataFrame): The input DataFrame containing a 'categories' column.

    Returns:
    - df (pd.DataFrame): The cleaned DataFrame with individual category columns.
    """
    # Split the 'categories' column into separate columns based on the delimiter ";"
    categories = df["categories"].str.split(";", expand=True)
    
    # Extract the category names from the first row of the new DataFrame
    row = categories.iloc[[1]]
    category_colnames = [x.split("-")[0] for x in row.values[0]]
    categories.columns = category_colnames  # Assign category names as column headers
    
    # Convert the values in each category column to integers (1 or 0)
    for column in categories:
        categories[column] = categories[column].astype(str).str[-1]  # Extract the last character
        categories[column] = categories[column].astype(int)  # Convert to integer
    
    # Drop the original 'categories' column from the DataFrame
    df.drop(["categories"], axis=1, inplace=True)
    
    # Concatenate the cleaned categories back into the original DataFrame
    df = pd.concat([df, categories], join="outer", axis=1)
    
    # Remove duplicate rows from the DataFrame
    df = df.drop_duplicates()
    
    return df


def save_data(df, database_filename):
    """
    Save the DataFrame to a SQLite database.

    This function connects to a SQLite database specified by the `database_filename`
    and saves the given DataFrame to a table named "messages". If the table already
    exists, it will be replaced with the new data.

    Parameters:
    - df (pd.DataFrame): The DataFrame containing data to be saved.
    - database_filename (str): The file path for the SQLite database.

    Returns:
    - None: The function does not return any value.
    """
    # Create a database engine to connect to the SQLite database
    engine = create_engine(f'sqlite:///{database_filename}')
    # Save the DataFrame to the database, replacing the existing 'messages' table if it exists
    df.to_sql("messages", engine, index=False, if_exists="replace")


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