import pandas as pd

import os
import pickle # Saving and loading models....
import logging


def titanic_preprocess(df):
    def _preprocess_age(df):
        df.loc[ df['Age'] <= 16, 'Age'] = 0
        df.loc[(df['Age'] > 16) & (df['Age'] <= 26), 'Age'] = 1
        df.loc[(df['Age'] > 26) & (df['Age'] <= 36), 'Age'] = 2
        df.loc[(df['Age'] > 36) & (df['Age'] <= 62), 'Age'] = 3
        df.loc[ df['Age'] > 62, 'Age'] = 4
        df['Age'] = df['Age'].astype(int)
        return df
    def _preprocess_fare(df):
        df.loc[ df['Fare'] <= 7.91, 'Fare'] = 0
        df.loc[(df['Fare'] > 7.91) & (df['Fare'] <= 14.454), 'Fare'] = 1
        df.loc[(df['Fare'] > 14.454) & (df['Fare'] <= 31), 'Fare']   = 2
        df.loc[(df['Fare'] > 31) & (df['Fare'] <= 99), 'Fare']   = 3
        df.loc[(df['Fare'] > 99) & (df['Fare'] <= 250), 'Fare']   = 4
        df.loc[ df['Fare'] > 250, 'Fare'] = 5
        df['Fare'] = df['Fare'].astype(int)
        return df
    def _preprocess_name(df):
        df['Name'] = df.Name.str.extract(' ?([A-Za-z]+)\.', expand = False)
        df['Name'].unique().tolist()
        df.rename(columns={'Name' : 'Title'}, inplace=True)
        # df['Title'] = df['Title'].replace(['Mme', 'Mlle', 'Ms', 'Miss'], 'Miss')
        df['Title'] = df['Title'].replace(['Rev', 'Dr', 'Col', 'Ms', 'Mlle', 'Major', 'Countess', 'Capt', 'Dona', 'Jonkheer', 'Lady', 'Sir', 'Mme', 'Don'], 'Other')
        return df
    
    # Converting ['PassengerId', 'Pclass','Age', 'SibSp', 'Parch', 'Fare'] features to numeric values
    df[['PassengerId', 'Pclass','Age', 'SibSp', 'Parch', 'Fare']] = df[['PassengerId', 'Pclass','Age', 'SibSp', 'Parch', 'Fare']].astype(float)

    # Preprocessing the Age feature
    df = _preprocess_age(df)

    # Preprocessing the Fare feature
    df = _preprocess_fare(df)

    # Preprocessing the Name feature
    df = _preprocess_name(df)
    
    return df

def titanic_transformer(df):
    def _transform_titanic(df):
        #Feature Transformation.....
        df['Sex'].replace({'male':0, 'female':1}, inplace=True)
        df['Embarked'].replace({'S':0, 'C':1,'Q':2}, inplace=True)
        df['Title'].replace({'Mr':0, 'Mrs':1,'Miss':2,'Master':3,'Other':4}, inplace=True)
        
        return df

    def _generate_titanic(df):
        # Creating Relatives feature by combining the SibSp and Parch features
        df['Relatives'] = df['SibSp'] + df['Parch']

        #Feature Generation...
        df.loc[df['Relatives'] > 0, 'Not_alone'] = 0
        df.loc[df['Relatives'] == 0, 'Not_alone'] = 1
        df['Not_alone'] = df['Not_alone'].astype(int)

        return df
    
    df = _transform_titanic(df)
    df = _generate_titanic(df)

    return df

def titanicDropFeatures(df):
    # Drop the columns that we don't need
    df.drop(columns = ['Cabin','PassengerId'],axis = 1,inplace = True)

    #Dropping the redundant columns...
    #Since the Ticket attribute has 681 unique tickets, it will be a bit lengthy to convert them into useful categories. So we will drop it from the dataset.
    df.drop(columns = ['Pclass','Age','SibSp','Not_alone','Ticket'],axis = 1,inplace = True)

    return df


def adultTransformer(X):
    # copy the array
    df = X.copy()
    df['capital_gain'] = df['capital_gain'] - df['capital_loss']

    df['education'] = df['education_num']

    df['education'] = df['education'].astype('int')

    # Dropping Residual features after features generation
    df.drop(['education_num', 'capital_loss', 'relationship'], axis = 1, inplace = True)

    return df

def createEncoder(X, y):
    # copy the array
    df = X.copy()
    df['income'] = y

    # Encoding the categorical variables
    encoder = {}
    #iterate through the columns
    for col in [i for i in df.columns if df[i].dtypes=='O' and i != 'income']:
        if col != 'marital_status':
            encoder[col] = df[[col, 'income']].groupby([col]).mean().to_dict()['income']
        else:
            encoder[col] = {
                'Never-married': 0,
                'Divorced': 0,
                'Separated': 0,
                'Widowed': 0,
                'Married-civ-spouse': 1,
                'Married-spouse-absent': 1,
                'Married-AF-spouse': 1 
            }
    logging.info('Encoder created')    
    return encoder

def adultCategoricalEncoder(X, encoder):
    # copy the array
    df = X.copy()
    
    for col in df.columns[df.dtypes=='O']:
        df[col] = df[col].map(dict(zip(list(df[col].unique()), list(map(encoder[col].get, list(df[col].unique())))))).astype('float')
    logging.info('Categorical Encoder applied')
    return df

def create_path(directory, filename):
    """
    This creates the path for the model

    Args:
        filename (str): path to save the Trained model
    """
    if directory != 'logs':
        logging.info(f'Creating the path for {filename}')
    os.makedirs(directory, exist_ok = True) # ONLY CREATE IF MODEL DIRECTORY DOESN'T EXISTS
    filePath = os.path.join(directory, filename) # models/filename
    if directory != 'logs':
        logging.info(f'Created the path for {filename}') 

    return filePath

def saveModel(model, filename):
    """
    This saves the model

    Args:
        model (python object): Trained model to 
        filename (str): path to save the Trained model
    """
    logging.info('Saving Trained model')
    filePath = create_path('models', filename)
    pickle.dump(model, open(filePath, 'wb'))
    logging.info(f'Saved the model to {filePath}')
    logging.info("====="*10)


def try_typecasting(cls, x):
    """
    This function tries to typecast the data

    Args:
        cls (class): type of the data
        x (str): data to be typecasted
    """
    try:
        return cls(x)
    except:
        return x
