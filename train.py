"""
author: Rohan Dhanraj
email: rdy5674@gmail.com
""" 

#IMPORT THE LIBRARIES....
import numpy as np # linear algebra....
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)....
# preprocessing and transforming data....
from utils import *

import statsmodels.api as sm
from patsy import dmatrices

# Model building and training....
from sklearn.preprocessing import StandardScaler, MinMaxScaler, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor

from  xgboost import XGBClassifier

from sklearn.datasets import load_boston

import logging


class Training:
    def __init__(self, items='all'):
        """Initialize training class
        Args:
            items: A String or a List of strings or 'all' training items
                default value: 'all' for training all the models on a single call
                Linear Regression,
                Logistic Regression,
                Decision Tree,
                Random Forest,
                XG Boost

        """
        self.lib = [items] if type(items)==str and items!='all' else items
        self.boston, self.affairs, self.titanic, self.adult = self.loadData()

    def loadData(self):
        """
        Load the data
        """
        logging.info('Loading the Boston data')
        boston = load_boston()
        boston_df = pd.DataFrame(boston.data)
        boston_df.columns = boston.feature_names
        boston_df['MDEV'] = boston.target
        logging.info('Boston Data Loaded: Successfully')
        logging.info("====="*10)

        logging.info('Loading the Affairs Data')
        affairs_df = sm.datasets.fair.load_pandas().data
        # add "affair" column: 1 represents having affairs, 0 represents not
        affairs_df['affair'] = (affairs_df.affairs > 0).astype(int)
        affairs_df = affairs_df.drop('affairs', axis=1)
        logging.info('Affairs Data Loaded: Successfully')
        logging.info("====="*10)

        logging.info('Loading the Titanic Data')
        titanic_url = 'https://raw.githubusercontent.com/BigDataGal/Python-for-Data-Science/master/titanic-train.csv'
        titanic_df = pd.read_csv(titanic_url)
        titanic_df.columns = ['PassengerId','Survived','Pclass','Name','Sex','Age','SibSp','Parch','Ticket','Fare','Cabin','Embarked']
        logging.info('Titanic Data Loaded: Successfully')
        logging.info('====='*10)

        logging.info('Loading Adult Census Income Data')
        adultTrain = pd.read_csv('http://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data', sep=', ', header = None, engine = 'python')
        adultTest = pd.read_csv('http://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test' , skiprows = 1, sep=', ', header = None, engine = 'python')
        col_labels = [
            'age',
            'workclass',
            'fnlwgt',
            'education',
            'education_num',
            'marital_status',
            'occupation',
            'relationship',
            'race',
            'sex',
            'capital_gain',
            'capital_loss',
            'hours_per_week',
            'native_country',
            'wage_class'
            ]

        adultTrain.columns = col_labels
        adultTest.columns = col_labels

        # Merging the Train and Test Datasets
        # 
        adult_df = pd.concat([adultTrain, adultTest], axis = 0)
        adult_df['income']=adult_df['wage_class'].map({'<=50K': 0, '>50K': 1, '<=50K.': 0, '>50K.': 1})
        adult_df = adult_df.drop(['wage_class'], axis = 1)
        logging.info('Adult Census Income Data Loaded: Successfully')
        logging.info('====='*10)

        return boston_df, affairs_df, titanic_df, adult_df

    
    def linearRegression(self):
        """
        Training Linear Regression Model
        """
        logging.info('Training Linear Regression Model')
        linearRegression_df = self.boston.copy()

        x = linearRegression_df.drop('MDEV', axis=1)
        y = linearRegression_df['MDEV']

        pre_process = ColumnTransformer(transformers=[
                                            (
                                                'drop_columns',
                                                'drop',
                                                [
                                                    'B'
                                                ]
                                            ),
                                            (
                                                'scale_data',
                                                StandardScaler(),
                                                list(x.columns)
                                            )
                                        ])

        linearRegression_pipeline = Pipeline(steps=[
                                                (
                                                    'pre_processing',
                                                    pre_process
                                                ),
                                                (
                                                    'model',
                                                    LinearRegression()
                                                )
                                        ]) 

        linearRegression_pipeline.fit(x, y)
        logging.info('Linear Regression Model Trained: Successfully')

        saveModel(linearRegression_pipeline, 'linearRegressorPipeline.sav')
        logging.info('Liner Regression Model Saved: Successfully')
        logging.info("====="*10)


    def logisticRegression(self):
        """
        Training Logistic Regression Model
        """
        logging.info('Training Logistic Regression Model')
        logisticRegression_df = self.affairs.copy()

        y, X = dmatrices('affair ~ rate_marriage + age + yrs_married + children + \
        religious + educ + C(occupation) + C(occupation_husb)', logisticRegression_df, return_type="dataframe")
        X = X.rename(columns = {'C(occupation)[T.2.0]':'occ_2',
        'C(occupation)[T.3.0]':'occ_3',
        'C(occupation)[T.4.0]':'occ_4',
        'C(occupation)[T.5.0]':'occ_5',
        'C(occupation)[T.6.0]':'occ_6',
        'C(occupation_husb)[T.2.0]':'occ_husb_2',
        'C(occupation_husb)[T.3.0]':'occ_husb_3',
        'C(occupation_husb)[T.4.0]':'occ_husb_4',
        'C(occupation_husb)[T.5.0]':'occ_husb_5',
        'C(occupation_husb)[T.6.0]':'occ_husb_6'})
        cols = X.columns

        y = np.ravel(y)

        #model = LogisticRegression(C=0.1, max_iter=500, solver='saga')
        logisticRegression = LogisticRegression(C=100)

        logisticRegression.fit(X, y)
        logging.info('Logistic Regression Model Trained: Successfully')

        saveModel([cols, logisticRegression], 'logisticRegressionClassifier.sav')
        logging.info('Logistic Regression Model Saved: Successfully')
        logging.info("====="*10)


    def decisionTree(self):
        """
        Training Decision Tree Model
        """
        logging.info('Training Decision Tree Model')
        decisionTree_df = self.titanic.copy()
        #FILL THE MISSING VALUES WITH THE MEAN & MODE VALUES.. 
        decisionTree_df['Embarked']=decisionTree_df['Embarked'].fillna(decisionTree_df['Embarked'].mode()[0])
        decisionTree_df['Age']=decisionTree_df['Age'].fillna(decisionTree_df['Age'].median())

        decisionTree_df = titanic_preprocess(decisionTree_df)
        decisionTree_df = titanic_transformer(decisionTree_df)
        decisionTree_df = titanicDropFeatures(decisionTree_df)

        #Feature Variables
        x = decisionTree_df.drop('Survived',axis=1)
        #Target Variable
        y = decisionTree_df['Survived']

        #Model Training
        decisionTreeClassifier = DecisionTreeClassifier(
            criterion='entropy',
            ccp_alpha=0.0021, 
            max_depth=16,
            min_samples_leaf=1,
            min_samples_split=2,
            class_weight='balanced',
            splitter='random',
            max_features=4,
            random_state=42
        )

        decisionTreeClassifier.fit(x, y)
        logging.info('Decision Tree Model trained: Successfully')

        saveModel(decisionTreeClassifier, 'decisionTreeClassifier.sav')
        logging.info('Decision Tree Model Saved: Successfully')
        logging.info("====="*10)


    def randomForest(self):
        """
        Training Random Forest Model
        """
        logging.info('Training Random Forest Model')
        randomForest_df = self.boston.copy()

        x = randomForest_df.drop('MDEV', axis=1)
        y = randomForest_df['MDEV']
        randomForest_pre_process = ColumnTransformer(transformers=[
                                            (
                                                'drop_columns',
                                                'drop',
                                                [
                                                    'B'
                                                ]
                                            ),
                                            (
                                                'scale_data',
                                                MinMaxScaler(),
                                                list(x.columns))
                                        ])
        randomForestModel = RandomForestRegressor(
            ccp_alpha=0.001,
            criterion='mae',
            max_depth=834,
            min_samples_split=9,
            n_estimators=61,
            random_state=137
            )

        randomForestPipeline = Pipeline(steps=[
                                            (
                                                'pre_processing',
                                                randomForest_pre_process
                                            ),
                                            (
                                                'model',
                                                randomForestModel
                                            )
                                        ]) 

        randomForestPipeline.fit(x, y)
        logging.info('Random Forest Model Trained: Successfully')

        saveModel(randomForestPipeline, 'randomForestPipeline.sav')
        logging.info('Random Forest Model Saved: Successfully')
        logging.info("====="*10)


    def xgBoost(self):
        """
        Training XG Boost Model
        """
        logging.info('Training XG Boost Model')
        xgBoost_df = self.adult.copy()
        
        logging.info("Creating encoder for Custom Transformer")
        # Creating encoder for Categorical variables
        encoder = createEncoder(xgBoost_df.drop(['income'], axis = 1), xgBoost_df['income'])

        # Dropping the irrelevant observations
        xgBoost_df = xgBoost_df.iloc[np.where((xgBoost_df.age > 16) & ((xgBoost_df.capital_gain > 100) | (xgBoost_df.capital_loss > 100)) & (xgBoost_df.fnlwgt > 1) & (xgBoost_df.hours_per_week > 0))]

        X = xgBoost_df.drop(['income'], axis = 1)
        y = xgBoost_df['income']

        

        # Function Transformers
        transform = FunctionTransformer(adultTransformer)
        encode = FunctionTransformer(adultCategoricalEncoder, kw_args = {'encoder':encoder})

        # Scaling The features
        # [listing the columns which won't be dropped by transformer1 & transformer2]
        scale = ColumnTransformer(transformers=[
            (
                'scaler',
                StandardScaler(),
                [
                    'age',
                    'workclass',
                    'fnlwgt',
                    'education',
                    'marital_status',
                    'occupation',
                    'race',
                    'sex',
                    'capital_gain',
                    'hours_per_week',
                    'native_country'
                ]
            )
        ])

        # Building model with best parameters
        params = {
            'alpha': 0.01,
            'booster': 'dart',
            'eta': 0.0008,
            'gamma': 0.0012,
            'lambda': 0.01,
            'learning_rate': 0.11,
            'max_depth': 50,
            'n_estimators': 670,
            'objective': 'binary:logistic',
            'random_state': 0,
            'tree_method': 'auto'
        }

        xgBoostModel = XGBClassifier( **params)
        # Save XG Boost Model separately before passing it to Pipeline
        
        # Defining the Pipeline and fitting it
        xgBoostPipeline = Pipeline(steps=[
                                    (
                                        'transformer1',
                                        transform
                                    ),
                                    (
                                        'transformer2',
                                        encode
                                    ),
                                    (
                                        'scaler',
                                        scale
                                    ),
                                    (
                                        'model',
                                        xgBoostModel
                                    )
                                ],
        verbose=True)

        # X_transformed = xgBoostPreProcess.fit_transform(X)
        # logging.info('Built XG Boost Model Preprocessor : Successfully')
        
        # saveModel(xgBoostPreProcess, 'xgBoostPreProcess.sav')
        # logging.info('XG Boost Model Preprocessor Saved: Successfully')

        xgBoostPipeline.fit(X, y)
        logging.info('XGBoost Model Trained: Successfully')
        
        saveModel(xgBoostPipeline, 'xgBoostPipeline.sav')
        # xgbModel_filename = 'xgBoostModel.json'
        # filePath = create_path('models', xgbModel_filename)
        # xgBoostModel.save_model(filePath)
        logging.info('XGBoost Model Saved: Successfully')
        logging.info("====="*10)
    
    def train_models(self):
        models_dict = {
            'Linear Regression': self.linearRegression,
            'Logistic Regression': self.logisticRegression,
            'Decision Tree': self.decisionTree,
            'Random Forest': self.randomForest,
            'XG Boost': self.xgBoost
        }

        if self.lib == 'all':
            for i in models_dict.keys():
                logging.info(f"Training {i}")
                models_dict[i]()
        else:
            for i in self.lib:
                logging.info(f"Training {i}")
                models_dict[i]()


def main():
    logging_str = "[%(asctime)s: %(levelname)s: %(module)s] %(message)s"
    filePath = create_path('logs', 'training_logs.log')
    logging.basicConfig(filename = filePath, level=logging.INFO, format=logging_str)

    logging.info('====='*50)
    logging.info('Training Started')
    train = Training()
    try:
        train.train_models()
        logging.info('Training Completed')
        logging.info('====='*50)
    except Exception as e:
        logging.error(f'Error: {e}')
        logging.info('Training Failed')
        logging.info("====="*10)


main()