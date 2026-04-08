import os
from typing import Sequence
from joblib import Memory
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, cross_val_predict
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, recall_score, f1_score, roc_auc_score, cohen_kappa_score, classification_report, confusion_matrix, make_scorer

# ---------- # ---------- # ---------- # ---------- # ---------- # ---------- #
class forestPipeline(Pipeline):

    def __init__(self, input_data):

        # Define Input Data
        self.input_data = input_data

        # Determine numerical and categorical columns for encoding
        num_cols = self.input_data.select_dtypes(exclude=['object','str']).columns.tolist()
        cat_cols = self.input_data.select_dtypes(include=['object','str']).columns.tolist()

        # Define Encoder
        encoder = ColumnTransformer(
            [
                ('num', StandardScaler(), num_cols),
                ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)
            ],
            n_jobs=-1)
        

        # Create Pipeline w/ encoder
        super().__init__([
                ('encoder', encoder),
                ('rf', RandomForestClassifier(random_state=RAND_STATE, bootstrap=True))
            ])

# ---------- # ---------- # ---------- # ---------- # ---------- # ---------- #
def trainModel_1(input_1_train, y_A_train) -> forestPipeline:
    '''
    From GridSearchCV:

    Model 1 GridSearch Parameters:
        rf__criterion : gini
        rf__max_depth : None
        rf__max_features : sqrt
        rf__min_samples_leaf : 2
        rf__min_samples_split : 2
        rf__n_estimators : 50
    '''

    # Define Parameters
    model_1_params = {
        "rf__criterion" : 'gini',
        "rf__max_depth" : None,
        "rf__max_features" : 'sqrt',
        "rf__min_samples_leaf" : 2,
        "rf__min_samples_split" : 2,
        "rf__n_estimators" : 50
    }

    # Create Pipeline
    forest_1 = forestPipeline(input_1_train)
    forest_1.set_params(**model_1_params)

    # Fit Model
    model_1 = forest_1.fit(input_1_train, y_A_train)

    return model_1

# ---------- # ---------- # ---------- # ---------- # ---------- # ---------- #
def trainModel_2(input_2_train, y_B_train) -> forestPipeline:
    '''
    From GridSearchCV:

    Model 2 GridSearch Parameters:
        rf__criterion : entropy
        rf__max_depth : None
        rf__max_features : None
        rf__min_samples_leaf : 2
        rf__min_samples_split : 16
        rf__n_estimators : 50
    '''

    # Define Parameters
    model_2_params = {
        "rf__criterion" : 'entropy',
        "rf__max_depth" : None,
        "rf__max_features" : None,
        "rf__min_samples_leaf" : 2,
        "rf__min_samples_split" : 16,
        "rf__n_estimators" : 50
    }

    # Create Pipeline
    forest_2 = forestPipeline(input_2_train)
    forest_2.set_params(**model_2_params)

    # Fit Model
    model_2 = forest_2.fit(input_2_train, y_B_train)

    return model_2

# ---------- # ---------- # ---------- # ---------- # ---------- # ---------- #

def trainModel_3(A_pred, B_pred, y_C_train) -> forestPipeline:
    '''
    From GridSearchCV:

    Model 3 GridSearch Parameters:
        rf__criterion : gini
        rf__max_depth : None
        rf__max_features : log2
        rf__min_samples_leaf : 8
        rf__min_samples_split : 2
        rf__n_estimators : 50
    '''

    # Define Parameters
    model_3_params = {
        "rf__criterion" : 'gini',
        "rf__max_depth" : None,
        "rf__max_features" : 'log2',
        "rf__min_samples_leaf" : 8,
        "rf__min_samples_split" : 2,
        "rf__n_estimators" : 50
    }

    # Define Model 3 Input
    input_3_train = pd.DataFrame({
        "Supervisory Conditions" : A_pred,
        "Operator Conditions" : B_pred
    })

    # Create Pipeline
    forest_3 = forestPipeline(input_3_train)
    forest_3.set_params(**model_3_params)

    # Fit Model
    model_3 = forest_3.fit(input_3_train,y_C_train)

    return model_3

# ---------- # ---------- # ---------- # ---------- # ---------- # ---------- #
def trainPipeline() -> tuple:

    # Read Dataset
    synthDataPath = "data/Simulated_Dataset.xlsx"
    synthData = pd.read_excel(synthDataPath)

    global RAND_STATE
    RAND_STATE = 1024

    # Split Dataset
    TEST_SPLIT = 0.2
    train_set, test_set = train_test_split(synthData,test_size=TEST_SPLIT, train_size=(1-TEST_SPLIT), random_state=RAND_STATE)

    # Training Set
    envrCond_train = train_set[["Light Conditions","Basic Meteorological Conditions","Wind Conditions (kt)","Temperature (C)"]]
    input_1_train = pd.concat((train_set["Employment Change vs Prior Period (%)"],envrCond_train),axis=1)
    persCond_train = train_set["Personnel Conditions"]

    y_A_train = train_set["Supervisory Conditions"]
    y_B_train = train_set["Operator Conditions"]
    y_C_train = train_set["Unsafe Conditions"]

    #-----#

    # Train Model 1 -> A prediction
    model_1 = trainModel_1(input_1_train=input_1_train,y_A_train=y_A_train)

    # Out of Fold Predictions for Model 2 Input
    A_pred_oof = cross_val_predict(model_1, input_1_train, y_A_train, cv=5,method='predict')

    #-----#

    # Define Model 2 Inputs
    input_2_train = pd.concat((envrCond_train,persCond_train),axis=1)
    input_2_train['Supervisory Conditions'] = A_pred_oof

    # Train Model 2 -> B prediction
    model_2 = trainModel_2(input_2_train=input_2_train, y_B_train=y_B_train)
    
    # Out of Fold Predictions for Model 3 Input
    B_pred_oof = cross_val_predict(model_2, input_2_train, y_B_train, cv=5,method='predict')

    #-----#

    # Train Model 3 -> C Prediction
    model_3 = trainModel_3(A_pred=A_pred_oof, B_pred=B_pred_oof, y_C_train=y_C_train)

    return model_1, model_2, model_3, test_set
