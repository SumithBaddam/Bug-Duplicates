import pandas as pd
import xgboost as xgb
import numpy as np
from sklearn.cross_validation import *
from sklearn.grid_search import GridSearchCV
from xgboost import XGBClassifier
from xgboost import plot_tree
import pylab as pl
from sklearn.metrics import roc_curve, auc
import pymongo
import sys
sys.path.insert(0, "/data/ingestion/")
from Utils import *
#Setting up the config.ini file parameters
settings = configparser.ConfigParser()
settings.read('/data/ingestion/config.ini')
import pickle 

#Ensemble model for training
def stacking_train(df, cluster):
    pred_columns=['actual','pred_cat','pred_text']
    stacking_df=df[pred_columns]
    train_x = stacking_df.drop('actual', axis=1).as_matrix()
    train_y = stacking_df['actual'].as_matrix()
    #X_resampled, y_resampled = SMOTE(kind='borderline1').fit_sample(train_x, train_y) #SMOTE
    model = XGBClassifier(objective='binary:logistic')
    #model.fit(X_resampled, y_resampled)
    model.fit(train_x, train_y)
    print('fitting done')
    filename = '/data/ingestion/bugDuplicates/bugDup_ensemble_3.txt'
    if os.path.exists(filename):
        os.remove(filename)
        print("File Removed!")
    with open(filename, 'wb') as handle:
        pickle.dump(model, handle, protocol=pickle.HIGHEST_PROTOCOL)    
    y_pred = model.predict_proba(train_x)[:,1]
    return y_pred

#Ensemble model for testing
def stacking_test(test_df, cluster):
    pred_columns=['pred_cat','pred_text']
    stacking_df=test_df[pred_columns]
    filename = '/data/ingestion/bugDuplicates/bugDup_ensemble.txt'
    with open(filename, 'rb') as f:
        model= pickle.load(f)   
    test_x = stacking_df.as_matrix()
    y_pred = model.predict_proba(test_x)[:,1]  
    return y_pred


def main():
collection = db['BugDupsTestSet_436_1452_results']
cursor = collection.find({})
df =  pd.DataFrame(list(cursor))
a = stacking_train(df, 1)
b = stacking_test(df, 1)


key = "csap_prod_database"
db = get_db(settings, key)
collection = db['BugDupsTestSet_436_1452_results']
cursor = collection.find({})
df =  pd.DataFrame(list(cursor))
fpr, tpr, thresholds = roc_curve(df['actual'], df['pred_ensemble'])
roc_auc = auc(fpr, tpr)
i = np.arange(len(tpr))
roc = pd.DataFrame({'fpr' : pd.Series(fpr, index=i),'tpr' : pd.Series(tpr, index = i), '1-fpr' : pd.Series(1-fpr, index = i), 'tf' : pd.Series(tpr - (1-fpr), index = i), 'thresholds' : pd.Series(thresholds, index = i)})
r = roc.ix[(roc.tf-0).abs().argsort()[:1]]
cut_off = list(r['thresholds'])[0]
