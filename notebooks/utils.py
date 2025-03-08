from sklearn.model_selection import KFold
from sklearn.metrics import log_loss, accuracy_score, precision_score, recall_score, f1_score
from sklearn import pipeline
import pandas as pd

def validacaoCruzada(X_train: pd.DataFrame = None,
                     y_train: pd.Series = None,
                     model: pipeline.Pipeline = None, 
                     folds: int = 5):
    

    kf = KFold(n_splits=folds, shuffle=True)
    metrics = {}

    logloss = f1 = precision = recall = accuracy = 0
    
    for i, (train_index, test_index) in enumerate(kf.split(X_train)):

        X_train_fold, X_test_fold = X_train.iloc[train_index], X_train.iloc[test_index]
        y_train_fold, y_test_fold = y_train.iloc[train_index], y_train.iloc[test_index]

        model.fit(X_train_fold, y_train_fold)
        y_pred = model.predict(X_test_fold)
        y_pred_proba = model.predict_proba(X_test_fold)


        logloss += log_loss(y_test_fold, y_pred_proba)
        f1 += f1_score(y_test_fold, y_pred)
        precision += precision_score(y_test_fold, y_pred)
        recall += recall_score(y_test_fold, y_pred)
        accuracy += accuracy_score(y_test_fold, y_pred)

    metrics['logloss']= (logloss/folds)
    metrics['f1']= (f1/folds)
    metrics['precision']= (precision/folds)
    metrics['recall']= (recall/folds)
    metrics['accuracy']= (accuracy/folds)


    return metrics


def treinarModelo(X_train, X_test,y_train, y_test, model):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)

    return log_loss(y_test, y_pred_proba), f1_score(y_test, y_pred), precision_score(y_test, y_pred), recall_score(y_test, y_pred), accuracy_score(y_test, y_pred), model