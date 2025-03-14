from sklearn.model_selection import KFold, RandomizedSearchCV
from sklearn.metrics import log_loss, accuracy_score, precision_score, recall_score, f1_score
from sklearn import pipeline
import pandas as pd
from skopt import BayesSearchCV


def validacaoCruzada(X_train: pd.DataFrame = None,
                     y_train: pd.Series = None,
                     model: pipeline.Pipeline = None, 
                     folds: int = 5) -> dict[str,float]:
    

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

    metrics['logloss'] = (logloss/folds)
    metrics['f1'] = (f1/folds)
    metrics['precision'] = (precision/folds)
    metrics['recall'] = (recall/folds)
    metrics['accuracy'] = (accuracy/folds)


    return metrics


def avaliarModelo(X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.Series, y_test: pd.Series, model: pipeline.Pipeline) -> tuple[pipeline.Pipeline, float, float, float, float, float]:
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)

    return model, log_loss(y_test, y_pred_proba), f1_score(y_test, y_pred), precision_score(y_test, y_pred), recall_score(y_test, y_pred), accuracy_score(y_test, y_pred)

#def otimizarModelo(X_train: pd.DataFrame, y_train: pd.Series, model: pipeline.Pipeline, params: dict[str, list], n_iter = int) -> tuple[pipeline.Pipeline, float]:
#    searcher = RandomizedSearchCV(estimator = model, param_distributions= params, cv = 5, scoring='neg_log_loss', n_iter = n_iter, error_score='raise', verbose = 3)
#    searcher.fit(X_train, y_train)
#    return searcher.best_estimator_, -searcher.best_score_


def otimizarModelo(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    model: pipeline.Pipeline,
    search_spaces: dict,
    n_iter: int = 50
) -> tuple[pipeline.Pipeline, float]:
    
    searcher = BayesSearchCV(
        estimator=model,
        search_spaces=search_spaces,
        n_iter=n_iter,
        cv=5,
        scoring='neg_log_loss',
        verbose=3,
        random_state=42,
        n_jobs=-1,
    )
    
    searcher.fit(X_train, y_train)
    return searcher.best_estimator_, -searcher.best_score_



def featureExtractor(df: pd.DataFrame) -> pd.DataFrame:

    df['coeficiente_rendimento'] = ((10*df['cgpa']) + df['ssc_marks'] + df['hsc_marks'])/3

    df['coeficiente_rendimento_por_personalidade'] =  df.groupby('personality_type')['coeficiente_rendimento'].transform('mean')

    df['testes_media'] = (df['aptitude_test_score'] + ( 10 * df['soft_skills_rating']))/2

    df['testes_media_por_personalidade'] =  df.groupby('personality_type')['testes_media'].transform('mean')

    df['experiencia_pratica'] = (  
    df['internships']  +   
    df['projects']  +  
    df['workshops_certifications'] 
)  
    
    df['diff_ssc_hsc'] = df['ssc_marks'] - df['hsc_marks']

    
    df['aptitude_ajustado'] = df['aptitude_test_score'] / df.groupby('personality_type')['aptitude_test_score'].transform('mean')  

   

    df['variancia_academica'] = df[['cgpa', 'ssc_marks', 'hsc_marks']].std(axis=1)  

    df['aplicacoes_por_experiencia'] = df['n_job_applications'] / (df['experiencia_pratica'] + 1)  


    df['softskills_x_personalidade'] = df['soft_skills_rating'] * df['personality_type'] 



    return df