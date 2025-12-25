# imoprt libraries
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score,f1_score,confusion_matrix
from sklearn.model_selection import StratifiedKFold,cross_val_score,RandomizedSearchCV
 


class FraudModelTrainer:
    """
    Generic fraud model trainer
    works for credit card & e-commerce fraud datasets
    """
    def __init__(self,random_state=42):
        self.random_state=random_state
        self.models={}
        self.results=[]
    #------------------------------
    # 1.Train baseline model
    #------------------------------
    def train_logistic_regression(self,x_train,y_train):
        lr=LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=self.random_state
        )# define logestic regression model
        lr.fit(x_train,y_train) # train the test set
        self.models["Logostic Regression"]=lr
        return lr
    #-------------------------------------
    # 2.Train ensemble model
    #-------------------------------------
    def train_random_forest(self,x_train,y_train):
        rf=RandomForestClassifier(n_estimators=200,
                                  max_depth=10,
                                  n_jobs=-1,
                                  random_state=self.random_state) # define the model
        rf.fit(x_train,y_train) # train on the test set
        self.models["Random Forest"]=rf
        return rf
    #---------------------------------------------
    # 3. Hyper Parameter Tunning 
    #--------------------------------------------
    def tune_random_forest(self,x_train,y_train,n_iter=20):
        rf=RandomForestClassifier(random_state=self.random_state,n_jobs=-1)
        param_dist = {
            "n_estimators": [300, 500],
            "max_depth": [5,20]
        } # define different prarmeters of random forest that can be tuned 
        
       
        search=RandomizedSearchCV(
            estimator=rf,
            param_distributions=param_dist,
            n_iter=n_iter,
            scoring="average_precision",
            cv=2,
            verbose=1,
            n_jobs=-1,
            random_state=self.random_state
        )
        search.fit(x_train,y_train) # apply on the training set
        best_model=search.best_estimator_
        self.models["Random Forest[Tuned]"]=best_model
        return best_model, search.best_params_, search.best_score_
    #------------------------------------------------------------
    # 4,Evaluate model
    #--------------------------------------------------------------
    def evaluate(self,model,x_test,y_test,model_name):
        y_pred=model.predict(x_test) # predict the model on test set 
        y_proba=model.predict_proba(x_test)[:,1] # predict the probabilities
        auc_pr=average_precision_score(y_test,y_proba)
        f1=f1_score(y_test,y_pred)
        cm=confusion_matrix(y_test,y_pred)
        self.results.append({
            "Model":model_name,
            "AUC_PR":auc_pr,
            "F1-Score":f1
        })
        return {
            "AUC_PR":auc_pr,
            "F1-SCore":f1,
            "Confusion Matrix":cm
        }
    #---------------------------------------------------------
    # 5,Cross-Validation
    #---------------------------------------------------------
    def cross_validation(self,model,X,Y,cv=5):
        skf=StratifiedKFold(
            n_splits=cv,
            shuffle=True,
            random_state=self.random_state
        )
        scores = cross_val_score(
            model,
            X,
            Y,
            scoring="average_precision",
            cv=skf,
            n_jobs=-1
        )
        return scores.mean(),scores.std()
    #---------------------------------------
     # 6,Compare Model
    #---------------------------------------
    def get_result_table(self):
        return pd.DataFrame(self.results) # get the result of each model as Dataframe



        
      
