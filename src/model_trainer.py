
# import libraries
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, f1_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_score, RandomizedSearchCV


class FraudModelTrainer:
    """
    Generic fraud model trainer
    Works for credit card & e-commerce fraud datasets
    """

    def __init__(self, random_state=42):
        # Set random state for reproducibility
        self.random_state = random_state
        
        # Dictionary to store trained models
        self.models = {}
        
        # List to store evaluation results
        self.results = []

    # ------------------------------
    # 1. Train baseline model
    # ------------------------------
    def train_logistic_regression(self, x_train, y_train):
        """
        Train a Logistic Regression model with class balancing
        """
        try:
            lr = LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                random_state=self.random_state
            )  # define logistic regression model

            lr.fit(x_train, y_train)  # train the model

            # Store trained model
            self.models["Logistic Regression"] = lr
            return lr

        except Exception as e:
            raise RuntimeError(f"Error training Logistic Regression: {e}")

    # -------------------------------------
    # 2. Train ensemble model
    # -------------------------------------
    def train_random_forest(self, x_train, y_train):
        """
        Train a baseline Random Forest classifier
        """
        try:
            rf = RandomForestClassifier(
                n_estimators=200,
                max_depth=10,
                n_jobs=-1,
                random_state=self.random_state
            )  # define the model

            rf.fit(x_train, y_train)  # train the model

            # Store trained model
            self.models["Random Forest"] = rf
            return rf

        except Exception as e:
            raise RuntimeError(f"Error training Random Forest: {e}")

    # ---------------------------------------------
    # 3. Hyperparameter Tuning
    # ---------------------------------------------
    def tune_random_forest(self, x_train, y_train, n_iter=20):
        """
        Tune Random Forest hyperparameters using RandomizedSearchCV
        """
        try:
            rf = RandomForestClassifier(
                random_state=self.random_state,
                n_jobs=-1
            )

            # Hyperparameter search space
            param_dist = {
                "n_estimators": [300, 500],
                "max_depth": [5, 20]
            }

            search = RandomizedSearchCV(
                estimator=rf,
                param_distributions=param_dist,
                n_iter=n_iter,
                scoring="average_precision",
                cv=2,
                verbose=1,
                n_jobs=-1,
                random_state=self.random_state
            )

            search.fit(x_train, y_train)  # run hyperparameter search

            best_model = search.best_estimator_

            # Store tuned model
            self.models["Random Forest[Tuned]"] = best_model

            return best_model, search.best_params_, search.best_score_

        except Exception as e:
            raise RuntimeError(f"Error tuning Random Forest: {e}")

    # ------------------------------------------------------------
    # 4. Evaluate model
    # ------------------------------------------------------------
    def evaluate(self, model, x_test, y_test, model_name):
        """
        Evaluate model using AUC-PR, F1-score, and confusion matrix
        """
        try:
            # Class predictions
            y_pred = model.predict(x_test)

            # Probability predictions (positive class)
            y_proba = model.predict_proba(x_test)[:, 1]


            auc_pr = average_precision_score(y_test, y_proba)
            f1 = f1_score(y_test, y_pred)
            cm = confusion_matrix(y_test, y_pred)

            # Save evaluation results
            self.results.append({
                "Model": model_name,
                "AUC_PR": auc_pr,
                "F1-Score": f1
            })

            return {
                "AUC_PR": auc_pr,
                "F1-Score": f1,
                "Confusion Matrix": cm
            }

        except Exception as e:
            raise RuntimeError(f"Error evaluating model {model_name}: {e}")

    # ---------------------------------------------------------
    # 5. Cross-Validation
    # ---------------------------------------------------------
    def cross_validation(self, model, X, Y, cv=5):
        """
        Perform stratified cross-validation using AUC-PR
        """
        try:
            skf = StratifiedKFold(
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

            return scores.mean(), scores.std()

        except Exception as e:
            raise RuntimeError(f"Error during cross-validation: {e}")

    # ---------------------------------------
    # 6. Compare Models
    # ---------------------------------------
    def get_result_table(self):
        """
        Return evaluation results as a pandas DataFrame
        """
        try:
            return pd.DataFrame(self.results)

        except Exception as e:
            raise RuntimeError(f"Error creating result table: {e}")