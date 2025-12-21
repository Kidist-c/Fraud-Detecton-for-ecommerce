import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline


class CreditCardDataTransformer:
    """
    Scaling + class imbalance handling for credit card fraud data
    """

    def __init__(
        self,
        amount_col='Amount',
        time_col='Time',
        target_col='Class',
        sampler=SMOTE(random_state=42)
    ):
        self.amount_col = amount_col
        self.time_col = time_col
        self.target_col = target_col
        self.sampler = sampler

        self.preprocessor = None
        self.pipeline = None

    def build_preprocessor(self):
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('scale', StandardScaler(), [self.amount_col, self.time_col])
            ],
            remainder='passthrough'  # keep V1–V28
        )

    def build_pipeline(self):
        self.pipeline = ImbPipeline(
            steps=[
                ('preprocessing', self.preprocessor),
                ('smote', self.sampler)
            ]
        )

    def fit_resample(self, df: pd.DataFrame):
        X = df.drop(columns=[self.target_col])
        y = df[self.target_col]

        self.build_preprocessor()
        self.build_pipeline()

        X_resampled, y_resampled = self.pipeline.fit_resample(X, y)
        return X_resampled, y_resampled

    def get_class_distribution(self, y):
        return y.value_counts(normalize=True) * 100
