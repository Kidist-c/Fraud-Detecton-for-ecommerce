# import libraries
import pandas as pd
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline



class FraudDataTransformer:
    """
    Handles:
      Scaling numerical features
      Encoding Catagorical features
      Class Imbalance Using SMOTE(train only)
    """
    def __init__(self,
                 numerical_features:list,
                 categorical_features:list,
                 target_col:str='class',
                 scaler=StandardScaler(),
                 sampler=SMOTE(random_state=42)
                 ):
        self.numerical_features=numerical_features
        self.categorical_features=categorical_features
        self.target_col=target_col
        self.scaler=scaler
        self.sampler=sampler
        self.preprocessor=None
        self.pipeline=None
    def build_preprocessor(self):
        """
        Build column-wise pipeline
        """
        self.preprocessor=ColumnTransformer(
            transformers=[('num',self.scaler,self.numerical_features),
                          ("cat", OneHotEncoder(handle_unknown="ignore"), self.categorical_features)])
        
    def build_pipeline(self):
        """
        Build full pipeline including SMOTE
        """
        self.pipeline=ImbPipeline(
            steps=[
                ("preprocessing",self.preprocessor),
                ("smote",self.sampler)
            ]
        )
    def fit_resample(self,df:pd.DataFrame):
        """
        Fit Preprocessing + Fit preprocessing + resampling on TRAINING data only
        """
        X = df.drop(columns=[self.target_col])
        y=df[self.target_col]

        self.build_preprocessor()
        self.build_pipeline()
        X_resampled, y_resampled = self.pipeline.fit_resample(X, y)
        return X_resampled, y_resampled
    def get_class_distribution(self, Y):
        """
        Return class distribution as percentages
        """
        return Y.value_counts(normalize=True) * 100

        