import os
import joblib
import xgboost as xgb
from src.logger import get_logger
from src.custom_exception import CustomException
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,log_loss,classification_report,confusion_matrix
logger=get_logger(__name__)

class ModelTraining:
    def __init__(self,input_path,output_path):
        self.input_path=input_path
        self.output_path=output_path
        self.model=xgb.XGBClassifier(eval_metric='logloss')
        self.X_train=None
        self.X_test=None
        self.y_train=None
        self.y_test=None
        os.makedirs(self.output_path,exist_ok=True)
        logger.info("Model Training Initailized...")
   
    def get_feature_types(self, df):
        categorical = [col for col in df.columns if df[col].dtype == 'object']
        numerical = [col for col in df.columns if df[col].dtype != 'object']
        return categorical, numerical
    
    def load_data(self):
        try:
             self.X_train=joblib.load(os.path.join(self.input_path,"X_train.pkl"))
             self.X_test=joblib.load(os.path.join(self.input_path,"X_test.pkl"))
             self.y_train=joblib.load(os.path.join(self.input_path,"y_train.pkl"))
             self.y_test=joblib.load(os.path.join(self.input_path,"y_test.pkl"))
             self.categorical, self.numerical = self.get_feature_types(self.X_train)
             logger.info("Processed Data loaded sucessfully...")
        except Exception as e:
            logger.error("Data Loading failed {e}")
            raise CustomException("Failed to load data",e)
    
    def create_feature_eng_pipeline(self):
        self.feat_eng_pip=None
        try:
            num_pip=Pipeline([
                ('imputer',SimpleImputer(strategy="mean"))
                               ])
            cat_pip=Pipeline([
                ('imputer',SimpleImputer(strategy='most_frequent')),
                ('onehot',OneHotEncoder(handle_unknown='ignore'))
            ])
            self.feat_eng_pip=ColumnTransformer([
                ("num",num_pip,self.numerical),
                ("cat",cat_pip,self.categorical)
            ])
            logger.info("Feature Engineering Pipeline Created...")
        except Exception as e:
            logger.error(f"Error while create feature pipeline{e}")
            raise CustomException("Failed to create feature pipeline",e)
        
    def train_model(self):
        self.model_pip=None
        try:
            self.model_pip=Pipeline([
            ('feat_eng',self.feat_eng_pip),
            ('model',self.model)
            ])
            self.model_pip.fit(self.X_train,self.y_train)

            joblib.dump(self.model_pip,os.path.join(self.output_path,'model.pkl'))
            logger.info("Training and saving of model done...")
        except Exception as e:
            logger.error(f"Error while Training Model {e}")
            raise CustomException("Failed to train model")
    def eval_model(self):
        try:
            training_score=self.model_pip.score(self.X_train,self.y_train)
            logger.info(f"Training model score :{training_score}")

            y_pred= self.model_pip.predict(self.X_test)

            accuracy = accuracy_score(self.y_test,y_pred)
            precision = precision_score(self.y_test,y_pred,average="weighted")
            recall = recall_score(self.y_test,y_pred,average="weighted")
            f1 = f1_score(self.y_test,y_pred,average="weighted")
            logger.info(f"Accuracy : {accuracy} ; Precision : {precision} ; Recall : {recall}  : F1-Score : {f1}")

            logger.info("Model evaluation done..")
        
        except Exception as e:
            logger.error(f"Error while evaluating model {e}")
            raise CustomException("Failed to evaluate model ", e)
        
    def run(self):
        self.load_data()
        self.create_feature_eng_pipeline()
        self.train_model()
        self.eval_model()

        logger.info("Model training and Evaluation Done...")

if __name__ == "__main__":
    trainer=ModelTraining("artifacts/processed","artifacts/models")
    trainer.run()





    