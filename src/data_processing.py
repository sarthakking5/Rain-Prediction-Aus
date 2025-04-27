import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
import os
from src.logger import get_logger
from src.custom_exception import CustomException




logger=get_logger(__name__)

class DataProcessing:
    def __init__(self,input_path,output_path):
        self.input_path=input_path
        self.output_path=output_path
        self.df=None

        os.makedirs(self.output_path,exist_ok=True)
        logger.info("Data Processing Initialized...")

    def load_data(self):
        try:
            self.df=pd.read_csv(self.input_path)
            logger.info("Data loaded sucessfully")
        except Exception as e:
            logger.error(f"Error while loading data {e}")
            raise CustomException("Failed to load data")
    
    def preprocess(self):
        self.numerical=None
        self.categoical=None
        try:
            self.categorical=[]
            self.numerical=[]

            for col in self.df.columns:
                if self.df[col].dtype=='object':
                   self. categorical.append(col)
                else:
                    self.numerical.append(col)

            self.df['Date']=pd.to_datetime(self.df['Date'])
            self.df['Year']=self.df['Date'].dt.year
            self.df['Month']=self.df['Date'].dt.month
            self.df['Day']=self.df['Date'].dt.day

            self.df.drop("Date",axis=1,inplace=True)

        except Exception as e:
            logger.error(f"Error while preprocess data {e}")
            raise CustomException("Failed to preprocess data",e)

    def label_encoder(self):
        try:
             self.df = self.df.dropna(subset=["RainTomorrow"])
             le=LabelEncoder()
             self.df["RainTomorrow"]=le.fit_transform(self.df['RainTomorrow'])
             logger.info("Target Variable Encoded...")
        except Exception as e:
            logger.error(f"Error while encoding Target {e}")
            raise CustomException("Failed to encode target",e)

    def create_feature_eng_pipeline(self):
        self.feat_eng_pip=None
        try:
            self.categoical.remove("RainTomorrow")
            num_pip=Pipeline([
                ('imputer',SimpleImputer(strategy="mean"))
                               ])
            cat_pip=Pipeline([
                ('imputer',SimpleImputer(strategy='most_frequent')),
                ('onehot',OneHotEncoder)
            ])
            self.feat_eng_pip=ColumnTransformer([
                ("num",num_pip,self.numerical),
                ("cat",cat_pip,self.categoical)
            ])
            logger.info("Feature Engineering Pipeline Created...")
        except Exception as e:
            logger.error(f"Error while create feature pipeline{e}")
            raise CustomException("Failed to create feature pipeline",e)
        
    def split_data(self):
            try:
                X=self.df.drop("RainTomorrow",axis=1)
                y=self.df["RainTomorrow"]
                X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
                logger.info(f"X_train columns: {list(X_train.columns)}")
                joblib.dump(X_train,os.path.join(self.output_path,"X_train.pkl"))
                joblib.dump(X_test,os.path.join(self.output_path,"X_test.pkl"))
                joblib.dump(y_train,os.path.join(self.output_path,"y_train.pkl"))
                joblib.dump(y_test,os.path.join(self.output_path,"y_test.pkl"))

                logger.info("Splitted and saved sucessfully...")
            except Exception as e:
                logger.error(f"Error while splitting Data{e}")
                raise CustomException("Splitting Data failed",e)
    def run(self):
        self.load_data()
        self.preprocess()
        self.label_encoder()
        self.split_data()
            
        logger.info("Data Processing Completed...")

if __name__=="__main__":
    processor=DataProcessing("artifacts/raw/data.csv","artifacts/processed")
    processor.run()


            
        
    



