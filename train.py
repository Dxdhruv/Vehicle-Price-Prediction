import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from data_preprocessing import DataPreprocessor
from evaluate import ModelEvaluator

class ModelTrainer:
    def __init__(self, data_path):
        self.data_path = data_path
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.preprocessor = DataPreprocessor()
        self.evaluate = ModelEvaluator()
        
    def load_path(self):
        df = pd.read_csv(self.data_path)
        X = df.drop(columns=['price'])
        y = df['price']
        return train_test_split(X, y, test_size=0.2, random_state=42)
    
    def train_model(self):
        X_train, X_test, y_train, y_test = self.load_path()
        
        X_train_transformed = self.preprocessor.fit_transform(X_train, y_train)
        X_test_transformed = self.preprocessor.transform(X_test)
        
        self.model.fit(X_train_transformed, y_train)
        
        y_pred = self.model.predict(X_test_transformed)
        self.evaluate.evaluate(y_test, y_pred)
        
        with open('model.pkl') as file:
            pickle.dump(self.model, file)
        print("Model Saved")
        
if __name__ == "__main__":
    trainer = ModelTrainer(r"C:\Users\Dell\Downloads\Data Science Stuff\Machine Learning\Internship Tasks\Data\car_data.csv")
    trainer.train_model()