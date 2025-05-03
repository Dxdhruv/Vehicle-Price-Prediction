from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import json
class ModelEvaluator:
    
    def __init__(self, model):
        self.model = model
        self.metrics = {}
    
    def evaluate(self, X_test, y_test):
        y_pred = self.model.predict(X_test)
        self.metrics['MAE'] = mean_absolute_error(y_test, y_pred)
        self.metrics['RMSE'] = mean_squared_error(y_test, y_pred, squared=False)
        self.metrics['R2_SCORE'] = r2_score(y_test, y_pred)
    
        return self.metrics

    def save_metrics(self, file_path="model_metrics.json"):
        with open(file_path, "w") as f:
            json.dump(self.metrics, f, indent=4)
        print(f"Model evaluation metrics saved to: {file_path}")
    