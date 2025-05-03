# import pandas as pd
# import numpy as np
# from sklearn.compose import ColumnTransformer
# from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
# from category_encoders import TargetEncoder
# from sklearn.model_selection import train_test_split
# import pickle

# class DataPreprocessor:
#     def __init__(self):
#         self.target_encoder = TargetEncoder()
#         self.column_transformer = ColumnTransformer(transformers=[
#             ('condition_ord', OrdinalEncoder(categories=[['salvage','fair','good','like new','excellent','new']]), ['condition']),
#             ('cylinders_ord', OrdinalEncoder(categories=[['other','3 cylinders','4 cylinders','5 cylinders','6 cylinders','8 cylinders','10 cylinders', '12 cylinders']]), ['cylinders']),
#             ('fuel_ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False), ['fuel']),
#             ('title_ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False), ['title_status']),
#             ('transmission_ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False), ['transmission']),
#             ('drive_ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False), ['drive']),
#             ('selling_category', OneHotEncoder(handle_unknown='ignore', sparse_output=False), ['selling_category']),
#             ('paint_color_category', OneHotEncoder(handle_unknown='ignore', sparse_output=False), ['paint_color']),
#             ('vehicle_type_category', OneHotEncoder(handle_unknown='ignore', sparse_output=False), ['vehicle_type_category']),
#             ('num_scaler', StandardScaler(), ['year', 'odometer'])
#         ], remainder="passthrough")

#     def categorize_sales(self, manufacturer):
#         sales_dict = {
#             "ford": 47679, "chevrolet": 38150, "toyota": 25088, "honda": 16068,
#             "nissan": 13906, "jeep": 13298, "bmw": 11433, "gmc": 10852, "ram": 10642,
#             "dodge": 9353, "mercedes-benz": 8111, "hyundai": 7727, "volkswagen": 7535,
#             "subaru": 7247, "lexus": 6553, "kia": 6361, "audi": 5969, "cadillac": 5221,
#             "acura": 4936, "chrysler": 4520, "buick": 4382, "mazda": 4225, "infiniti": 3923,
#             "lincoln": 3448, "volvo": 2695, "mitsubishi": 2649, "mini": 2019, "pontiac": 1773,
#             "jaguar": 1641, "rover": 1340, "mercury": 915, "porsche": 876, "saturn": 834,
#             "alfa-romeo": 815, "fiat": 691, "tesla": 605, "harley-davidson": 110,
#             "datsun": 53, "aston-martin": 9, "land rover": 7, "ferrari": 6, "morgan": 3
#         }
#         sales = sales_dict.get(manufacturer, 0)
#         return "High Selling" if sales > 25000 else "Medium Selling" if sales > 10000 else "Low Selling"

#     def categorize_vehicle_type(self, vehicle_type):
#         sales_dict = {
#             "sedan": 80903, "SUV": 68178, "pickup": 38134, "truck": 25365,
#             "other": 18335, "coupe": 17401, "hatchback": 16154, "wagon": 9591,
#             "van": 7309, "convertible": 7270, "mini-van": 5028
#         }
#         sales = sales_dict.get(vehicle_type, 0)
#         return "Popular" if sales > 25000 else "Medium Popular" if sales > 10000 else "Less Popular"

#     def categorize_paint_color(self, paint_color):
#         sales_dict = {
#             "white": 74515, "black": 62522, "silver": 43751, "blue": 32260, "red": 31073,
#             "grey": 24774, "green": 7186, "brown": 7000, "custom": 6014, "orange": 2006,
#             "yellow": 1915, "purple": 652
#         }
#         sales = sales_dict.get(paint_color, 0)
#         return "Popular color" if sales > 20000 else "Less Popular color"

#     def fit(self, X_train, y_train):
#         X_train = X_train.copy()
#         X_train['selling_category'] = X_train['manufacturer'].apply(self.categorize_sales)
#         X_train['vehicle_type_category'] = X_train['type'].apply(self.categorize_vehicle_type)
#         X_train['paint_color_category'] = X_train['paint_color'].apply(self.categorize_paint_color)

#         self.target_encoder.fit(X_train[['model']], y_train)
#         self.column_transformer.fit(X_train)

#     def transform(self, X):
#         X = X.copy()
#         X['selling_category'] = X['manufacturer'].apply(self.categorize_sales)
#         X['vehicle_type_category'] = X['type'].apply(self.categorize_vehicle_type)
#         X['paint_color_category'] = X['paint_color'].apply(self.categorize_paint_color) 

#         X[['model']] = self.target_encoder.transform(X[['model']])
#         transformed_data = self.column_transformer.transform(X)
#         return pd.DataFrame(transformed_data, columns=self.column_transformer.get_feature_names_out())
    
# def save_processor():
#     data = pd.read_csv(r"C:\Users\Dell\Downloads\Data Science Stuff\Machine Learning\Internship Tasks\Data\car_data.csv")
#     X = data.drop(columns=['price'])
#     y = data['price']
        
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#     X_train = X_train.dropna()
#     X_test = X_test.dropna()
        
#     preprocessor = DataPreprocessor()
#     preprocessor.fit(X_train, y_train)
        
#     with open('preprocessor.pkl','wb') as f:
#         pickle.dump(preprocessor, f)
        
#     print("Preprocessor Saved Successfully")

# if __name__ == "__main__":
#     save_processor()

import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from category_encoders import TargetEncoder
from sklearn.model_selection import train_test_split
import pickle

class DataPreprocessor:
    def __init__(self):
        self.target_encoder = TargetEncoder()
        self.column_transformer = ColumnTransformer(transformers=[
            ('condition_ord', OrdinalEncoder(categories=[['salvage','fair','good','like new','excellent','new']]), ['condition']),
            ('cylinders_ord', OrdinalEncoder(categories=[['other','3 cylinders','4 cylinders','5 cylinders','6 cylinders','8 cylinders','10 cylinders', '12 cylinders']]), ['cylinders']),
            ('fuel_ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False), ['fuel']),
            ('title_ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False), ['title_status']),
            ('transmission_ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False), ['transmission']),
            ('drive_ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False), ['drive']),
            ('selling_category', OneHotEncoder(handle_unknown='ignore', sparse_output=False), ['selling_category']),
            ('paint_color_category', OneHotEncoder(handle_unknown='ignore', sparse_output=False), ['paint_color']),
            ('vehicle_type_category', OneHotEncoder(handle_unknown='ignore', sparse_output=False), ['vehicle_type_category']),
            ('num_scaler', StandardScaler(), ['year', 'odometer'])
        ], remainder="passthrough")

    def categorize_sales(self, manufacturer):
        sales_dict = {
            "ford": 47679, "chevrolet": 38150, "toyota": 25088, "honda": 16068,
            "nissan": 13906, "jeep": 13298, "bmw": 11433, "gmc": 10852, "ram": 10642,
            "dodge": 9353, "mercedes-benz": 8111, "hyundai": 7727, "volkswagen": 7535,
            "subaru": 7247, "lexus": 6553, "kia": 6361, "audi": 5969, "cadillac": 5221,
            "acura": 4936, "chrysler": 4520, "buick": 4382, "mazda": 4225, "infiniti": 3923,
            "lincoln": 3448, "volvo": 2695, "mitsubishi": 2649, "mini": 2019, "pontiac": 1773,
            "jaguar": 1641, "rover": 1340, "mercury": 915, "porsche": 876, "saturn": 834,
            "alfa-romeo": 815, "fiat": 691, "tesla": 605, "harley-davidson": 110,
            "datsun": 53, "aston-martin": 9, "land rover": 7, "ferrari": 6, "morgan": 3
        }
        sales = sales_dict.get(manufacturer, 0)
        return "High Selling" if sales > 25000 else "Medium Selling" if sales > 10000 else "Low Selling"

    def categorize_vehicle_type(self, vehicle_type):
        sales_dict = {
            "sedan": 80903, "SUV": 68178, "pickup": 38134, "truck": 25365,
            "other": 18335, "coupe": 17401, "hatchback": 16154, "wagon": 9591,
            "van": 7309, "convertible": 7270, "mini-van": 5028
        }
        sales = sales_dict.get(vehicle_type, 0)
        return "Popular" if sales > 25000 else "Medium Popular" if sales > 10000 else "Less Popular"

    def categorize_paint_color(self, paint_color):
        sales_dict = {
            "white": 74515, "black": 62522, "silver": 43751, "blue": 32260, "red": 31073,
            "grey": 24774, "green": 7186, "brown": 7000, "custom": 6014, "orange": 2006,
            "yellow": 1915, "purple": 652
        }
        sales = sales_dict.get(paint_color, 0)
        return "Popular color" if sales > 20000 else "Less Popular color"

    def fit(self, X_train, y_train):
        X_train = X_train.copy()
        X_train['selling_category'] = X_train['manufacturer'].apply(self.categorize_sales)
        X_train['vehicle_type_category'] = X_train['type'].apply(self.categorize_vehicle_type)
        X_train['paint_color_category'] = X_train['paint_color'].apply(self.categorize_paint_color)

        self.target_encoder.fit(X_train[['model']], y_train)
        self.column_transformer.fit(X_train)

    def transform(self, X):
        X = X.copy()
        X['selling_category'] = X['manufacturer'].apply(self.categorize_sales)
        X['vehicle_type_category'] = X['type'].apply(self.categorize_vehicle_type)
        X['paint_color_category'] = X['paint_color'].apply(self.categorize_paint_color) 

        X[['model']] = self.target_encoder.transform(X[['model']])
        transformed_data = self.column_transformer.transform(X)
        return pd.DataFrame(transformed_data, columns=self.column_transformer.get_feature_names_out())

def train_and_save():
    data = pd.read_csv(r"C:\Users\Dell\Downloads\Data Science Stuff\Machine Learning\Internship Tasks\Data\car_data.csv")
    X = data.drop(columns=['price'])
    y = data['price']
        
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train = X_train.dropna()
    X_test = X_test.dropna()
        
    preprocessor = DataPreprocessor()
    preprocessor.fit(X_train, y_train)
        
    with open('preprocessor.pkl','wb') as f:
        pickle.dump(preprocessor, f)
        
    print("Preprocessor Saved Successfully")

# if __name__ == "__main__":
#     train_and_save()
