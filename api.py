# import pickle
# import pandas as pd
# from fastapi import FastAPI
# from preprocessing import DataPreprocessor
# from pydantic import BaseModel

# app = FastAPI()

# with open('trained_model.pkl', 'rb') as file:
#     model = pickle.load(file)

# with open('preprocessor.pkl', 'rb') as file:
#     preprocessor = pickle.load(file)

# class InputData(BaseModel):
#     manufacturer: str
#     model: str
#     condition: str
#     cylinders: str
#     fuel: str
#     title_status: str
#     transmission: str
#     drive: str
#     type: str
#     paint_color: str
#     year: int 
#     odometer: float
    
# @app.post('/predict')
# async def predict(data: InputData):
    
#     try:
#         df = pd.DataFrame([data.dict()])
#         processed_data = preprocessor.transform(df)
#         prediction = model.predict(processed_data)[0]
#         return {"predicted_price": prediction}
#     except Exception as e:
#         return {"error": str(e)}

import pickle
import pandas as pd
from fastapi import FastAPI
from data_preprocessing import DataPreprocessor  # Import directly
from pydantic import BaseModel

# app = FastAPI()

# with open('trained_model.pkl', 'rb') as file:
#     model = pickle.load(file)

# with open('preprocessor.pkl', 'rb') as file:
#     preprocessor = pickle.load(file)

# class InputData(BaseModel):
    # manufacturer: str
    # model: str
    # condition: str
    # cylinders: str
    # fuel: str
    # title_status: str
    # transmission: str
    # drive: str
    # type: str
    # paint_color: str
    # year: int 
    # odometer: float
    
# @app.post('/predict')
# async def predict(data: InputData):
    
#     try:
#         df = pd.DataFrame([data.dict()])
#         processed_data = preprocessor.transform(df)
#         prediction = model.predict(processed_data)[0]
#         return {"predicted_price": prediction}
#     except Exception as e:
#         return {"error": str(e)}