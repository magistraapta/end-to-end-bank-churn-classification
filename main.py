from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

# basemodel for the model input
class Users(BaseModel):
    age: float
    credit_score: float
    balance: float
    estimated_salary: float
    
# load scikit-learn model using joblub
joblib_in = open('bank-churn-model.pkl', 'rb')
model = joblib.load(joblib_in)

# setup fastpi
app = FastAPI()

@app.get("/")
def root():
    return {"message": "Bank Churn Prediction"}

# function to post data into model to give a prediction
@app.post("/bank-churn/predict")
async def predict(data: Users):
    data_dict = data.dict()
    
    # Extract fields and convert to list
    input_features = [
        data_dict['age'],
        data_dict['credit_score'],
        data_dict['balance'],
        data_dict['estimated_salary']
    ]

    # Prepare the input for prediction, ensure it is 2D
    input_data = np.array([input_features], dtype=np.float32)
    
    # Predict using the pre-loaded model
    prediction = model.predict(input_data)

    # Return the prediction as a list
    if prediction == 0:
        return "the customer is likely to not churn"
    else:
        return "the customer is likely to churn"

