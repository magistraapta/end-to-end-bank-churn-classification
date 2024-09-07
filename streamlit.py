import streamlit as st
import requests

API_URL = "http://127.0.0.1:8000/bank-churn/predict"  # Ensure the correct endpoint

def main():
    st.title("Bank Churn Prediction")

    # Input fields
    age = st.number_input('Age')
    credit_score = st.number_input('Credit Score')
    balance = st.number_input('Balance')
    estimated_salary = st.number_input('Estimated Salary')

    if st.button('Predict'):
        # Prepare the input data as a dictionary
        input_data = {
            "age": age,
            "credit_score": credit_score,
            "balance": balance,
            "estimated_salary": estimated_salary
        }
        
        # Send the request to the FastAPI endpoint
        headers = {'Content-Type': 'application/json'}
        response = requests.post(API_URL, json=input_data, headers=headers)

        # Check if the response is successful
        if response.status_code == 200:
            # Display the prediction result
            st.write(f"Prediction: {response.json()}")
        else:
            st.write(f"Error: {response.status_code} - {response.text}")

if __name__ == "__main__":
    main()
