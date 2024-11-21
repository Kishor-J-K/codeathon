from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

# Load the model
model = joblib.load('Loan_approval_1.joblib')

# Feature names
features = ['person_age', 'person_income', 'person_home_ownership',
            'person_emp_length', 'loan_intent', 'loan_grade', 'loan_amnt',
            'loan_int_rate', 'loan_percent_income', 'cb_person_default_on_file',
            'cb_person_cred_hist_length']

@app.route('/')
def home():
    return render_template('form.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Extract input data from the form
    input_data = {
        'person_age': int(request.form['person_age']),
        'person_income': float(request.form['person_income']),
        'person_home_ownership': request.form['person_home_ownership'],
        'person_emp_length': float(request.form['person_emp_length']),
        'loan_intent': request.form['loan_intent'],
        'loan_grade': request.form['loan_grade'],
        'loan_amnt': float(request.form['loan_amnt']),
        'loan_int_rate': float(request.form['loan_int_rate']),
        'loan_percent_income': float(request.form['loan_percent_income']),
        'cb_person_default_on_file': request.form['cb_person_default_on_file'],
        'cb_person_cred_hist_length': int(request.form['cb_person_cred_hist_length']),
    }

    # Convert input data to DataFrame
    data_df = pd.DataFrame([input_data])

    # Preprocess the data
    data_df[model['encoded_cols']] = model['encoder'].transform(data_df[model['categorical_cols']])
    data_df[model['numeric_cols']] = model['scaler'].transform(data_df[model['numeric_cols']])

    # Predict using the model
    prediction = model['model'].predict(data_df[model['input_cols']])
    result = 'Approved' if prediction[0] == 1 else 'Rejected'

    return render_template('result.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)
