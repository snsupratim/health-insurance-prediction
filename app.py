from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np

# Update your model path and load the health insurance model
insurance_model_path = 'insurance.pkl'  # Replace with your actual model path
with open(insurance_model_path, 'rb') as file:
    insurance_model = pickle.load(file)

app = Flask(__name__)

# Define the route for health insurance charge prediction
@app.route('/predict_charges', methods=['POST'])
def predict_charges():
    # Extract data from form for health insurance prediction
    age = request.form.get('age', type=int)
    sex = request.form.get('sex')
    bmi = request.form.get('bmi', type=float)
    children = request.form.get('children', type=int)
    smoker = request.form.get('smoker')

    # Convert categorical data
    sex = 1 if sex == 'male' else 0
    smoker = 1 if smoker == 'yes' else 0

    # Validate inputs
    if age is None or bmi is None or children is None:
        return render_template('insurance.html', prediction_text='Invalid input. Please provide all fields.')

    # Create numpy array for prediction
    final_features = np.array([[age, sex, bmi, children, smoker]])

    # Make health insurance charge prediction
    predicted_charges = insurance_model.predict(final_features)[0]

    return render_template('insurance.html', prediction_text='Predicted Insurance Charges: ${:.2f}'.format(predicted_charges))

# Define the main page route
@app.route('/')
def main_page():
    return render_template('insurance.html')

if __name__ == "__main__":
    app.run(debug=True)
