from flask import Flask, request, render_template
import joblib
import numpy as np

# Initialize the Flask app
app = Flask(__name__)

# Load the trained model and scaler
model = joblib.load('model.pkl')  # Ensure this file exists in the correct folder
scaler = joblib.load('scaler.pkl')  # Ensure this file exists in the correct folder


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Retrieve the values entered by the user from the form
        user_input = [float(x) for x in request.form.values()]

        # Debug: Print input to console for troubleshooting
        print("User input:", user_input)

        # Convert the input into a numpy array and reshape it for the model
        final_features = np.array(user_input).reshape(1, -1)
        print("Reshaped features:", final_features)

        # Scale the user input
        scaled_features = scaler.transform(final_features)
        print("Scaled features:", scaled_features)

        # Make the prediction using the model
        prediction = model.predict(scaled_features)
        print("Prediction:", prediction)

        # Interpret the prediction
        if prediction[0] == 1:
            result = "The patient is likely to have heart disease."
        else:
            result = "The patient is unlikely to have heart disease."

        # Return the result and display it on the same page
        return render_template('index.html', prediction_text=result)

    except Exception as e:
        # In case of an error, show it on the page and print to the console
        print("Error during prediction:", e)
        return render_template('index.html', prediction_text=f"An error occurred: {str(e)}")


if __name__ == "__main__":
    app.run(debug=True)
