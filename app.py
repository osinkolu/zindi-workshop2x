from flask import Flask, request, jsonify
import joblib
import pandas as pd


"""{
  "loc": "Katsina",
  "title": "Semi-detached duplex",
  "bedroom": 2.0,
  "bathroom": 2,
  "parking_space": 1
}"""

# Create a dictionary for mapping
house_type_ranks = {
"Apartment":1,
"Flat":2,
"Cottage":3,
"Semi-detached duplex":4,
"Terrace duplex":5,
"Bungalow":6,
"Townhouse":7,
"Detached duplex":8,
"Penthouse":9,
"Mansion":10,
}

# Create a dictionary for mapping
# Complete the mapping of states to integers
state_to_int = {
    'Lagos': 1,
    'Kano': 2,
    'Oyo': 3,
    'Kaduna': 4,
    'Rivers': 5,
    'Delta': 6,
    'Ogun': 7,
    'Ibadan': 8,
    'Enugu': 9,
    'Ekiti': 10,
    'Anambra': 11,
    'Kogi': 12,
    'Kwara': 13,
    'Benue': 14,
    'Plateau': 15,
    'Edo': 16,
    'Ondo': 17,
    'Akwa Ibom': 18,
    'Osun': 19,
    'Bauchi': 20,
    'Niger': 21,
    'Adamawa': 22,
    'Kebbi': 23,
    'Sokoto': 24,
    'Abia': 25,
    'Cross River': 26,
    'Yobe': 27,
    'Zamfara': 28,
    'Ebonyi': 29,
    'Imo': 30,
    'Jigawa': 31,
    'Taraba': 32,
    'Bayelsa': 33,
    'Nassarawa': 34,
    'Gombe': 35,
    'Borno': 36
}




app = Flask(__name__)

# Load the pickled model
model = joblib.load('lgbmodel.pkl')


@app.route('/')
def index():
    return("Hi, this is a GET request to show that the API works. Welcome to Zindi the Workshop 😉")

@app.route('/predict', methods=['POST'])
def predict():
    # Retrieve the JSON payload from the request
    data = request.get_json()
    print(data)

    # Convert the JSON data into a DataFrame
    features = pd.DataFrame(data, index=[0])

    features['title'] = features['title'].map(house_type_ranks)
    features['loc'] = features['loc'].map(state_to_int)

    # Make predictions using the loaded model
    prediction = model.predict(features)

    # Prepare the response as JSON
    response = {'Price Prediction': prediction[0]}

    return jsonify(response)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)