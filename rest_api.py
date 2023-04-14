from flask import Flask, request, jsonify
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from joblib import load
from tensorflow import keras



class RestApi:
    
    #Initializes an instance of the Prediction class
    #Loads trained machine learning
    def __init__(self) -> None:
     
        self.logistic_regression = load('LogisticRegression.joblib')
        self.random_forest = load('RandomForest.joblib')
        self.deep_learning = keras.models.load_model('deep_learning.h5')

    #Normalizes the input data using the MinMaxScaler
    @staticmethod
    def data_normalization(data: np.ndarray) -> np.ndarray:
        scaler = MinMaxScaler()
        sample_prescaled = scaler.fit_transform(data)
        return sample_prescaled
    
    #Separates the data into individual samples
    def sample_separation(self, model, data):
        prediction = []
        for element in data:
            value = np.array([element])
            normalized_data = self.data_normalization(value)
            if model == 'deep_learning':
                pred = model.predict(normalized_data)
                pred_process = np.argmax(pred, axis=1)
                prediction.append(pred_process.item())
            else:
                pred = model.predict(normalized_data)
                prediction.append(pred.item())
        return prediction
    
    #Own heuretic function
    def heuristic_model(self, data: np.ndarray) -> list:
        predictions = []
        for i in range(0, len(data)):
            value = data[i][0]

            if value >= 0 and value < 2000:
                predictions.append(3)
            
            if value >= 2000 and value < 2300:
                predictions.append(4)
            
            if value >= 2300 and value < 2600:
                predictions.append(3)
            
            if value >= 2600 and value < 2900:
                predictions.append(5)
            
            if value >= 2900 and value < 3100:
                predictions.append(2)
            
            if value >= 3100 and value < 3250:
                predictions.append(1)
            
            if value >= 3250: 
                predictions.append(7)

        return predictions
    
    #Predicts the output of a logistic regression model
    def logistic_regression_predict(self, data: np.ndarray) -> list:
        model = self.logistic_regression
        return self.sample_separation(model, data)
    
    # Predicts the output of a random forest model
    def random_forest_predict(self, data: np.ndarray) -> list:
        model = self.logistic_regression
        return self.sample_separation(model, data)
    
    #Predicts the output of a deep learning model
    def deep_learning_predict(self, data: np.ndarray) -> list:
        model = self.deep_learning
        prediction = []
        for element in data:
            value = np.array([element])
            normalized_data = self.data_normalization(value)
            pred = model.predict(normalized_data)
            pred_process = np.argmax(pred, axis=1)
            prediction.append(pred_process.item())
        return prediction
    
    #Predicts the output of a selected machine learning model using the given data
    def predict_model(self, model_name: int, data: np.ndarray) -> list:
        prediction = 'Wrong model selected'
        try:
            if model_name == "logistic_regression":
                prediction = self.logistic_regression_predict(data)
            elif model_name == "random_forest":
                prediction = self.random_forest_predict(data)
            elif model_name == 'heuristic':
                prediction = self.heuristic_model(data)
            elif model_name == 'deep_learning':
                prediction = self.deep_learning_predict(data)

                return prediction
        except (ValueError, TypeError) as e:
            return f"Error occurred: {str(e)}"
        return prediction


app = Flask(__name__)
rest_api = RestApi()

#Renders the welcome message for the REST API
@app.route('/')
def index():
    return 'Welcome to the REST API for machine learning models.'


#Calls the predict_model method from the RestAPI class and returns the predicted output
@app.route('/predict/<model_name>', methods=['POST'])
def predict(model_name):
    data = request.get_json(force=True)['data']
    prediction = rest_api.predict_model(model_name, data)
    return jsonify(prediction)


if __name__ == '__main__':
    app.run(debug=True)