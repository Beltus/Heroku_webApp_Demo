import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle #Needed to read the model pickle saved in project code

#create flask app
app = Flask(__name__)

#load the model pickle file
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/') #this renders the home page named here as index.html
def home():
    return render_template('index.html') #render index.html page as home page of our webapp

@app.route('/predict',methods=['POST']) #create /predict as a post method. calls the predict() 
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [int(x) for x in request.form.values()] #gets all input values entered by user
    final_features = [np.array(int_features)] #convert values to arrays
    prediction = model.predict(final_features) #make prediction on array values using our trained model

    output = round(prediction[0], 2) #get outpit

    return render_template('index.html', prediction_text='Employee Salary should be $ {}'.format(output)) #render results on the homepage with additional text


#run the flask
if __name__ == "__main__":
    app.run(debug=True)
