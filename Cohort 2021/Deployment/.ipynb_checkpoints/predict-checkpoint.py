"""
Saving the model to flask
"""

"""
importing libraries
"""
from flask import Flask, request, jsonify 
import pickle

"""
Loading objects for prediction are stored
"""
with open('model2.bin', 'rb') as file:
    model = pickle.load(file)

with open('dv.bin', 'rb') as file2:
    dv = pickle.load(file2)

"""
serving the model on a flask app
"""
app = Flask("churn")
@app.route("/pred", methods=['POST'])

def predict():
    print("Loading input")

    customer = request.get_json()

    # print(f"input loaded: {customer} ")

    X = dv.transform([customer])

    y_pred = model.predict_proba(X)[0,1]

    churn = y_pred >= 0.5

    result = {
        'churn probability' : float(y_pred),
        'churn': bool(churn)
    }

    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8080)


