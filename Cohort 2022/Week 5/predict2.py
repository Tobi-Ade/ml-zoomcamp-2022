from flask import Flask, request, jsonify
import pickle 

read_model = "model1.bin"
read_dv = "dv.bin"

with open(read_model, "rb") as model_file:
    model = pickle.load(model_file)

with open(read_dv, "rb") as dv_file:
    dv = pickle.load(dv_file)

score_app = Flask("credit")

@score_app.route("/", methods=(["POST"]))

def predict():
    
    client = request.get_json()
    X = dv.transform([client])
    y_pred = model.predict_proba(X)[0, 1]
    credit = y_pred >= 0.5 
    
    result = {
        'score': float(y_pred),
        'credit': bool(credit)
    }
    
    return jsonify(result)

if __name__ == "__main__":
    score_app.run(debug=True, host="0.0.0.0", port=9696)


            