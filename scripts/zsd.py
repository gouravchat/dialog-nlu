from transformers import pipeline


classifier = pipeline("zero-shot-classification")

from flask import Flask, jsonify, request

app = Flask(__name__)

def initialize():
    global zsd_model
    zsd_model = pipeline("zero-shot-classification")

@app.route('/', methods=['GET', 'POST'])
def hello():
    return 'hello from ZSD service'

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    input_json = request.json
    utterance = input_json["utterance"]
    labels = input_json["candidate_labels"]
    response = classifier(utterance, labels)
    return jsonify(response)



if __name__ =="__main__":

    print(('Starting the Server'))
    initialize()
    # Run app
    app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False)
