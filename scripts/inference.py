# imports
from dialognlu import TransformerNLU, AutoNLU
from dialognlu.readers.goo_format_reader import Reader

# reading datasets
# train_path = "data/snips/train"
# val_path = "data/snips/valid"
# train_dataset = Reader.read(train_path)
# val_dataset = Reader.read(val_path)

# # configurations of the model
# config = {
#     "pretrained_model_name_or_path": "distilbert-base-uncased",
#     "from_pt": False,
# }
# # create a joint NLU model from configurations
# nlu_model = TransformerNLU.from_config(config)

# # training the model
# nlu_model.train(train_dataset, val_dataset, epochs=3, batch_size=64)

# # saving model
# save_path = "saved_models/joint_distilbert_model"
# nlu_model.save(save_path)

# # loading the model and do incremental training

# # loading model
# nlu_model = AutoNLU.load(save_path)

# # Continue training
# nlu_model.train(train_dataset, val_dataset, epochs=1, batch_size=64)

# # evaluate the model
# test_path = "../data/snips/test"
# test_dataset = Reader.read(test_path)
# token_f1_score, tag_f1_score, report, acc = nlu_model.evaluate(test_dataset)
# print('Slot Classification Report:', report)
# print('Slot token f1_score = %f' % token_f1_score)
# print('Slot tag f1_score = %f' % tag_f1_score)
# print('Intent accuracy = %f' % acc)

# do prediction


# utterance = "add sabrina salerno to the grime instrumentals playlist"
# result = nlu_model.predict(utterance)



from flask import Flask, jsonify, request

app = Flask(__name__)

def initialize():
    global nlu_model
    save_path = "saved_models/joint_distilbert_model"
    nlu_model = AutoNLU.load(save_path)

@app.route('/', methods=['GET', 'POST'])
def hello():
    return 'hello from NLU service'

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    input_json = request.json
    utterance = input_json["utterance"]
    print(utterance)
    response = nlu_model.predict(utterance)
    return jsonify(response)



if __name__ =="__main__":

    print(('Starting the Server'))
    initialize()
    # Run app
    app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False)


