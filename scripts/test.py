# imports
from dialognlu import TransformerNLU, AutoNLU
from dialognlu.readers.goo_format_reader import Reader

# reading datasets
train_path = "data/snips/train"
val_path = "data/snips/valid"
train_dataset = Reader.read(train_path)
val_dataset = Reader.read(val_path)

# configurations of the model
config = {
    "pretrained_model_name_or_path": "distilbert-base-uncased",
    "from_pt": False,
}
# create a joint NLU model from configurations
nlu_model = TransformerNLU.from_config(config)

# training the model
nlu_model.train(train_dataset, val_dataset, epochs=1, batch_size=64)

# saving model
save_path = "saved_models/joint_distilbert_model"
nlu_model.save(save_path)

# loading the model and do incremental training

# loading model
nlu_model = AutoNLU.load(save_path)

# Continue training
nlu_model.train(train_dataset, val_dataset, epochs=1, batch_size=64)

# evaluate the model
test_path = "../data/snips/test"
test_dataset = Reader.read(test_path)
token_f1_score, tag_f1_score, report, acc = nlu_model.evaluate(test_dataset)
print('Slot Classification Report:', report)
print('Slot token f1_score = %f' % token_f1_score)
print('Slot tag f1_score = %f' % tag_f1_score)
print('Intent accuracy = %f' % acc)

# do prediction
utterance = "add sabrina salerno to the grime instrumentals playlist"
result = nlu_model.predict(utterance)