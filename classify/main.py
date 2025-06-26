import fasttext

model_path = "./classify/models/fasttext_model.bin"
test_data_path = "./classify/data/test_data.txt"

model = fasttext.load_model(model_path)

for label in model.labels:
    label.replace('__label__', '')

def classify_text(query):
    labels, probabilities = model.predict(query, k=1)  # k is the number of top labels to return
    
    print(f"Predicted label: {labels[0]}")
    print(f"Probability: {probabilities[0]}")
    
    return label[0].replace('__label__', '')