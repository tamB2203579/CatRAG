import fasttext
import time
from ClassificationFasttext import preprocess_text
model_path = './models/fasttext_model_tnn.bin'


model = fasttext.load_model(model_path)
test_data_path = './content/test_data.txt'

start_time = time.time()
nexamples, precision, recall = model.test(test_data_path)
elapsed_time = time.time() - start_time

print("Labels learned by model:")
for label in model.labels:
    print(label.replace('__label__', ''))

print(f'Time: {elapsed_time:.4f} seconds')
print(f'Precision: {precision}')
print(f'Recall: {recall}')
print(f'Number of examples: {nexamples}')


def classify_text(model):
    while True:
        text = input("Enter text to classify (or 'quit' to exit): ")
        if text.lower() == 'quit':
            break
        text = preprocess_text(text)
        labels, probabilities = model.predict(text, k=1)  # k is the number of top labels to return
        
        print(f"Predicted label: {labels[0]}")
        print(f"Probability: {probabilities[0]}")

classify_text(model)