from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import fasttext

model_path = "./classify/models/fasttext_model.bin"
test_data_path = "./data/classify/test_data.txt"

model = fasttext.load_model(model_path)

# Fix the labels processing - create a proper list with cleaned labels
labels_list = [label.replace('__label__', '') for label in model.labels]

def classify_text(query):
    labels, probabilities = model.predict(query, k=1)  # k is the number of top labels to return
    
    if probabilities[0] >= 0.9:
        return labels[0].replace('__label__', '')
    else:
        return None

def evaluate_model():
    # Read test data
    true_labels = []
    texts = []
    
    with open(test_data_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                # FastText format is "__label__LABEL text"
                parts = line.split(' ', 1)
                if len(parts) == 2:
                    label, text = parts
                    true_labels.append(label.replace('__label__', ''))
                    texts.append(text)
    
    # Predict labels for all test data
    predicted_labels = []
    for text in texts:
        labels, _ = model.predict(text, k=1)
        predicted_labels.append(labels[0].replace('__label__', ''))
    
    # Calculate metrics
    accuracy = accuracy_score(true_labels, predicted_labels)
    precision, recall, f1, _ = precision_recall_fscore_support(
        true_labels, predicted_labels, average='weighted'
    )
    
    # Print results
    print("Model Evaluation Results:")
    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1-Score: {f1}")
    
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }

if __name__ == "__main__":
    import time
    start_time = time.time()
    metrics = evaluate_model()
    print("Testing time", (time.time() - start_time)*1000) # Time ins milisecond