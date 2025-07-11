from sentence_transformers import util
from embedding import Embedding
from graph_rag import GraphRAG
from tqdm import tqdm
import pandas as pd
import numpy as np

graphrag = GraphRAG(model_name="gpt-4o-mini")
embedd_model = Embedding()

def split_into_units(text):
    return [chunk.strip() for chunk in text.split(".") if chunk.strip()]

def cal_accuracy(ground_truth: str, response: str, threshold: float = 0.8):
    gt_units = split_into_units(ground_truth)
    res_units = split_into_units(response)

    if not gt_units or not res_units:
        return False

    gt_embeddings = [embedd_model.embed_query(unit) for unit in gt_units]
    res_embeddings = [embedd_model.embed_query(unit) for unit in res_units]

    accuracy_matches = 0
    for gt in gt_embeddings:
        for res in res_embeddings:
            similarities = util.cos_sim(gt, res).item()
            if similarities >= threshold:
                accuracy_matches += 1
    return True if accuracy_matches >= len(gt_units) else False

def cal_precision_recall(ground_truth: str, response: str, threshold: float = 0.8):
    gt_units = split_into_units(ground_truth)
    res_units = split_into_units(response)

    if not gt_units or not res_units:
        return 0.0, 0.0

    gt_embeddings = [embedd_model.embed_query(unit) for unit in gt_units]
    res_embeddings = [embedd_model.embed_query(unit) for unit in res_units]

    recall_matches = 0
    for gt_embed in gt_embeddings:
        similarities = [util.cos_sim(gt_embed, res_embed).item() for res_embed in res_embeddings]
        if max(similarities) >= threshold:
            recall_matches += 1
    recall = recall_matches / len(gt_units)

    precision_matches = 0
    for res_embed in res_embeddings:
        similarities = [util.cos_sim(res_embed, gt_embed).item() for gt_embed in gt_embeddings]
        if max(similarities) >= threshold:
            precision_matches += 1
    precision = precision_matches / len(res_units)

    return precision, recall

def evaluate(excel_path, output_excel_path="300_evaluation.xlsx"):
    df = pd.read_excel(excel_path).sample(n=300, random_state=42)
    accuracy_scores = []
    precision_scores = []
    recall_scores = []
    f1_scores = []

    results_data = []

    for index, row in tqdm(df.iterrows(), total=len(df), desc="Evaluating"):
        question = row['question']
        expected_answer = row['ground_truth']

        # Generate response using the GraphRAG system
        result = graphrag.generate_response(question)
        bot_answer = result["response"]
        
        # --- Semantic Accuracy Evaluation ---
        is_accurate = cal_accuracy(expected_answer, bot_answer)
        accuracy_scores.append(1 if is_accurate else 0)

        # --- Precision, Recall, F1-score Evaluation ---
        precision, recall = cal_precision_recall(expected_answer, bot_answer)
        precision_scores.append(precision)
        recall_scores.append(recall) 

        # Calculate F1-score using standard formula
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        f1_scores.append(f1)

        print(f"Question: {question}")
        print(f"Expected Answer: {expected_answer}")
        print(f"Bot Answer: {bot_answer}")
        print(f"Accuracy: {is_accurate}")
        print(f"Precision: {precision:.2f}, Recall: {recall:.2f}, F1-score: {f1:.2f}\n")

        results_data.append({
            'question': question,
            'ground_truth': expected_answer,
            'answer': bot_answer,
            'accuracy': 1 if is_accurate else 0,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
        })

    # Calculate averages
    avg_accuracy = np.mean(accuracy_scores)
    avg_precision = np.mean(precision_scores)
    avg_recall = np.mean(recall_scores)
    avg_f1_score = np.mean(f1_scores)

    # Append averages to the results
    results_data.append({
        'question': 'AVERAGE',
        'ground_truth': '',
        'answer': '',
        'accuracy': avg_accuracy,
        'precision': avg_precision,
        'recall': avg_recall,
        'f1_score': avg_f1_score,
    })

    results_df = pd.DataFrame(results_data)
    results_df.to_excel(output_excel_path, index=False)
    print(f"Evaluation results saved to {output_excel_path}")

def main():
    excel_path = "./data//evaluate/dataset.xlsx"
    evaluate(excel_path)

main()