from sentence_transformers import util
from tqdm import tqdm
import pandas as pd
import numpy as np

from classify import classify_text
from embedding import Embedding
from graph_rag import GraphRAG

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

    for gt in gt_embeddings:
        sims = [util.cos_sim(gt, res).item() for res in res_embeddings]
        if max(sims) >= threshold:
            return True

    return False

def calculate_tp_tn_fp_fn(ground_truth: str, bot_answer: str, threshold: float = 0.8):
    i_dont_know_phrases = ["tôi không biết", "không có thông tin"]
    
    # Normalize answers for comparison
    gt_lower = str(ground_truth).lower().strip()
    bot_lower = str(bot_answer).lower().strip()

    is_gt_unknown = any(phrase in gt_lower for phrase in i_dont_know_phrases)
    is_bot_unknown = any(phrase in bot_lower for phrase in i_dont_know_phrases)

    # Case 1: Ground truth indicates the answer is unknown.
    if is_gt_unknown:
        if is_bot_unknown:
            # Correctly identified as unknown.
            return 0, 1, 0, 0  # TN
        else:
            # Bot provided an answer when it should have said "I don't know".
            return 0, 0, 0, 1  # FN
    
    # Case 2: Ground truth provides an answer.
    else:
        # Use semantic similarity to check accuracy
        is_accurate = cal_accuracy(ground_truth, bot_answer, threshold)
        if is_accurate:
            # Bot provided a correct answer.
            return 1, 0, 0, 0  # TP
        else:
            # Bot provided an answer, but it was incorrect.
            return 0, 0, 1, 0  # FP

def evaluate(excel_path, output_excel_path=None):
    df = pd.read_excel(excel_path)
    TP, TN, FP, FN = 0, 0, 0, 0
    results_data = []

    for index, row in tqdm(df.iterrows(), total=len(df), desc="Evaluating"):
        question = row['question']
        expected_answer = row['ground_truth']
        
        label = classify_text(question)
        bot_answer = graphrag.generate_response(question, label)["response"]
        # bot_answer = row["answer"]

        # --- Evaluation ---
        tp, tn, fp, fn = calculate_tp_tn_fp_fn(expected_answer, bot_answer)
        TP += tp
        TN += tn
        FP += fp
        FN += fn

        is_accurate_flag = (tp == 1 or tn == 1)

        print(f"Question: {question}")
        print(f"Expected Answer: {expected_answer}")
        print(f"Bot Answer: {bot_answer}")
        print(f"Result: {'Correct' if is_accurate_flag else 'Incorrect'} (TP:{tp}, TN:{tn}, FP:{fp}, FN:{fn})")

        results_data.append({
            'question': question,
            'ground_truth': expected_answer,
            'answer': bot_answer,
            'TP': tp,
            'TN': tn,
            'FP': fp,
            'FN': fn
        })

    # Calculate final metrics
    total = TP + FN + FP + TN
    accuracy = (TP + TN) / total if total > 0 else 0
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    results_df = pd.DataFrame(results_data)
    
    # Add summary row with metrics
    summary_row = pd.DataFrame({
        'question': ['SUMMARY METRICS'],
        'ground_truth': [f'Accuracy: {accuracy:.4f}'],
        'answer': [f'Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1_score:.4f}'],
        'TP': [TP],
        'TN': [TN],
        'FP': [FP],
        'FN': [FN]
    })
    
    # Append summary to results
    final_df = pd.concat([results_df, summary_row], ignore_index=True)
    
    if output_excel_path:
        if not output_excel_path.endswith('.xlsx'):
            output_excel_path += '.xlsx'
        final_df.to_excel(output_excel_path, index=False)
        print(f"\nEvaluation results saved to {output_excel_path}")

    print(f"Final Metrics - Accuracy: {accuracy:.2f}, Precision: {precision:.2f}, Recall: {recall:.2f}, F1-Score: {f1_score:.2f}")
    print(f"TP: {TP}, TN: {TN}, FP: {FP}, FN: {FN}")


def main():
    excel_path = "data/evaluate/dataset.xlsx"
    output_path = "1000_evaluation.xlsx"
    evaluate(excel_path, output_path)

if __name__ == "__main__":
    main()
