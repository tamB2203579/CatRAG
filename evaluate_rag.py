import pandas as pd
from tqdm import tqdm
from sentence_transformers import util

from classify import classify_text
from embedding import Embedding
from graph_rag import GraphRAG
from rag import RAG

# Initialize global embedding model
embedding_model = Embedding()

# Constants
UNKNOWN_PHRASES = ["tôi không biết", "không có thông tin"]


def split_sentences(text: str):
    """Splits a block of text into sentences."""
    return [chunk.strip() for chunk in text.split(".") if chunk.strip()]


def is_similar(ground_truth: str, response: str, threshold: float):
    """Checks if any sentence in the bot response is semantically similar to the ground truth."""
    gt_units = split_sentences(ground_truth)
    res_units = split_sentences(response)

    if not gt_units or not res_units:
        return False

    gt_embeddings = [embedding_model.embed_query(unit) for unit in gt_units]
    res_embeddings = [embedding_model.embed_query(unit) for unit in res_units]

    for gt_emb in gt_embeddings:
        similarities = [util.cos_sim(gt_emb, res_emb).item() for res_emb in res_embeddings]
        if max(similarities) >= threshold:
            return True

    return False


def calculate_confusion_components(ground_truth: str, bot_answer: str, threshold: float):
    """Returns TP, TN, FP, FN based on comparison between ground truth and bot answer."""
    gt_text = str(ground_truth).lower().strip()
    bot_text = str(bot_answer).lower().strip()

    is_gt_unknown = any(phrase in gt_text for phrase in UNKNOWN_PHRASES)
    is_bot_unknown = any(phrase in bot_text for phrase in UNKNOWN_PHRASES)

    if is_gt_unknown:
        if is_bot_unknown:
            # Correctly identified as unknown.
            return 0, 1, 0, 0  # TN
        # Bot provided an answer when it should have said "I don't know".
        return 0, 0, 0, 1      # FN

    if is_similar(ground_truth, bot_answer, threshold):
        # Bot provided a correct answer.
        return 1, 0, 0, 0      # TP
    # Bot provided an answer, but it was incorrect.
    return 0, 0, 1, 0          # FP


def evaluate(
    input_path: str,
    output_path: str = None,
    graphrag: GraphRAG = None,
    rag: RAG = None,
    threshold: float = 0.8
):
    """Evaluates model performance using TP, TN, FP, FN metrics."""
    df = pd.read_excel(input_path)
    TP, TN, FP, FN = 0, 0, 0, 0
    results = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Evaluating"):
        question = row['question']
        ground_truth = row['ground_truth']

        if graphrag:
            label = classify_text(question)
            response = graphrag.generate_response(question, label)["response"]
        else:
            response = rag.generate_response(question)["response"]

        tp, tn, fp, fn = calculate_confusion_components(ground_truth, response, threshold)
        TP += tp
        TN += tn
        FP += fp
        FN += fn

        result_flag = "Correct" if tp or tn else "Incorrect"
        print(f"\nQuestion: {question}\nExpected: {ground_truth}\nAnswer: {response}\nResult: {result_flag} (TP:{tp}, TN:{tn}, FP:{fp}, FN:{fn})")

        results.append({
            'question': question,
            'ground_truth': ground_truth,
            'answer': response,
            'TP': tp, 'TN': tn, 'FP': fp, 'FN': fn
        })

    # Compute metrics
    total = TP + TN + FP + FN
    accuracy = (TP + TN) / total if total else 0
    precision = TP / (TP + FP) if (TP + FP) else 0
    recall = TP / (TP + FN) if (TP + FN) else 0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) else 0

    metrics_row = {
        'question': 'SUMMARY METRICS',
        'ground_truth': f'Accuracy: {accuracy:.4f}',
        'answer': f'Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1_score:.4f}',
        'TP': TP, 'TN': TN, 'FP': FP, 'FN': FN
    }

    results_df = pd.DataFrame(results + [metrics_row])

    if output_path:
        output_path = output_path if output_path.endswith('.xlsx') else output_path + '.xlsx'
        results_df.to_excel(output_path, index=False)
        print(f"\nEvaluation results saved to {output_path}")

    print(f"\nFinal Metrics - Accuracy: {accuracy:.2f}, Precision: {precision:.2f}, Recall: {recall:.2f}, F1-Score: {f1_score:.2f}")
    print(f"TP: {TP}, TN: {TN}, FP: {FP}, FN: {FN}")


def main():
    technique = input("Enter technique ('graphrag' or 'rag'): ").strip().lower()
    model_name = input("Enter model name (default 'gpt-4o-mini'): ").strip() or "gpt-4o-mini"
    threshold = float(input("Enter similarity threshold (default 0.8): ").strip() or 0.8)
    input_path = "data/evaluate/dataset.xlsx"
    output_path = "evaluation_result.xlsx"

    if technique not in ["graphrag", "rag"]:
        print("Invalid technique. Please enter 'graphrag' or 'rag'.")
        return

    graphrag = GraphRAG(model_name=model_name) if technique == "graphrag" else None
    rag = RAG(model_name=model_name) if technique == "rag" else None

    evaluate(input_path, output_path, graphrag=graphrag, rag=rag, threshold=threshold)


if __name__ == "__main__":
    main()
