from sentence_transformers import SentenceTransformer, util
from sklearn.metrics import accuracy_score
from ..finetuning_sentence_encoder.finetune_dataset import create_test_dataset
import os
import numpy as np
import torch


def compute_similarity(test_data, model, batch_size=64, device='cuda'):
    results = []

    model.to(device)
    model.eval()

    with torch.no_grad():
        for start_idx in range(0, len(test_data), batch_size):
            batch = test_data[start_idx: start_idx + batch_size]

            anchors = [item["anchor"] for item in batch]
            positives = [item["positive"] for item in batch]
            negatives = [item["negatives"] for item in batch]

            # flatten list of negatives for batch processing
            flat_negatives = [neg for sublist in negatives for neg in sublist]

            # encode all texts
            anchor_embeddings = model.encode(anchors, convert_to_tensor=True, device=device)
            positive_embeddings = model.encode(positives, convert_to_tensor=True, device=device)
            negative_embeddings = model.encode(flat_negatives, convert_to_tensor=True, device=device)

            # reshape negatives to match original shape
            negative_embeddings = negative_embeddings.view(len(batch), -1, negative_embeddings.shape[-1])

            for i in range(len(batch)):
                pos_score = util.pytorch_cos_sim(anchor_embeddings[i], positive_embeddings[i]).item()
                neg_scores = util.pytorch_cos_sim(anchor_embeddings[i], negative_embeddings[i]).squeeze().tolist()

                all_scores = [(pos_score, "positive")] + [(score, "negative") for score in neg_scores]
                all_scores.sort(reverse=True, key=lambda x: x[0])
                positive_rank = next(j for j, (_, label) in enumerate(all_scores) if label == "positive") + 1

                results.append({
                    "anchor": anchors[i],
                    "positive_score": pos_score,
                    "negative_scores": neg_scores,
                    "positive_rank": positive_rank,
                    "correct": positive_rank == 1
                })

    return results
            

def compute_similarity_01(test_data, model, num_samples=10):
    results = []
    max_samples = num_samples

    for i, item in enumerate(test_data):
        if i >= max_samples:
            break

        anchor = item["anchor"]
        positive = item["positive"]
        negatives = item["negatives"]
        
        # Encode anchor and PDDL candidates
        anchor_embedding = model.encode(anchor, convert_to_tensor=True)
        positive_embedding = model.encode(positive, convert_to_tensor=True)
        negative_embeddings = model.encode(negatives, convert_to_tensor=True)
        
        # Compute cosine similarities
        pos_score = util.pytorch_cos_sim(anchor_embedding, positive_embedding).item()
        neg_scores = util.pytorch_cos_sim(anchor_embedding, negative_embeddings).squeeze().tolist()

        # Combine scores and rank them
        all_scores = [(pos_score, "positive")] + [(score, "negative") for score in neg_scores]
        all_scores.sort(reverse=True, key=lambda x: x[0])  # Sort by similarity in descending order

        # Find rank of the correct (positive) example (1-based index)
        positive_rank = next(i for i, (_, label) in enumerate(all_scores) if label == "positive") + 1

        curr_res = {
            "anchor": anchor,
            "positive_score": pos_score,
            "negative_scores": neg_scores,
            "positive_rank": positive_rank,
            "correct": positive_rank == 1  # Correct if positive is ranked first
        }
        
        print(f"Results for iteration {i}:\nPositive score: {curr_res['positive_score']}\nNegative Scores:{curr_res['negative_scores']}\nPositive Rank:{curr_res['positive_rank']}\n")

        # Store results
        results.append(curr_res)

    return results

def evaluate_model(results, k=3):
    """
    Evaluates the model using Precision, Recall, F1-score, Accuracy, Precision@K, and MRR.
    """
    # Standard classification metrics
    y_true = [1] * len(results)
    y_pred = [1 if item["correct"] else 0 for item in results]

    # precision = precision_score(y_true, y_pred, zero_division=0)
    # recall = recall_score(y_true, y_pred, zero_division=0)
    # f1 = f1_score(y_true, y_pred, zero_division=0)
    accuracy = accuracy_score(y_true, y_pred)

    # Precision@K (is the correct answer in the top K?)
    precision_at_k = np.mean([1 if item["positive_rank"] <= k else 0 for item in results])

    # Mean Reciprocal Rank (MRR)
    mrr = np.mean([1.0 / item["positive_rank"] for item in results])

    return {
        "accuracy": accuracy,
        f"precision@{k}": precision_at_k,
        "mrr": mrr,
    }


def save_metrics(metrics, results_pth, filename):
    os.makedirs(results_pth, exist_ok=True)
    with open(os.path.join(results_pth, filename), "w") as f:
         for metric, value in metrics.items():
            f.write(f"{metric}: {value}\n")
    

if __name__ == "__main__":

    model_path = "data/03_models/codebert-base-trained"
    test_data = create_test_dataset()

    model = SentenceTransformer(model_path)
    results = compute_similarity_01(test_data=test_data, model=model, num_samples=20)
    metrics = evaluate_model(results)

    print(metrics)