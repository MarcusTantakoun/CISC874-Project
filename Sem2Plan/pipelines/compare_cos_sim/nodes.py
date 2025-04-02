from sentence_transformers import SentenceTransformer, util
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import matplotlib.pyplot as plt
from ..finetuning_sentence_encoder.finetune_dataset import create_test_dataset


model = SentenceTransformer("all-roberta-large-v1")

def compute_similarity(test_data):
    results = []
    
    for item in test_data:
        anchor = item["anchor"]
        positive = item["positive"]
        negatives = item["negatives"]
        
        # ecode anchor and PDDL candidates
        anchor_embedding = model.encode(anchor, convert_to_tensor=True)
        positive_embedding = model.encode(positive, convert_to_tensor=True)
        negative_embeddings = model.encode(negatives, convert_to_tensor=True)
        
        # compute cosine similarities
        pos_score = util.pytorch_cos_sim(anchor_embedding, positive_embedding).item()
        neg_scores = util.pytorch_cos_sim(anchor_embedding, negative_embeddings).squeeze(0)
        
        # find highest similarity score among negatives
        max_neg_score = neg_scores.max().item()
        
        # determine correctness
        is_correct = pos_score > max_neg_score
        
        results.append({
            "anchor": anchor,
            "positive_score": pos_score,
            "max_negative_score": max_neg_score,
            "correct": is_correct
        })
        
    return results

def evaluate_model(results):
    y_true = [1] * len(results) # all true cases are positive examples
    y_pred = [1 if item["correct"] else 0 for item in results] # model predictions
    
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    accuracy = accuracy_score(y_true, y_pred)
    
    return {"precision": precision, "recall": recall, "f1_score": f1, "accuracy": accuracy}


def plot_similarity_scores(results):
    anchors = [item["anchor"][:50] + "..." for item in results]
    pos_scores = [item["positive_score"] for item in results]
    neg_scores = [item["max_negative_score"] for item in results]
    
    plt.plot(anchors, pos_scores, marker="o", linestyle="-", label="Positive Scores", color="blue")
    plt.plot(anchors, neg_scores, marker="x", linestyle="--", label="Max Negative Scores", color="red")
    
    plt.xlabel("Test Cases")
    plt.ylabel("Similarity Score")
    plt.xticks(rotation=90)
    plt.title("Positive vs. Negative Similarity Scores")
    plt.legend()
    plt.show()
    

if __name__ == "__main__":
    test_data = create_test_dataset()
    similarity_results = compute_similarity(test_data)
    metrics = evaluate_model(similarity_results)
    plot_similarity_scores(similarity_results)
    
    print("Evaluation Metrics:", metrics)
    
    
