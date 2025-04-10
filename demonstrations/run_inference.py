import os
import sys

def run_inference():
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    import argparse
    from sentence_transformers import SentenceTransformer
    from Sem2Plan.pipelines.compare_cos_sim.nodes import compute_similarity_01, evaluate_model
    from Sem2Plan.pipelines.finetuning_sentence_encoder.finetune_dataset import create_test_dataset

    parser = argparse.ArgumentParser(description="Sentence encoder inference")
    parser.add_argument("--model_path", type=str, default="microsoft/codebert-base")
    parser.add_argument("--num_samples", type=int, default=10)
    args = parser.parse_args()

    test_data = create_test_dataset()

    model = SentenceTransformer(args.model_path)
    results = compute_similarity_01(test_data=test_data, model=model, num_samples=args.num_samples)
    metrics = evaluate_model(results)

    mrr_score = 1/metrics['mrr']
    print(metrics)
    print(f"Final MRR score: {mrr_score}")

if __name__ == "__main__":
    run_inference()
    
    # example: python demonstrations/run_inference.py --model_path data/03_models/codebert-base-trained --num_samples 20
    # python demonstrations/run_inference.py --model_path data/03_models/all-roberta-large-trained --num_samples 20