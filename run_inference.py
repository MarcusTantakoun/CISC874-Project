# run_inference.py
from sentence_transformers import SentenceTransformer
from Sem2Plan.pipelines.compare_cos_sim.nodes import compute_similarity, evaluate_model, save_metrics
from Sem2Plan.pipelines.finetuning_sentence_encoder.finetune_dataset import create_test_dataset
import gc
import torch

if __name__ == "__main__":
    test_data = create_test_dataset()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model_paths = [
        ("/home/tant2002/projects/def-zhu2048/tant2002/CISC874-Project/data/03_models/finetuned_sentence_encoder_batch_32_2025-03-31_21-27-32/final", "codebert-base-trained"),
        ("/home/tant2002/scratch/codebert-base", "codebert-base"),
        ("/home/tant2002/projects/def-zhu2048/tant2002/CISC874-Project/data/03_models/finetuned_sentence_encoder_batch_32_2025-04-01_01-15-09/final", "all-roberta-large-v1-trained"),
        ("/home/tant2002/scratch/all-roberta-large-v1", "all-roberta-large-v1")
    ]

    # codeBERT-trained: /home/tant2002/projects/def-zhu2048/tant2002/CISC874-Project/data/03_models/finetuned_sentence_encoder_batch_32_2025-03-31_21-27-32/final
    # codeBERT: /home/tant2002/scratch/codebert-base
    # all-RoBERTa-trained: /home/tant2002/projects/def-zhu2048/tant2002/CISC874-Project/data/03_models/finetuned_sentence_encoder_batch_32_2025-04-01_01-15-09/final
    # all-RoBERTa-large-v1: /home/tant2002/scratch/all-roberta-large-v1

    for model_path, model_type in model_paths:
        print(f"Running inference for {model_type}")
        model = SentenceTransformer(model_path)
        results = compute_similarity(test_data=test_data, model=model, device=device)
        metrics = evaluate_model(results)
        save_metrics(metrics, f"data/04_results/{model_type}", "evaluation_metrics.txt")
        print(f"Evaluation Metrics ({model_type}):", metrics)

        del model
        gc.collect()
        torch.cuda.empty_cache()
