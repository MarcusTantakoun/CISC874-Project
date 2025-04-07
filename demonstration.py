"""
RUNS PDDL GENERATION
"""
def run_generate_pddl_example():
    import argparse
    from Sem2Plan.pipelines.generate_dataset.generate_pddl import Blocksworld

    parser = argparse.ArgumentParser(description="Blocksworld Problem Generator")
    parser.add_argument("--name", type=str, default="blocksworld")
    parser.add_argument("--ops", type=int, default=4)
    parser.add_argument("--blocks", type=int, default=10)
    parser.add_argument("--max_iterations", type=int, default=3)
    args = parser.parse_args()

    b = Blocksworld()
    b.generate_problem(dataset_dir="data/05_demonstration/blocksworld", args=args)


"""
RUNS PDDL CONVERSION TO NATURAL LANGUAGE
"""
def run_convert_pddl_example():
    from Sem2Plan.pipelines.generate_dataset.convert_pddl import Blocksworld

    b = Blocksworld()
    b.convert_pddl_to_nl("data/05_demonstration/blocksworld")


"""
RUNS INFERENCE ON TEST DATASET
"""
def run_inference(model, num_samples):
    from sentence_transformers import SentenceTransformer
    from Sem2Plan.pipelines.compare_cos_sim.nodes import compute_similarity_01, evaluate_model
    from Sem2Plan.pipelines.finetuning_sentence_encoder.finetune_dataset import create_test_dataset

    model_path = model
    test_data = create_test_dataset()

    model = SentenceTransformer(model_path)
    results = compute_similarity_01(test_data=test_data, model=model, num_samples=num_samples)
    metrics = evaluate_model(results)

    print(metrics)


if __name__ == "__main__":
    run_generate_pddl_example()

    run_convert_pddl_example()

    run_inference(model="data/03_models/codebert-base-trained", num_samples=50)
    run_inference(model="microsoft/codebert-base", num_samples=50)
