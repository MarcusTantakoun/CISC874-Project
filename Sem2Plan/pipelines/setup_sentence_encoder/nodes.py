"""
Since SentenceTransformer expects models from sentence-transformers/*, 
you need to manually wrap CodeBERT as a bi-encoder.
"""

from sentence_transformers import SentenceTransformer, models

def create_sentence_encoder_helper(setup_sentence_encoder_cfg):
    model_name = setup_sentence_encoder_cfg['model_name']
    model_type = setup_sentence_encoder_cfg['model_type']

    if model_type == "bi_encoder":

        try:
            model = SentenceTransformer(model_name)

        except ValueError as ve:
            print(f"[ERROR] {ve}")
            print(f"Unsupported model name: {model_name}")
            return None

        return model
    else:
        return None

def init_bi_encoder(setup_sentence_encoder_cfg):
    model_type = setup_sentence_encoder_cfg['model_type']
    is_evaluated = setup_sentence_encoder_cfg['is_evaluated']
    model = create_sentence_encoder_helper(setup_sentence_encoder_cfg)
    
    if not is_evaluated:
        # run some tests
        if model_type == "bi_encoder":
            
            target_query_str = "The robot is on a grid with 4 rows and 3 columns. \npos-0-0 pos-0-1 pos-0-2 \npos-1-0 pos-1-1 pos-1-2 \npos-2-0 pos-2-1 pos-2-2 \npos-3-0 pos-3-1 pos-3-2 \nThe depot for new blocks is at pos-2-0. \nThe robot is at pos-2-0. \nThe maximum height of blocks is 3. \nYour goal is to build blocks so that the height at pos-0-1 is 3, the height at pos-1-0 is 2. \nYou cannot have an unplaced block at the end."
            target_list = ['(define (problem termes-0036-4x3x3-random_towers_4x3_3_2_1019)\n    (:domain termes)\n    (:objects n0 n1 n2 n3 - numb pos-0-0 pos-0-1 pos-0-2 pos-1-0 pos-1-1 pos-1-2 pos-2-0 pos-2-1 pos-2-2 pos-3-0 pos-3-1 pos-3-2 - position)\n    (:init (IS-DEPOT pos-2-0) (NEIGHBOR pos-0-0 pos-0-1) (NEIGHBOR pos-0-0 pos-1-0) (NEIGHBOR pos-0-1 pos-0-0) (NEIGHBOR pos-0-1 pos-0-2) (NEIGHBOR pos-0-1 pos-1-1) (NEIGHBOR pos-0-2 pos-0-1) (NEIGHBOR pos-0-2 pos-1-2) (NEIGHBOR pos-1-0 pos-0-0) (NEIGHBOR pos-1-0 pos-1-1) (NEIGHBOR pos-1-0 pos-2-0) (NEIGHBOR pos-1-1 pos-0-1) (NEIGHBOR pos-1-1 pos-1-0) (NEIGHBOR pos-1-1 pos-1-2) (NEIGHBOR pos-1-1 pos-2-1) (NEIGHBOR pos-1-2 pos-0-2) (NEIGHBOR pos-1-2 pos-1-1) (NEIGHBOR pos-1-2 pos-2-2) (NEIGHBOR pos-2-0 pos-1-0) (NEIGHBOR pos-2-0 pos-2-1) (NEIGHBOR pos-2-0 pos-3-0) (NEIGHBOR pos-2-1 pos-1-1) (NEIGHBOR pos-2-1 pos-2-0) (NEIGHBOR pos-2-1 pos-2-2) (NEIGHBOR pos-2-1 pos-3-1) (NEIGHBOR pos-2-2 pos-1-2) (NEIGHBOR pos-2-2 pos-2-1) (NEIGHBOR pos-2-2 pos-3-2) (NEIGHBOR pos-3-0 pos-2-0) (NEIGHBOR pos-3-0 pos-3-1) (NEIGHBOR pos-3-1 pos-2-1) (NEIGHBOR pos-3-1 pos-3-0) (NEIGHBOR pos-3-1 pos-3-2) (NEIGHBOR pos-3-2 pos-2-2) (NEIGHBOR pos-3-2 pos-3-1) (SUCC n1 n0) (SUCC n2 n1) (SUCC n3 n2) (at pos-2-0) (height pos-0-0 n0) (height pos-0-1 n0) (height pos-0-2 n0) (height pos-1-0 n0) (height pos-1-1 n0) (height pos-1-2 n0) (height pos-2-0 n0) (height pos-2-1 n0) (height pos-2-2 n0) (height pos-3-0 n0) (height pos-3-1 n0) (height pos-3-2 n0))\n    (:goal (and (height pos-0-0 n0) (height pos-0-1 n3) (height pos-0-2 n0) (height pos-1-0 n2) (height pos-1-1 n0) (height pos-1-2 n0) (height pos-2-0 n0) (height pos-2-1 n0) (height pos-2-2 n0) (height pos-3-0 n0) (height pos-3-1 n0) (height pos-3-2 n0) (not (has-block))))\n)', '(define (problem termes-0036-4x3x3-random_towers_4x3_3_2_1019)\n    (:domain termes)\n    (:objects n0 n1 n2 n3 - numb pos-0-0 pos-0-1 pos-0-2 pos-1-0 pos-1-1 pos-1-2 pos-2-0 pos-2-1 pos-2-2 pos-3-0 pos-3-1 pos-3-2 - position)\n    (:init (IS-DEPOT pos-2-0) (NEIGHBOR pos-0-0 pos-0-1) (NEIGHBOR pos-0-0 pos-1-0) (NEIGHBOR pos-0-1 pos-0-0) (NEIGHBOR pos-0-1 pos-0-2) (NEIGHBOR pos-0-1 pos-1-1) (NEIGHBOR pos-0-2 pos-0-1) (NEIGHBOR pos-0-2 pos-1-2) (NEIGHBOR pos-1-0 pos-0-0) (NEIGHBOR pos-1-0 pos-1-1) (NEIGHBOR pos-1-0 pos-2-0) (NEIGHBOR pos-1-1 pos-0-1) (NEIGHBOR pos-1-1 pos-1-0) (NEIGHBOR pos-1-1 pos-1-2) (NEIGHBOR pos-1-1 pos-2-1) (NEIGHBOR pos-1-2 pos-0-2) (NEIGHBOR pos-1-2 pos-1-1) (NEIGHBOR pos-1-2 pos-2-2) (NEIGHBOR pos-2-0 pos-1-0) (NEIGHBOR pos-2-0 pos-2-1) (NEIGHBOR pos-2-0 pos-3-0) (NEIGHBOR pos-2-1 pos-2-0) (NEIGHBOR pos-2-1 pos-2-2) (NEIGHBOR pos-2-1 pos-3-1) (NEIGHBOR pos-2-2 pos-1-2) (NEIGHBOR pos-2-2 pos-2-1) (NEIGHBOR pos-2-2 pos-3-2) (NEIGHBOR pos-3-0 pos-2-0) (NEIGHBOR pos-3-0 pos-3-1) (NEIGHBOR pos-3-1 pos-2-1) (NEIGHBOR pos-3-1 pos-3-0) (NEIGHBOR pos-3-1 pos-3-2) (NEIGHBOR pos-3-2 pos-2-2) (NEIGHBOR pos-3-2 pos-3-1) (SUCC n1 n0) (SUCC n2 n1) (SUCC n3 n2) (at pos-2-0) (height pos-0-0 n0) (height pos-0-1 n0) (height pos-0-2 n0) (height pos-1-0 n0) (height pos-1-1 n0) (height pos-1-2 n0) (height pos-2-0 n0) (height pos-2-1 n0) (height pos-2-2 n0) (height pos-3-0 n0) (height pos-3-1 n0) (height pos-3-2 n0))\n    (:goal (and (not (height pos-1-2 n0)) (NEIGHBOR pos-2-1 pos-1-1) (height pos-0-0 n0) (height pos-0-1 n3) (height pos-0-2 n0) (height pos-1-0 n2) (height pos-2-0 n0) (height pos-2-1 n0) (height pos-3-0 n0) (height pos-3-1 n0) (height pos-3-2 n0) (not (has-block))))\n)']
            
            query_embedding = model.encode(target_query_str, convert_to_tensor=True)
            corpus_embedding = model.encode(target_list, convert_to_tensor=True)
            
            similarity_scores = model.similarity(query_embedding, corpus_embedding)
            print(similarity_scores)
        
if __name__ == "__main__":
    setup_sentence_encoder_cfg = {
        "model_name": "all-roberta-large-v1",
        "model_type": "bi_encoder",
        "device": "mps",  # Change to "cuda" if using GPU
        "is_evaluated": False
    }
    
    init_bi_encoder(setup_sentence_encoder_cfg)

    setup_sentence_encoder_cfg = {
        "model_name": "microsoft/codebert-base",
        "model_type": "bi_encoder",
        "device": "mps",  # Change to "cuda" if using GPU
        "is_evaluated": False
    }
    
    init_bi_encoder(setup_sentence_encoder_cfg)