from sentence_transformers import SentenceTransformer

def create_sentence_encoder_helper(setup_sentence_encoder_cfg):
    model_name = setup_sentence_encoder_cfg['model_name']
    model_type = setup_sentence_encoder_cfg['model_type']

    device = setup_sentence_encoder_cfg['device']

    if model_type == "bi_encoder":
        model = SentenceTransformer(model_name, device=device)
    else: return None

    return model

def init_bi_encoder(setup_sentence_encoder_cfg):
    model_type = setup_sentence_encoder_cfg['model_type']
    is_evaluated = setup_sentence_encoder_cfg['is_evaluated']
    model = create_sentence_encoder_helper(setup_sentence_encoder_cfg)
    
    if not is_evaluated:
        # run some tests
        if model_type == "bi_encoder":
            
            target_query_str = "Hello world"
            target_list = ['Dogs and cats', 'I like mammals', 'Hi earth.']
            
            query_embedding = model.encode(target_query_str, convert_to_tensor=True)
            corpus_embedding = model.encode(target_list, convert_to_tensor=True)
            
            similarity_scores = model.similarity(query_embedding, corpus_embedding)
            print(similarity_scores)
        
if __name__ == "__main__":
    setup_sentence_encoder_cfg = {
        "model_name": "all-MiniLM-L6-v2",
        "model_type": "bi_encoder",
        "device": "cpu",  # Change to "cuda" if using GPU
        "is_evaluated": False
    }
    
    init_bi_encoder(setup_sentence_encoder_cfg)