from Sem2Plan.pipelines.finetuning_sentence_encoder.nodes import train_sentence_encoder

if __name__ == "__main__":
    
    setup_sentence_encoder_cfg = {
        "model_name": "/home/tant2002/scratch/codebert-base",
        "model_type": "bi_encoder",
        "device": "cuda",  # Change to "cuda" if using GPU
        "is_evaluated": False
    }

    finetuning_encoder_cfg = {
        "train_batch_size": 256,
        "training_epoch": 40,
        "is_finetune_complete": False
    }

    train_sentence_encoder(setup_sentence_encoder_cfg=setup_sentence_encoder_cfg, finetuning_encoder_cfg=finetuning_encoder_cfg)