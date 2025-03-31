from Sem2Plan.pipelines.finetuning_sentence_encoder.nodes import train_sentence_encoder
import os
from datetime import datetime
from pathlib import Path

if __name__=="__main__":

    print("THIS IS WORKING")

    # save path of model
    output_dir = os.path.join("data/03_models", f"finetuned_sentence_encoder_batch_test_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}")
    print(f"Model will be saved at: {output_dir}")
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    file_path = os.path.join(output_dir, "training_log.txt")

    # Write some content into the file
    with open(file_path, "w") as f:
        f.write("Training started...\n")

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