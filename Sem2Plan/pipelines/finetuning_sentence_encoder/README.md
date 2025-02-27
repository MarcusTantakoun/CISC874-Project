**FYI, what is easy negatives, semi-hard negatives and hard negatives**

In our case where we want to match the descriptions with the action schema, the easy negatives can be the action schema from other domains, the semi-hard negatives can be the action schema from the same domain but different action, and the hard negatives can be the action schema from the same domain but manipulated a bit (polluted with noise).

## Fine-tuning with easy negatives, semi-hard negatives, and hard negatives triplet loss
In our case where we want to match the descriptions with the action schema, the easy negatives can be the action schema from other domains, the semi-hard negatives can be the action schema from the same domain but different action, and the hard negatives can be the action schema from the same domain but manipulated a bit (polluted with noise).


## How to train in Sentence Transformer library 
- ref: https://github.com/UKPLab/sentence-transformers/blob/master/examples/training/nli/training_nli_v3.py
