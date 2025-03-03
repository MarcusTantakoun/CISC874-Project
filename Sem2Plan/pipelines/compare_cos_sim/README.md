Calculates cosine similarity between action schema and the natural language description.

## What to compare 
We may have the following settings:
1. Natural language descriptions vs. Action Schema
2. Explanation vs. Action Schema
3. Natural language descriptions vs. Explanation + Action Schema

## Experiments for testing hard negatives similarity score 
For each action schema, we will create 10 different hard negatives. We will then test if the cross encoder can gives higher similarity score to the correct action schema compared to the hard negatives.

We will draw a box plot for where hue are different comparing settings, x-axis are positives, easy negatives (inter-domain) semi-hard negatives (intra-domain) and hard negatives (polluted action with noise).

The process will also output a dataframe csv file 

## Fine-tuning with easy negatives, semi-hard negatives, and hard negatives
In our case where we want to match the descriptions with the action schema, the easy negatives can be the action schema from other domains, the semi-hard negatives can be the action schema from the same domain but different action, and the hard negatives can be the action schema from the same domain but manipulated a bit (polluted with noise).
