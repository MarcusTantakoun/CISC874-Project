import matplotlib.pyplot as plt
import json

with open('codebert-base.txt', 'r') as file:
    lines = file.readlines()
    
logs = [json.loads(line.strip().replace("'", '"').replace("nan", "null").replace("inf", "null")) for line in lines if line.strip()]



# Extract step-wise losses and corresponding epochs
epochs = [entry['epoch'] for entry in logs if 'loss' in entry]
losses = [entry['loss'] for entry in logs if 'loss' in entry]

# Plotting
plt.figure(figsize=(10, 5))
plt.plot(epochs, losses, marker='o', linewidth=1.5)
plt.title('Training Loss Curve')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True)
plt.tight_layout()

plt.savefig('training_loss_curve_codebert.png') 
