import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Define the paths where the model and tokenizer are saved
model_save_path = "/content/sample_data/model"
tokenizer_save_path = "/content/sample_data/tokenizer"

# Check if GPU is available and select the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the saved model and tokenizer
model = AutoModelForSequenceClassification.from_pretrained(model_save_path)
model.to(device)  # Move model to GPU if available
tokenizer = AutoTokenizer.from_pretrained(tokenizer_save_path)

# Load the test dataset
test_df = pd.read_csv('/content/test.csv')


# Define a function to get predictions
def predict(sentence):
    inputs = tokenizer(sentence, return_tensors="pt", padding=True, truncation=True).to(device)  # Move inputs to GPU if available
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    prediction = torch.argmax(logits, dim=-1)
    return prediction.item()

# Initialize lists to store true and predicted labels
true_labels = []
predicted_labels = []

# Iterate over each row in the test dataset
for index, row in test_df.iterrows():
    sentence = row['text']
    true_label = row['label']
    # Get the predicted label
    predicted_label = predict(sentence)
    # Append the true and predicted labels to the lists
    true_labels.append(true_label)
    predicted_labels.append(predicted_label)

# Convert predicted labels to a NumPy array and save it
predicted_labels_np = np.array(predicted_labels)

# Calculate metrics
accuracy = accuracy_score(true_labels, predicted_labels)
print(f"Accuracy: {accuracy:.4f}")


#plot result for each class

# Assuming you have two lists: true_labels and predicted_labels
test_df = pd.read_csv('/content/test.csv')
true_labels = test_df['label'] # Replace with your true labels
predicted_labels = predicted_labels_np  # Replace with your predicted labels

# Define the number of classes
num_classes = 8
class_labels = [
    'bart',
    'bloom',
    'codegen',
    'gpt2',
    'gpt_neo',
    'opt',
    't5',
    'xlnet'
]

# Initialize a dictionary to store correct predictions and total counts per class
correct_per_class = {i: 0 for i in range(num_classes)}
total_per_class = {i: 0 for i in range(num_classes)}

# Iterate over both true and predicted labels
for true, pred in zip(true_labels, predicted_labels):
    total_per_class[true] += 1  # Count total instances for the true class
    if true == pred:
        correct_per_class[true] += 1  # Count correct predictions for the true class

# Calculate accuracy for each class (multiply by 100 to get percentage)
accuracy_per_class = {i: (correct_per_class[i] / total_per_class[i]) * 100 if total_per_class[i] > 0 else 0 for i in range(num_classes)}

# Plotting
accuracies = [accuracy_per_class[i] for i in range(num_classes)]

plt.figure(figsize=(8, 6))
bars = plt.bar(class_labels, accuracies, color='#1f77b4')  # Darker blue color (hex code)
plt.xlabel('Class Label')
plt.ylabel('Accuracy (%)')
plt.title('Accuracy for Each Class Label')
plt.ylim([0, 105])  # Accuracy as a percentage from 0 to 100
#plt.grid(True)

# Add the accuracy value on top of each bar
for bar, accuracy in zip(bars, accuracies):
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, yval + 1, f'{accuracy:.2f}%', ha='center', va='bottom')

plt.savefig('accuracy.pdf')
# Display the plot
plt.show()