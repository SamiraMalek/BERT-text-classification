import torch
import pandas as pd
from transformers import GPTNeoForCausalLM, AutoTokenizer

# Load the CSV file
file_path = '/scratch/sxm6547/CSE_584/Midterm_project/Constraint_Train.csv'
df = pd.read_csv(file_path)

# Use the first 2000 rows
df = df.head(2000)

# Load the pre-trained tokenizer and model
model_name = 'EleutherAI/gpt-neo-125M'  # You can use 'gpt-neo-125M' for a smaller version #EleutherAI/gpt-neo-1.3B'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = GPTNeoForCausalLM.from_pretrained(model_name)

# Move the model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Function to generate text using GPT-Neo model
def generate_text_gpt_neo(prompt, max_new_tokens=50):
    inputs = tokenizer.encode(prompt, return_tensors='pt', truncation=True, max_length=10).to(device)
    outputs = model.generate(inputs, max_new_tokens=max_new_tokens, no_repeat_ngram_size=2, early_stopping=True)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Generate text for each tweet
generated_texts = []
for tweet in df['tweet']:
    generated_text = generate_text_gpt_neo(tweet)
    generated_texts.append(generated_text)

# Save input and generated text in CSV
df['generated_text'] = generated_texts
output_file_path = '/scratch/sxm6547/CSE_584/Midterm_project/gpt_neo.csv'
df.to_csv(output_file_path, index=False)

print(f"Input and generated texts saved to {output_file_path}")