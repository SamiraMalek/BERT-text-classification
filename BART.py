import torch
import pandas as pd
from transformers import BartForConditionalGeneration, BartTokenizer

# Load the CSV file
file_path = '/scratch/sxm6547/CSE_584/Midterm_project/Constraint_Train.csv'  # Update this to your file path
df = pd.read_csv(file_path)

# Ensure you use the first 2000 rows and process the 'tweet' column
df = df.head(2000)

# Load the pre-trained tokenizer and model
model_name = 'facebook/bart-base'
tokenizer = BartTokenizer.from_pretrained(model_name)
model = BartForConditionalGeneration.from_pretrained(model_name)

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Function to generate text using the model
def generate_text_bart(prompt, max_new_tokens=50):
    # Encode the input, limit to 10 tokens
    inputs = tokenizer.encode(prompt, return_tensors='pt', truncation=True, max_length=10).to(device)
    
    # Generate text
    outputs = model.generate(inputs, max_new_tokens=max_new_tokens, num_return_sequences=1, no_repeat_ngram_size=2, early_stopping=True)
    
    # Decode and return the generated text
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Prepare lists for storing input and generated output
input_texts = []
generated_texts = []

# Iterate over each tweet, truncate to 10 tokens, and generate text
for tweet in df['tweet']:
    # Generate text based on the truncated input and save the output
    generated_text = generate_text_bart(tweet)
    generated_texts.append(generated_text)

# Save the input and generated output in a new CSV
df['generated_text'] = generated_texts
output_file_path = '/scratch/sxm6547/CSE_584/Midterm_project/bart.csv'  # Update this with your desired output path
df.to_csv(output_file_path, index=False)

print(f"Input and generated texts saved to {output_file_path}")