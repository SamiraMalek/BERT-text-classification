import torch
import pandas as pd
from transformers import BloomForCausalLM, AutoTokenizer

# Load the CSV file
file_path = '/content/Constraint_Train.csv'  # Update this to your file path
df = pd.read_csv(file_path)

# Ensure you use the first 2000 rows and process the 'tweet' column
df = df.head(2000)

# Load the pre-trained tokenizer and model
model_name = 'bigscience/bloom-560m'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = BloomForCausalLM.from_pretrained(model_name)

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Function to generate text using the model
def generate_text_bloom(prompt, max_new_tokens=50):
    inputs = tokenizer.encode(prompt, return_tensors='pt', truncation=True, max_length=10).to(device)
    outputs = model.generate(inputs, max_new_tokens=max_new_tokens, num_return_sequences=1, no_repeat_ngram_size=2, early_stopping=True)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Prepare lists for storing inputs and generated outputs
input_texts = []
generated_texts = []

# Apply the text generation to each tweet (first 20 tokens)
i = 0
for tweet in df['tweet']:
    i += 1
    print(i)
    # Truncate the input tweet to 20 tokens and save it
    truncated_input = tokenizer.decode(tokenizer.encode(tweet, truncation=True, max_length=10), skip_special_tokens=True)
    input_texts.append(truncated_input)

    # Generate text based on the truncated input and save the output
    generated_text = generate_text_bloom(truncated_input)
    print(generated_text)
    generated_texts.append(generated_text)

# Save the input and generated output in a new CSV
df['input_text'] = input_texts
df['generated_text'] = generated_texts
output_file_path = 'input_and_generated_tweets.csv'  # Update this with your desired output path
df.to_csv(output_file_path, index=False)

print(f"Input and generated texts saved to {output_file_path}")