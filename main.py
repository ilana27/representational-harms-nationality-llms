import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import login

# Login to Hugging Face (ensure to set your token)
login(token="")  # This prompts for token input securely

# Load the DataFrame
filtered_df = pd.read_json("data/filtered_search_results.json", orient='records', lines=True)

# Load model and tokenizer
model_name = "google/gemma-2-2b-it"

try:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
except Exception as e:
    raise RuntimeError(f"Failed to load model or tokenizer: {e}")

# Check device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Function to classify mentions
def classify_nationality_mention(texts, mentions):
    classifications = []

    # Loop over texts and mentions for generation
    for text, mention in zip(texts, mentions):
        prompt = f"""
        Text: "{text}"
        Mention: "{mention}"
        Question: Does the mention refer to the language, name, or the nationality of a character? 
        Answer with one word, your options are: Language, Name, Nationality, or Uncertain.
        Answer:
        """

        inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to(device)
        with torch.no_grad():
            output = model.generate(**inputs, max_length=inputs.input_ids.shape[1] + 10)
        
        answer = tokenizer.decode(output[0], skip_special_tokens=True)
        
        # Extracting the model's judgment
        if "Language" in answer:
            classifications.append("Language")
        elif "Name" in answer:
            classifications.append("Name")
        elif "Nationality" in answer:
            classifications.append("Nationality")
        else:
            classifications.append("Uncertain")

    return classifications

# Analyze all narratives and store results in JSON format
classifications = []
batch_size = 16  # Adjust the batch size according to your GPU memory

for i in range(0, len(filtered_df), batch_size):
    batch_df = filtered_df.iloc[i:i + batch_size]
    narratives = batch_df["LLM Response"].tolist()
    mentions = batch_df["Country"].tolist()

    # Classify batch
    batch_classifications = classify_nationality_mention(narratives, mentions)
    classifications.extend(batch_classifications)

# Add classifications as a new column to the DataFrame
filtered_df["Classifications"] = classifications

# Store the filtered DataFrame in a JSON file
output_file = "data/filtered_nationality_classification_results.json"
filtered_df.to_json(output_file, orient="records", lines=True)

print(f"Filtered DataFrame with classifications saved to '{output_file}'")

# Display first few rows to verify
print(filtered_df.head())
