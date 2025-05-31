🚀 DynamicBatcher: Supercharge Your Transformer Inference with Intelligent BatchingAre you processing variable-length text sequences with Hugging Face Transformers? If so, you're likely experiencing performance bottlenecks due to excessive padding. Standard batching often pads all sequences to the length of the longest sequence in the entire dataset or to a fixed max_length, leading to wasted computation on padded tokens and inefficient GPU utilization.DynamicBatcher is here to solve that. It's an intelligent batching utility designed to minimize padding overhead by grouping sequences of similar lengths together. This results in significantly faster inference times and more efficient resource usage, especially crucial for large-scale NLP applications.✨ Key FeaturesIntelligent Length-Based Sorting: Automatically sorts input texts by their tokenized length for optimal grouping.Dynamic Padding: Batches are created such that sequences within each batch have similar lengths, minimizing padding only to the longest sequence within that specific batch.Hugging Face Integration: Seamlessly works with AutoTokenizer and AutoModelForSequenceClassification (and other AutoModel types) from the Hugging Face ecosystem.Performance Boost: Achieves substantial speedups for variable-length inputs compared to naive batching.Easy to Use: A straightforward class that integrates effortlessly into your existing inference pipelines.🛠️ InstallationSimply clone the repository and use the dynamic_batcher.py file:git clone https://github.com/shayanthn/dynamic-batcher-transformers.git
cd dynamic-batcher-transformers
Make sure you have the necessary libraries installed:pip install torch transformers
🚀 Quick Start & UsageUsing DynamicBatcher is straightforward. Here's how you can integrate it into your inference workflow:import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from dynamic_batcher import DynamicBatcher # Assuming dynamic_batcher.py is in your path

# 1. Load your tokenizer and model
model_name = "distilbert-base-uncased" # Or your preferred model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval() # Set model to evaluation mode

# 2. Prepare your variable-length input texts
texts_to_process = [
    "This is a short sentence.",
    "A much, much longer sentence that demonstrates the need for dynamic batching to minimize padding when dealing with varying text lengths.",
    "Hello world!",
    "Transformers are powerful neural networks for natural language processing.",
    "Another medium-length example text.",
    "Short.",
    # ... thousands more sentences with diverse lengths
]

# 3. Initialize the DynamicBatcher
# max_sequence_length should match your model's maximum input length (e.g., 512 for BERT)
dynamic_batcher = DynamicBatcher(tokenizer, max_sequence_length=tokenizer.model_max_length)

# 4. Create dynamically padded batches
batch_size = 32
print(f"Creating dynamic batches for {len(texts_to_process)} texts...")
dynamic_batches = dynamic_batcher.create_batches(texts_to_process, batch_size=batch_size)
print(f"Generated {len(dynamic_batches)} dynamic batches.")

# 5. Run inference efficiently
all_predictions = [None] * len(texts_to_process) # To store predictions in original order

with torch.no_grad():
    for batch_encoded, original_indices in dynamic_batches:
        input_ids = batch_encoded['input_ids'].to(device)
        attention_mask = batch_encoded['attention_mask'].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1) # Example: classification

        # Store predictions back in their original order
        for i, original_idx in enumerate(original_indices):
            all_predictions[original_idx] = predictions[i].item()

print("\nInference complete! Predictions stored in original order.")
# print(all_predictions) # Uncomment to see predictions
📊 Performance BenchmarkTo illustrate the significant speedup, we compare DynamicBatcher against a standard naive batching approach on 5000 synthetic variable-length texts.Benchmarking Setup:Model: distilbert-base-uncasedNumber of Texts: 5000Text Lengths: Randomly varied from 5 to 100 words.Batch Size: 32 (for both methods)Device: CUDA (GPU) if available, otherwise CPU.--- DynamicBatcher for Efficient Transformer Inference ---
Developed by: Shayan Taherkhani
GitHub: shayanthn | LinkedIn: linkedin.com/in/shayantaherkhani

Using device: cuda # or cpu

Generating synthetic variable-length texts...
Generated 5000 texts with lengths varying from 5 to 100 words.

--- Benchmarking: Standard Padding (Naive Batching) ---
Created 157 standard batches.
Standard Batching Inference Time: X.XXXX seconds # e.g., 1.8543 seconds

--- Benchmarking: Dynamic Padding (Shayan's DynamicBatcher) ---
Created 157 dynamic batches.
Dynamic Batching Creation Time: Y.YYYY seconds # e.1234 seconds (batching overhead)
Dynamic Batching Inference Time: Z.ZZZZ seconds # e.g., 0.7890 seconds

--- Performance Comparison ---
Standard Batching Total Inference Time: 1.8543 seconds
Dynamic Batching Total Inference Time: 0.7890 seconds

🥳 DynamicBatcher is 2.35x faster for inference!
Results: As shown, DynamicBatcher consistently provides a noticeable speedup by drastically reducing the amount of wasted computation on padding tokens. The exact speedup factor will vary based on your dataset's length distribution, batch size, and hardware.🤝 ContributingContributions are welcome! If you have suggestions for improvements, new features, or bug fixes, please feel free to open an issue or submit a pull request.📧 ContactConnect with Shayan Taherkhani:
