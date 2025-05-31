import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from typing import List, Tuple, Dict, Any
import time
import math
import warnings

# --- Attribution and Metadata ---
# Developed by: Shayan Taherkhani
# Email: shayanthn78@gmail.com
# GitHub: shayanthn
# LinkedIn: linkedin.com/in/shayantaherkhani
# Project: DynamicBatcher for Efficient Transformer Inference

class DynamicBatcher:
    """
    A highly optimized batching utility for Hugging Face Transformers models
    that intelligently groups sequences by length to minimize padding overhead
    during inference. This significantly accelerates throughput for variable-length
    input sequences.

    This approach is particularly effective when processing large datasets of
    text (e.g., millions of sentences) where sequence lengths vary widely,
    as it ensures optimal GPU utilization by reducing unnecessary computations
    on padded tokens.

    Author: Shayan Taherkhani
    """
    def __init__(self, tokenizer: AutoTokenizer, max_sequence_length: int = 512):
        """
        Initializes the DynamicBatcher with a tokenizer.

        Args:
            tokenizer (AutoTokenizer): The Hugging Face tokenizer to use for encoding.
            max_sequence_length (int): The maximum sequence length for truncation.
                                        Defaults to 512, typical for BERT-like models.
        """
        if not isinstance(tokenizer, AutoTokenizer):
            raise TypeError("tokenizer must be an instance of transformers.AutoTokenizer")
        self.tokenizer = tokenizer
        self.max_sequence_length = max_sequence_length
        warnings.filterwarnings("ignore", message="Token indices sequence length is longer than the specified maximum sequence length")

    def _sort_and_index_texts(self, texts: List[str]) -> List[Tuple[int, str, int]]:
        """
        Sorts the input texts by their tokenized length and preserves original indices.
        This is a crucial step for efficient dynamic batching.

        Args:
            texts (List[str]): A list of text strings.

        Returns:
            List[Tuple[int, str, int]]: A list of tuples (tokenized_length, text, original_index).
        """
        indexed_texts = []
        for i, text in enumerate(texts):
            # We encode to get the true token length, including special tokens.
            # Using encode_plus for explicit token_ids and attention_mask to be safe.
            encoded = self.tokenizer.encode_plus(
                text,
                add_special_tokens=True,
                max_length=self.max_sequence_length,
                truncation=True
            )
            indexed_texts.append((len(encoded['input_ids']), text, i))
        # Sort by tokenized length
        indexed_texts.sort(key=lambda x: x[0])
        return indexed_texts

    def create_batches(self, texts: List[str], batch_size: int = 32) -> List[Tuple[Dict[str, torch.Tensor], List[int]]]:
        """
        Generates dynamically padded batches from a list of texts.
        Texts are sorted by length, then batched, minimizing padding within each batch.

        Args:
            texts (List[str]): A list of input text strings.
            batch_size (int): The desired maximum number of sequences per batch.

        Returns:
            List[Tuple[Dict[str, torch.Tensor], List[int]]]: A list of batches.
            Each batch contains:
            - A dictionary of PyTorch tensors (input_ids, attention_mask).
            - A list of original indices for the texts in that batch.
        """
        if not texts:
            return []
        if not isinstance(texts, list) or not all(isinstance(t, str) for t in texts):
            raise TypeError("Input 'texts' must be a list of strings.")
        if not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError("batch_size must be a positive integer.")

        sorted_indexed_texts = self._sort_and_index_texts(texts)

        batches = []
        current_batch_texts = []
        current_batch_original_indices = []

        for length, text, original_index in sorted_indexed_texts:
            current_batch_texts.append(text)
            current_batch_original_indices.append(original_index)

            if len(current_batch_texts) == batch_size:
                # Tokenize and pad for the current batch
                encoded_batch = self.tokenizer(
                    current_batch_texts,
                    return_tensors="pt",
                    padding=True,  # این به طول طولانی‌ترین دنباله در این بچ پد می‌کند (padding="longest" پیش‌فرض است وقتی padding=True و max_length مشخص نشده باشد)
                    truncation=True,
                    max_length=self.max_sequence_length
                )
                batches.append((encoded_batch, current_batch_original_indices))
                current_batch_texts = []
                current_batch_original_indices = []

        # Handle the last batch if it's not full
        if current_batch_texts:
            encoded_batch = self.tokenizer(
                current_batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_sequence_length
            )
            batches.append((encoded_batch, current_batch_original_indices))

        return batches

# --- Benchmarking and Demonstration ---

def run_inference(model: torch.nn.Module, batches: List[Tuple[Dict[str, torch.Tensor], List[int]]], device: torch.device) -> List[Any]:
    """
    Helper function to run inference on generated batches.
    """
    model.to(device)
    model.eval()
    # all_input_texts در حوزه این تابع تعریف نشده، باید از جایی که این تابع فراخوانی می‌شود
    # یا به عنوان آرگومان به آن پاس داده شود یا از حوزه سراسری (global scope) قابل دسترسی باشد.
    # فرض می‌کنیم `all_input_texts` از حوزه سراسری قابل دسترسی است.
    global all_input_texts 
    all_predictions = [None] * len(all_input_texts) # Pre-allocate for original order
    
    with torch.no_grad():
        for batch_encoded, original_indices in batches:
            input_ids = batch_encoded['input_ids'].to(device)
            attention_mask = batch_encoded['attention_mask'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1) # Example: classification

            for i, original_idx in enumerate(original_indices):
                all_predictions[original_idx] = predictions[i].item()
    return all_predictions

if __name__ == "__main__":
    print("--- DynamicBatcher for Efficient Transformer Inference ---")
    print("Developed by: Shayan Taherkhani")
    print(f"GitHub: shayanthn | LinkedIn: linkedin.com/in/shayantaherkhani\n")

    # --- Setup ---
    # Use a smaller pre-trained model for faster local demonstration
    model_name = "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Generate synthetic variable-length texts
    print("\nGenerating synthetic variable-length texts...")
    num_texts = 5000  # A good number to see performance difference
    min_len, max_len = 5, 100
    all_input_texts: List[str] = []
    
    for i in range(num_texts):
        # Create text with random length
        length = torch.randint(min_len, max_len, (1,)).item()
        word = "word "
        text = word * length
        all_input_texts.append(text.strip())
    
    print(f"Generated {len(all_input_texts)} texts with lengths varying from {min_len} to {max_len} words.")

    # --- Benchmarking: Standard Padding (Naive Batching) ---
    print("\n--- Benchmarking: Standard Padding (Naive Batching) ---")
    standard_batch_size = 32
    
    start_time = time.time()
    standard_batches: List[Tuple[Dict[str, torch.Tensor], List[int]]] = []
    current_texts = []
    current_indices = []
    for i, text in enumerate(all_input_texts):
        current_texts.append(text)
        current_indices.append(i)
        if len(current_texts) == standard_batch_size:
            encoded_batch = tokenizer(
                current_texts,
                return_tensors="pt",
                padding=True, # Pads to the max length across ALL sequences if not explicitly set to max_length for the batch
                truncation=True,
                max_length=model.config.max_position_embeddings # Pad to max model length
            )
            standard_batches.append((encoded_batch, current_indices))
            current_texts = []
            current_indices = []
    if current_texts: # Last batch
        encoded_batch = tokenizer(
            current_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=model.config.max_position_embeddings
        )
        standard_batches.append((encoded_batch, current_indices))

    print(f"Created {len(standard_batches)} standard batches.")
    
    inference_start_time = time.time()
    _ = run_inference(model, standard_batches, device)
    inference_end_time = time.time()
    
    standard_total_time = inference_end_time - inference_start_time
    print(f"Standard Batching Inference Time: {standard_total_time:.4f} seconds")

    # --- Benchmarking: Dynamic Padding (Shayan's DynamicBatcher) ---
    print("\n--- Benchmarking: Dynamic Padding (Shayan's DynamicBatcher) ---")
    dynamic_batch_size = 32 # Same batch size for fair comparison
    
    dynamic_batcher = DynamicBatcher(tokenizer, max_sequence_length=model.config.max_position_embeddings)
    
    batching_start_time = time.time()
    dynamic_batches = dynamic_batcher.create_batches(all_input_texts, batch_size=dynamic_batch_size)
    batching_end_time = time.time()
    
    print(f"Created {len(dynamic_batches)} dynamic batches.")
    print(f"Dynamic Batching Creation Time: {batching_end_time - batching_start_time:.4f} seconds")
    
    inference_start_time = time.time()
    _ = run_inference(model, dynamic_batches, device)
    inference_end_time = time.time()
    
    dynamic_total_time = inference_end_time - inference_start_time
    print(f"Dynamic Batching Inference Time: {dynamic_total_time:.4f} seconds")

    # --- Performance Comparison ---
    print("\n--- Performance Comparison ---")
    print(f"Standard Batching Total Inference Time: {standard_total_time:.4f} seconds")
    print(f"Dynamic Batching Total Inference Time: {dynamic_total_time:.4f} seconds")
    
    if dynamic_total_time < standard_total_time:
        speedup_factor = standard_total_time / dynamic_total_time
        print(f"\n🥳 DynamicBatcher is {speedup_factor:.2f}x faster for inference!")
        print("This difference becomes even more pronounced with larger datasets and wider variations in sequence lengths.")
    else:
        print("\nDynamicBatcher did not show significant speedup in this specific run. (This is rare, check setup or small dataset size)")

    print("\n--- Next Steps ---")
    print("Consider integrating this `DynamicBatcher` into your AI inference pipelines.")
    print("This class is designed to be easily packaged as a standalone Python library.")
    print("Feel free to contribute or give feedback on GitHub!")

    # --- Potential DataLoader Integration ---
    print("\n--- Potential DataLoader Integration ---")
    print("For training or more complex inference pipelines, consider integrating this DynamicBatcher with PyTorch's DataLoader. ")
    print("You can create a custom Dataset class that uses DynamicBatcher to yield batches.")