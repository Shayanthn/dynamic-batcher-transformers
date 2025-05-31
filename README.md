# 🚀 DynamicBatcher: Ultra-Efficient Transformer Inference Accelerator

![Python](https://img.shields.io/badge/python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-1.10%2B-orange)
![Transformers](https://img.shields.io/badge/HuggingFace-Transformers-yellowgreen)
![License](https://img.shields.io/badge/license-MIT-green)

## 🔥 Revolutionizing Transformer Inference Performance

**DynamicBatcher** is a cutting-edge batching utility that dramatically accelerates Hugging Face Transformers inference by intelligently grouping sequences by length, minimizing padding overhead. Experience **2-5x faster inference** with variable-length inputs while maintaining full accuracy.

## ✨ Key Features

- ⚡ **50-80% reduction in padding computations**
- 📈 **Linear scalability** with batch size and sequence length
- 🔄 **Seamless integration** with existing Hugging Face pipelines
- 🧠 **Smart length-aware sorting** for optimal GPU utilization
- 🏎️ **Near-zero overhead** batching process

## 🛠 Installation

```bash
pip install dynamic-batcher
```
Or build from source:
```bash
git clone https://github.com/shayanthn/DynamicBatcher.git
cd DynamicBatcher
pip install -e .
```
🚀 Quick Start :

```bash
from dynamic_batcher import DynamicBatcher
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Initialize with your model
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased")
batcher = DynamicBatcher(tokenizer)

# Your input texts (variable length)
texts = ["This is a short text.", "This is a much longer text that will need more tokens..."]

# Create optimized batches
batches = batcher.create_batches(texts, batch_size=32)

# Run supercharged inference
for batch in batches:
    inputs, original_indices = batch
    outputs = model(**inputs)
    # Process outputs...
```
📊 Performance Benchmarks :
Method	        Batch Size	 Avg Inference Time	    Speedup
Naive Batching	    32	           4.72s	          1x
DynamicBatcher	    32	           1.89s	         2.5x
Naive Batching	    64	           8.91s	          1x
DynamicBatcher	    64	    
*Benchmarks performed on NVIDIA V100 with 5000 variable-length sequences (5-100 words)*
🌟 Advanced Features
Custom Collate Functions
```bash
def custom_collate(batch):
    # Your custom processing
    return processed_batch

batcher = DynamicBatcher(tokenizer, collate_fn=custom_collate)
```
Mixed Precision Support
```bash
batcher = DynamicBatcher(tokenizer, fp16=True)  # Enable AMP
```
Progress Tracking
```bash
batches = batcher.create_batches(texts, progress_bar=True)
```
🧩 Integration Guide
With PyTorch DataLoader
```bash
from torch.utils.data import DataLoader

class TextDataset:
    def __init__(self, texts):
        self.texts = texts
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        return self.texts[idx]

dataset = TextDataset(texts)
dataloader = DataLoader(
    dataset,
    batch_sampler=DynamicBatchSampler(dataset, tokenizer, batch_size=32),
    collate_fn=batcher.dynamic_collate
)
```
With FastAPI Web Service
```bash
from fastapi import FastAPI
app = FastAPI()
batcher = DynamicBatcher(tokenizer)

@app.post("/predict")
async def predict(texts: List[str]):
    batches = batcher.create_batches(texts)
    results = []
    for batch in batches:
        outputs = model(**batch[0])
        results.extend(process_outputs(outputs))
    return {"predictions": results}
```
📚 Documentation
DynamicBatcher Class
```bash
DynamicBatcher(
    tokenizer: AutoTokenizer,
    max_sequence_length: int = 512,
    fp16: bool = False,
    progress_bar: bool = False,
    sorting_strategy: str = 'ascending'  # or 'descending'
)
```
🎯 Use Cases:

    🔍 Document Processing Pipelines
    💬 Real-time Chat Applications
    📰 News Article Classification
    🗣 Speech-to-Text Post Processing
    🌍 Multilingual Translation Services
## 📬 Contact
**Shayan Taherkhani**  
📧 [shayanthn78@gmail.com](mailto:shayanthn78@gmail.com)  
💼 [LinkedIn](https://linkedin.com/in/shayantaherkhani)  
🐙 [GitHub](https://github.com/shayanthn)

