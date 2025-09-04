# LLM_TRANSFORMER_TINY_SHAKESPEAR
<h5 style='color:gray;'>10.79M Parameters Model<h5>

This project is a **character-level language model** built completely from scratch using PyTorch. I implemented every part of the model architecture myself, without relying on pre-built transformer libraries. The goal was to reproduce the foundations of modern large language models on a small dataset and demonstrate a deep understanding of how attention mechanisms and transformers actually work under the hood.

---

## Dataset

The model is trained on the **Tiny Shakespeare dataset** (a ~1 MB collection of Shakespeare’s plays). This dataset is widely used for experimenting with character-level models because it is compact, self-contained, and produces text with distinct Shakespearean style.

---

## Model Architecture

The model follows a scaled-down GPT-like architecture:

- **Embedding layer** for tokens and positional encoding  
- **Multi-head self-attention** with masked attention  
- **Feed-forward layers** with non-linearities  
- **Residual connections + LayerNorm** for stability  
- **Final linear head** projecting to vocabulary size  

This architecture allows the model to capture dependencies between characters over long contexts and generate coherent text.

---

## Hyperparameters

| Parameter          | Value   |
|--------------------|---------|
| Batch size         | 64      |
| Block size         | 256     |
| Embedding size     | 384     |
| Attention heads    | 6       |
| Transformer layers | 6       |
| Dropout            | 0.2     |
| Learning rate      | 3e-4    |
| Optimizer          | AdamW   |

**Total Parameters:** 10,788,929  
**Trainable Parameters:** 10,788,929  

---

## Training

- Data split: **90% training / 10% validation**  
- Model trained with character-level cross-entropy loss  
- Training loop includes periodic evaluation on validation data  
- Implemented a simple generation function to autoregressively sample text  

---

## Results

After training, the model generates text that resembles Shakespearean dialogue. While it is not perfect English, it mimics the style, rhythm, and structure of plays in the dataset. Example outputs include stage directions, invented character names, and archaic-looking vocabulary.

---

## Repository Structure
```
.
├── input.txt # Tiny Shakespeare dataset
├── model_dir/ # Saved model checkpoints
│ ├── texter.pkl
│ └── texter.pth
├── model_test.py # Script to load model and generate samples
└── train.py # Full model + training loop implementation
```


---

## Final Notes

I wrote this project **entirely from scratch**, including data preprocessing, model definition, training loop, and generation. No pre-made transformer libraries were used.  

This work reflects my ability to not just use modern AI tools, but to understand and build them from the ground up.

