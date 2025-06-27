# FirstGPTFromScratch

A custom GPT (Generative Pre-trained Transformer) implementation built from scratch and trained on Lewis Carroll's "Alice's Adventures in Wonderland".

## Model Description

This is a **GPT-based transformer language model** trained from scratch on Lewis Carroll's "Alice's Adventures in Wonderland". This model demonstrates a custom implementation of the GPT architecture for text generation tasks, specifically fine-tuned on classic literature.

## Model Details

- **Model Type**: GPT (Generative Pre-trained Transformer)
- **Architecture**: Custom transformer-based language model
- **Training Data**: Alice's Adventures in Wonderland by Lewis Carroll
- **Language**: English
- **Library**: PyTorch
- **Model Size**: ~4.2M parameters (based on complete_gpt_model.pth)

## Training Details

### Dataset
- **Source**: Alice's Adventures in Wonderland (complete text)
- **Size**: 1,033 lines of text
- **Preprocessing**: Custom tokenization using character-level or subword tokenization

### Training Configuration
- **Epochs**: 3 (checkpoint files available for each epoch)
- **Optimizer**: Likely AdamW (standard for transformer models)
- **Training Files**:
  - `checkpoint_epoch_1.pth` (12.2MB)
  - `checkpoint_epoch_2.pth` (12.2MB) 
  - `checkpoint_epoch_3.pth` (12.2MB)
  - `best_model.pth` (4.14MB) - Best performing checkpoint
  - `complete_gpt_model.pth` (4.20MB) - Final trained model

## Files in this Repository

| File | Size | Description |
|------|------|-------------|
| `complete_gpt_model.pth` | 4.20MB | Final trained model weights |
| `best_model.pth` | 4.14MB | Best performing model checkpoint |
| `checkpoint_epoch_1.pth` | 12.2MB | Training checkpoint after epoch 1 |
| `checkpoint_epoch_2.pth` | 12.2MB | Training checkpoint after epoch 2 |
| `checkpoint_epoch_3.pth` | 12.2MB | Training checkpoint after epoch 3 |
| `tokenizer.pkl` | 37.3KB | Custom tokenizer for the model |
| `dataset.txt` | 51KB | Training dataset (Alice in Wonderland) |
| `Notebook1.ipynb` | 4.1MB | Training notebook with implementation |

## Usage

### Loading the Model

```python
import torch
import pickle

# Load the tokenizer
with open('tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

# Load the model
model = torch.load('complete_gpt_model.pth', map_location='cpu')
model.eval()
```

### Text Generation

```python
def generate_text(model, tokenizer, prompt, max_length=100):
    model.eval()
    with torch.no_grad():
        # Tokenize input
        input_ids = tokenizer.encode(prompt)
        
        # Generate text
        for _ in range(max_length):
            # Your generation logic here
            # This will depend on your specific implementation
            pass
    
    return generated_text

# Example usage
prompt = "Alice was beginning to get very tired"
generated = generate_text(model, tokenizer, prompt)
print(generated)
```

## Model Performance

The model has been trained for 3 epochs on the Alice in Wonderland dataset. Performance metrics and loss curves can be found in the training notebook (`Notebook1.ipynb`).

### Expected Outputs
Given the training on Alice in Wonderland, the model should generate text in a similar style to Lewis Carroll's writing, with:
- Victorian-era English vocabulary and sentence structure
- Whimsical and fantastical content
- Character references from the original story
- Descriptive and narrative prose style

## Training Process

The training was conducted using:
1. **Data Preprocessing**: Text cleaning and tokenization
2. **Model Architecture**: Custom GPT implementation
3. **Training Loop**: 3 epochs with checkpoint saving
4. **Validation**: Best model selection based on validation metrics

## Limitations

- **Dataset Size**: Trained on a single book, limiting vocabulary and style diversity
- **Domain Specificity**: Optimized for Lewis Carroll's writing style
- **Scale**: Relatively small model compared to modern large language models
- **Context Length**: Limited context window typical of smaller transformer models

## Ethical Considerations

- This model is trained on public domain literature (Alice in Wonderland)
- The training data is from 1865 and may contain outdated language or concepts
- The model is intended for educational and demonstration purposes

## Citation

If you use this model, please cite:

```bibtex
@misc{karthik2024alice_gpt,
  title={1st Demo GPT Based Architecture Model},
  author={Karthik},
  year={2024},
  howpublished={Hugging Face Model Hub},
  url={https://huggingface.co/karthik-2905/1st_Demo_GPT_Based_Architecture_Model}
}
```

## License

This model is released under the MIT License. The training data (Alice's Adventures in Wonderland) is in the public domain.

## 🚀 Quick Start

1. **Clone the repository**:
   ```bash
   git clone https://github.com/GruheshKurra/FirstGPTFromScratch.git
   cd FirstGPTFromScratch
   ```

2. **Install dependencies**:
   ```bash
   pip install torch numpy matplotlib jupyter
   ```

3. **Run the training notebook**:
   ```bash
   jupyter notebook Notebook1.ipynb
   ```

4. **Download trained models** from [Hugging Face](https://huggingface.co/karthik-2905/1st_Demo_GPT_Based_Architecture_Model)

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🔗 Related Links

- **Hugging Face Model**: [karthik-2905/1st_Demo_GPT_Based_Architecture_Model](https://huggingface.co/karthik-2905/1st_Demo_GPT_Based_Architecture_Model)
- **Training Dataset**: Alice's Adventures in Wonderland (Public Domain)

## Contact

For questions or issues, please open an issue in this repository.

---

*This project was created as a learning exercise to demonstrate GPT architecture implementation and training from scratch on classic literature.* 