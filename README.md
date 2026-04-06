# GPT-from-Scratch: A Decoder-Only Transformer Implementation

In this repository, I follow the book [Build a Large Language Model (From Scratch)](https://github.com/rasbt/LLMs-from-scratch) by Sebastian Raschka, implementing a GPT-style language model using [Python](https://www.python.org/downloads/) and [PyTorch](https://pytorch.org/).

The project utilizes OpenAI's [tiktoken](https://github.com/openai/tiktoken) to tokenize text and a transformer architecture to predict the next token.

## 🛠️ Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/llm-from-scratch.git
   cd llm-from-scratch
   ```

2. Create a virtual environment and install dependencies:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```

## 📈 Usage

### 1. Prepare training data.
You can provide your own `.txt` file containing the text on which the model should be trained. Alternatively, you can download the American Standard Version bible from [openbible](https://www.openbible.com) using the provided script:
```bash
python -m scripts.download_asv
```
This will save a cleaned version under `./datasets/asv.txt`. Its distinct style makes a model trained on it quite entertaining.

### 2. Prepare the config file
You can set the model hyperparameters as well as the path to the `.txt` file to train on in a `.json` file in the `./config/` folder. The file `config/asv.json` provides a starting point that you can modify.

### 3. Training
You are all set to train your model! Assuming your config file is `config/myconfig.json`, you can start training with:
```bash
python -m scripts.train --config config/myconfig.json --epochs 10
```
Training metrics (losses) are automatically saved to `checkpoints/loss_history.json`.

### 4. Inference
You can generate text from a trained checkpoint. Assuming your config file is `config/myconfig.json`, you can generate text with:
```bash
python -m scripts.generate --config config/myconfig.json --temperature 0.8 --max-new-tokens 100 --prompt "In the beginning"
```