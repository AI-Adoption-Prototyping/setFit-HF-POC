# SetFit Training and Evaluation

This repository contains scripts for training and evaluating SetFit models on the AG News dataset. SetFit is a few-shot learning framework that uses sentence transformers for efficient text classification.

## My Usecase

I'm working on implementing role-based tagging for a workorder system. We've been manually tagging tickets for years and have accumulated a large dataset, making this an ideal starting point for data cleaning and demonstrating AI capabilities. The initial plan is to add AI-generated tags as comments for support staff to evaluate, with the goal of eventually automating the assignment process.

### Plans for the Future

**Phase 1: Iterative Model Improvement**

Once we have the tagging system working, we'll establish a regular retraining cycle. This will help us maintain model accuracy as our data evolves and create a better, more closely reviewed dataset over time. We'll retrain the model after tickets have been closed and reviewed. Starting with the most recent 150 tickets, we'll use a 70/15/15 train/validation/test split. Over time, we'll adjust both the dataset size and retraining frequency to find the optimal balance between model performance and operational efficiency.

**Phase 2: Automated Pipeline and Agentic Interactions**

The second phase will involve building an automated pipeline with agentic interactions for live data processing. The initial agentic system will focus on tagging, incorporating heuristic methods and machine learning approaches with the ability to learn from feedback. The next phase will expand to:
- Generating improved ticket subjects and descriptions
- Summarizing ticket content
- Upon ticket closure, automatically creating a summarized solution from the ticket messages and appending it to the ticket 

## Overview

SetFit is particularly useful when you have limited labeled data. It works by:

1. Using a pretrained sentence transformer to create embeddings
2. Training a classification head on a small number of labeled examples
3. Achieving strong performance with minimal training data

## Scripts

### `setfit_01.py` - Model Training

This script trains a SetFit model on the AG News dataset using few-shot learning.

#### Purpose

The script demonstrates how to:

- Load and prepare a dataset for SetFit training
- Use few-shot learning with only 100 training examples
- Train a SetFit model with a pretrained sentence transformer backbone
- Evaluate the model and save it for later use

#### Parameters and Configuration

**Dataset Loading:**

- `load_dataset("SetFit/ag_news")`: Loads the AG News dataset from Hugging Face
  - **Why**: AG News is a standard text classification benchmark with 4 classes (World, Sports, Business, Science/Technology)

**Data Splitting:**

- `train_test_split(test_size=0.2, seed=42)`: Splits training data into train/validation sets
  - `test_size=0.2`: Uses 20% of training data for validation
  - `seed=42`: Ensures reproducible splits
  - **Why**: Validation set is needed for monitoring training progress and preventing overfitting

**Few-Shot Sampling:**
- `sample_dataset(train_set, num_samples=100)`: Samples only 100 examples from the training set
  - `num_samples=100`: Uses just 100 labeled examples
  - **Why**: SetFit's strength is few-shot learning - it can achieve good performance with very little data, making it efficient and cost-effective

**Training Arguments:**
- `output_dir="results"`: Directory to save training outputs
  - **Why**: Stores checkpoints, model weights, and training metadata
- `batch_size=4`: Number of examples per training batch
  - **Why**: Small batch size is appropriate for few-shot learning and helps with memory efficiency
- `num_epochs=1`: Number of training epochs
  - **Why**: SetFit typically converges quickly with few-shot learning, so one epoch is often sufficient

**Model Selection:**

- `SetFitModel.from_pretrained("BAAI/bge-small-en-v1.5")`: Uses BGE (BAAI General Embedding) small model
  - **Why**: BGE models are state-of-the-art sentence transformers optimized for retrieval and similarity tasks, making them excellent backbones for SetFit

**Evaluation:**

- `trainer.evaluate(test_set)`: Evaluates on the held-out test set
  - **Why**: Provides unbiased performance metrics on unseen data

**Model Saving:**

- `model.save_pretrained("setfit-ag-news-20251209")`: Saves the trained model locally
  - **Why**: Allows the model to be loaded later for inference without retraining

---

### `setfit_02.py` - Model Evaluation and Inference

This script loads a trained SetFit model and evaluates it on the test set, providing detailed analysis of predictions.

#### Purpose

The script demonstrates how to:

- Load a previously trained SetFit model from local storage
- Run inference on test examples
- Calculate accuracy manually
- Analyze incorrect predictions with detailed output

#### Parameters and Configuration

**Model Loading:**

- `SetFitModel.from_pretrained("setfit-ag-news-20251209")`: Loads the model from local directory
  - **Why**: Uses the model saved by `setfit_01.py` without needing to retrain

**Label Mapping:**

- `LABEL_MAP = {0: "World", 1: "Sports", 2: "Business", 3: "Science/Technology"}`
  - **Why**: Converts numeric labels to human-readable category names for better interpretability

**Prediction Methods:**

- `model.predict([text])`: Returns the predicted class label
  - **Why**: Gets the most likely class for each text
- `model.predict_proba([text])`: Returns probability distribution over classes
  - **Why**: Provides confidence scores for predictions, useful for understanding model certainty

**Evaluation Approach:**

- Manual iteration through test set with accuracy calculation
  - **Why**: Provides fine-grained control over evaluation, allowing detailed analysis of incorrect predictions including the text, predicted label, true label, and confidence score

---

## Usage

### Training a Model

This took a little over an hour on my M1 mac with 16G or RAM. I ran out of memory, which happens early, a few times. I ended up killing large memory processes and adjusting paramters.

```bash
python setfit_01.py
```

This will:

1. Download the AG News dataset
2. Train a SetFit model with 100 examples
3. Evaluate on the test set
4. Save the model to `setfit-ag-news-20251209/`

### Evaluating a Trained Model

I plan to do a lot more research and review. The first thing that jumped out to me is I agree with the model over dataset more often than I suspected... Reviewing manually is a fun and educational starting point for me.

```bash
python setfit_02.py
```

This will:

1. Load the trained model from `setfit-ag-news-20251209/`
2. Evaluate on the test set
3. Print accuracy and detailed information about incorrect predictions

---

## Dependencies

- `setfit`: SetFit library for few-shot learning
- `datasets`: Hugging Face datasets library
- `sentence-transformers`: For the embedding model backbone

Install with:
```bash
pip install setfit datasets sentence-transformers
```

---

## Bonus: Hugging Face Login Script

### `scripts/huggingface_login.py` - Automated Hugging Face Authentication

This utility script automates the process of logging into Hugging Face, with support for multiple authentication methods. I needed to develpe secret management for work and thought this would be a good learning opportunity. Infisical has the infrastructure I like but will evaluate Azure Key Vault for work. 

#### Purpose

The script provides a convenient way to authenticate with Hugging Face Hub, which is required for:

- Downloading datasets and models
- Uploading models to the Hub
- Accessing private repositories
- Using Hugging Face CLI functions

#### Features

**Automatic Login Detection:**

- `whoami()`: Checks if already logged in
  - **Why**: Avoids unnecessary re-authentication and respects existing sessions

**Multiple Secret Sources:**

1. **Environment Variables** (`.env` file):
   - Looks for `HUGGINGFACE_APIKEY` in `.env`
   - **Why**: Simple local development setup

2. **Infisical Secrets Store**:

   - Falls back to Infisical if `.env` doesn't contain the key
   - Requires configuration in `.env`:
     - `SECRET_CLIENT_ID`
     - `SECRET_CLIENT_SECRET`
     - `SECRET_ENVIRONMENT_SLUG`
     - `SECRET_PATH`
     - `SECRET_PROJECT_SLUG`
   - **Why**: Secure secret management for teams and production environments

**Cache Inspection:**

- `scan_cache_dir()`: Lists all cached Hugging Face repositories
  - **Why**: Helps identify which models/datasets are already downloaded locally

#### Usage

```bash
python scripts/huggingface_login.py
```

#### Configuration

Create a `.env` file in the parent directory with:

```env
# Option 1: Direct API key
HUGGINGFACE_APIKEY=your_token_here

# Option 2: Infisical configuration (if using secrets store)
SECRET_CLIENT_ID=your_client_id
SECRET_CLIENT_SECRET=your_client_secret
SECRET_ENVIRONMENT_SLUG=dev
SECRET_PATH=/
SECRET_PROJECT_SLUG=your_project
```

#### Using with Hugging Face CLI

After running the login script, you can use Hugging Face CLI commands:

```bash
# List your models
huggingface-cli repo list

# Upload a model
huggingface-cli upload username/model-name ./path/to/model

# Download a model
huggingface-cli download username/model-name

# View cache information
huggingface-cli scan-cache
```

The script ensures you're authenticated before running any CLI commands that require authentication.

#### Why This Approach?

1. **Security**: API keys are stored securely (not in code)
2. **Flexibility**: Supports both local development and team environments
3. **Convenience**: Automates the login process
4. **Integration**: Works seamlessly with Hugging Face CLI tools
5. **Error Handling**: Provides clear feedback about authentication status

---

## Model Outputs

After training, the model is saved with the following structure:
```
setfit-ag-news-20251209/
├── model.safetensors          # Model weights
├── config.json                # Model configuration
├── config_setfit.json         # SetFit-specific config
├── tokenizer.json             # Tokenizer files
└── ...
```

This format is compatible with Hugging Face's model loading system and can be easily shared or deployed.

