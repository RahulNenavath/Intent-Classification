# Intent-Classification

A compact intent classification project that provides an end-to-end pipeline for:
- preparing a vector database (Chroma),
- setting up the LLM environment,
- running inference to classify intents.

## Quick start (recommended)

These instructions assume you are on a Unix-like machine (Linux or macOS).  
Adjust shell commands for Windows (PowerShell / conda on Windows).

---

### 1. Clone the repository

```bash
git clone https://github.com/RahulNenavath/Intent-Classification.git
cd Intent-Classification
```

### 2. Create a conda environment
If you don’t have Miniconda installed, use the helper script:
```bash
chmod +x setup_miniconda.sh
./setup_miniconda.sh
```
Then create and configure the project environment:
```bash
chmod +x setup.sh
./setup.sh
```

### 3. Install Python dependencies and the package (editable)
From the repository root:
```bash
pip install -e .
```
This installs all dependencies (from pyproject.toml + requirements.txt) and registers the intent_classification package in editable development mode.

### 4. Explore the source layout (optional)
The Python package lives inside src/:
```bash
ls src
```
All python -m ... commands below must be run from the repository root once the package is installed.

### 5. Set up the Chroma vector DB
This initializes and populates your persistent Chroma database using your utterance dataset + intent descriptions.
From the project root:
```bash
python -m vector_db.setup_db
```
This will:
- create/connect to the persistent Chroma DB directory (configured in src/config.py)
- embed utterances
- compute representative utterances per intent (k-center greedy)
- upsert into Chroma collections

If Chroma requires environment variables (e.g., storage path), configure them beforehand. See src/config.py.

### 6. Set up the LLM runtime
The project includes a setup script for the local or remote LLM backend (MLX / Groq / other).
From the project root:
```bash
python -m llm.setup_llm
```
This script configures the LLM model, keys, and runtime settings used by the inference pipeline.
See src/config.py for model name and settings.

### 7. Run inference (RAG-based intent classification)
Execute the main inference module:
```bash
python -m inference
```
This will:
- accept a user utterance
- retrieve candidate intents + representative utterances from Chroma
- pass them into the LLM for RAG-based classification
- output the predicted intent
