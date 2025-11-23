import pandas as pd
import json
from pathlib import Path
from config import Config
from chromadb_manager import ChromaDBManager

def setup_vector_db_from_data(
    intent_utterance_tsv: Path, 
    intent_description_json: Path, 
    vector_db_dir: Path, 
    encoder_model_name: str,
    intent_collection_name: str,
    utterance_collection_name: str,
    k_rep: int
    ) -> None:
    """
    Sets up the Chroma vector database from the provided data files.

    - intent_utterance_tsv: TSV file with columns ["intent", "utterance"]
    - intent_description_json: JSON file mapping intent names to descriptions
    - vector_db_dir: Directory to store the persistent Chroma DB
    - k_rep: Number of representative utterances to select per intent
    - encoder_model_name: SentenceTransformer model name for embeddings
    - intent_collection_name: Name of the intents collection in ChromaDB
    - utterance_collection_name: Name of the utterances collection in ChromaDB
    """

    # Load data
    df_utterances = pd.read_csv(intent_utterance_tsv, sep="\t")
    with open(intent_description_json, "r") as f:
        intent_to_desc = json.load(f)

    # Initialize ChromaDBManager
    chroma_manager = ChromaDBManager(
        chroma_dir=vector_db_dir,
        model_name=encoder_model_name,
        intents_collection_name=intent_collection_name,
        utterances_collection_name=utterance_collection_name,
    )

    # Build the database from the dataframe
    chroma_manager.build_from_dataframe(df_utterances, intent_to_desc, k_rep)
    
    
if __name__ == "__main__":
    config = Config()
    intent_utterance_tsv = config.data_dir / "intent_utterances.tsv"
    intent_description_json = config.data_dir / "intent_descriptions.json"
    setup_vector_db_from_data(
        intent_utterance_tsv=intent_utterance_tsv,
        intent_description_json=intent_description_json,
        vector_db_dir=config.vector_db_dir,
        encoder_model_name=config.encoder_model_name,
        intent_collection_name=config.intent_collection_name,
        utterance_collection_name=config.utterance_collection_name,
        k_rep=config.representative_utterances_k,
    )