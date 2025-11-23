from vector_db import ChromaDBManager
from config import Config

if __name__ == '__main__':
    config = Config()
    chroma_manager = ChromaDBManager(
        chroma_dir=config.vector_db_dir,
        model_name=config.encoder_model_name,
        intents_collection_name=config.intent_collection_name,
        utterances_collection_name=config.utterance_collection_name,
    )
    query = 'Where do I apply my coupon code at checkout?'
    result = chroma_manager.search(
        query=query,
        top_k_per_collection=20,
        top_k_intents=10,
    )
    
    for r in result:
        print(f"intent={r['intent_name']} | count={r['count']} | best_rank={r['best_rank']}\ndescription={r['description']}")
        print()
    
    