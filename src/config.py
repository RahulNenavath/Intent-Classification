from pathlib import Path


class Config:
    def __init__(self) -> None:
        self.project_dir: Path = Path(__file__).resolve().parent.parent
        self.data_dir: Path = self.project_dir / "Data"
        self.project_src_dir: Path = self.project_dir / "src"
        self.vector_db_dir: Path = self.project_src_dir / "vector_db"
        self.model_dir: Path = self.project_src_dir / "model"

        # Ensure they exist (except src which should already be present)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.vector_db_dir.mkdir(parents=True, exist_ok=True)
        
        self.encoder_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
        self.intent_collection_name : str = "intent_desc"
        self.utterance_collection_name : str = "intent_utterances_repr"
        self.representative_utterances_k: int = 10
        self.candidate_pool: int = 20
        self.top_k: int = 10
        
        self.mlx_model_repo_id: str = "mlx-community/Llama-3.2-3B-Instruct-8bit"
        self.unknown_intent_name: str = "UNKNOWN"

    def as_dict(self) -> dict:
        """Return paths as a simple serializable dictionary of strings."""
        return {
            "project_dir": str(self.project_dir),
            "data_dir": str(self.data_dir),
            "project_src_dir": str(self.project_src_dir),
            "vector_db_dir": str(self.vector_db_dir),
        }


__all__ = ["Config"]