from pathlib import Path
from typing import Tuple
import shutil
from config import Config
from mlx_lm import load

from huggingface_hub import snapshot_download


def setup_mlx_model(
	repo_id: str,
	models_dir: Path = None,
	force_download: bool = False,
	allow_patterns: Tuple[str, ...] = ("*.json", "*.model", "*.txt", "*.py", "*.mlx", "*.safetensors")
) -> None:
    
    repo_folder_name = repo_id.split("/")[-1]
    model_folder = models_dir / repo_folder_name
    model_folder.mkdir(parents=True, exist_ok=True)
    
    # If force_download, remove any existing contents (carefully).
    if force_download and model_folder.exists():
        for item in model_folder.iterdir():
            if item.is_file() or item.is_symlink():
                item.unlink()
            else:
                shutil.rmtree(item)
    
    # Download snapshot (this will use HF cache internally). We download into a
    # temporary snapshot folder; if target_dir is empty we can just keep it.
    print(f"[setup_mlx_model] Downloading snapshot for {repo_id}...")
    snapshot_path = snapshot_download(
        repo_id=repo_id,
        allow_patterns=list(allow_patterns),
    )
    
    snapshot_path = Path(snapshot_path)
    
    # Copy snapshot contents into target_dir if not already populated.
    if not any(model_folder.iterdir()):
        print(f"[setup_mlx_model] Populating target directory: {model_folder}")
        for item in snapshot_path.iterdir():
            dest = model_folder / item.name
            if item.is_dir():
                shutil.copytree(item, dest)
            else:
                shutil.copy2(item, dest)
    else:
        print(f"[setup_mlx_model] Target directory already populated: {model_folder}")

__all__ = ["setup_mlx_model"]

if __name__ == "__main__":
    config = Config()
    setup_mlx_model(
        repo_id=config.mlx_model_repo_id,
        models_dir=config.model_dir,
    )
