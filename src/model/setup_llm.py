"""Utilities for preparing local MLX LLM models.

Provides a helper `setup_mlx_model` that downloads (if needed) a Llama 3 3B
instruct variant compatible with `mlx_lm` and returns a ready (model, tokenizer)
tuple for downstream generation managers.

Usage example:

	from model.setup_llm import setup_mlx_model
	model, tokenizer, model_path = setup_mlx_model()

By default it uses the MLX community port of Llama 3 3B instruct. You can
override with another MLX-compatible repo.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple
import shutil
import os
from config import Config

from huggingface_hub import snapshot_download
from mlx_lm import load

DEFAULT_MLX_LLAMA_3B_REPO = "mlx-community/Llama-3.2-3B-Instruct"


def setup_mlx_model(
	repo_id: str = DEFAULT_MLX_LLAMA_3B_REPO,
	target_dir: Path | str | None = None,
	force_download: bool = False,
	allow_patterns: Tuple[str, ...] = ("*.json", "*.model", "*.txt", "*.py", "*.mlx", "*.safetensors"),
) -> Tuple[object, object, Path]:
	"""Download (if needed) a MLX-format Llama 3 3B model locally and load it.

	Args:
		repo_id: Hugging Face repo id of the MLX model.
		target_dir: Optional path to store a project-local copy. If None, a
			`Data/models/<repo_name>` folder under project root is used (root
			inferred relative to this file).
		force_download: If True, re-download snapshot even if already cached.
		allow_patterns: File patterns to include from the repo.

	Returns:
		(model, tokenizer, model_path) where `model_path` is the local directory
		containing the downloaded snapshot used for loading.
	"""
	# Determine default target directory inside project root.
	if target_dir is None:
		project_root = Path(__file__).resolve().parents[2]  # .../IntentClassification
		models_root = project_root / "Data" / "models"
		models_root.mkdir(parents=True, exist_ok=True)
		repo_folder_name = repo_id.split("/")[-1]
		target_dir = models_root / repo_folder_name
	else:
		target_dir = Path(target_dir)
		target_dir.mkdir(parents=True, exist_ok=True)

	# If force_download, remove any existing contents (carefully).
	if force_download and target_dir.exists():
		for item in target_dir.iterdir():
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
	if not any(target_dir.iterdir()):
		print(f"[setup_mlx_model] Populating target directory: {target_dir}")
		for item in snapshot_path.iterdir():
			dest = target_dir / item.name
			if item.is_dir():
				shutil.copytree(item, dest)
			else:
				shutil.copy2(item, dest)
	else:
		print(f"[setup_mlx_model] Target directory already populated: {target_dir}")

	# Load with mlx_lm (load() accepts repo id or path; we prefer path now).
	print(f"[setup_mlx_model] Loading MLX model from {target_dir} ...")
	model, tokenizer = load(str(target_dir))
	print("[setup_mlx_model] Model + tokenizer ready.")
	return model, tokenizer, target_dir


__all__ = ["setup_mlx_model", "DEFAULT_MLX_LLAMA_3B_REPO"]

