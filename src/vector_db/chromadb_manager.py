import chromadb
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any
from utils.representative_utterance_selection import mmr_select
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction


class ChromaDBManager:
    """
    Single handler for all interactions with Chroma.

    - Uses a persistent local Chroma DB.
    - Creates/loads two collections:
        1) intents_desc             (docs: f"{intent}\\n{description}")
        2) intent_utterances_repr   (docs: representative utterances selected via MMR)

    - Both collections use the SAME metadata schema:
        {
            "intent_name": str,
            "description": str,
            "representative_utterances": List[str]
        }

    - Provides:
        * build_from_dataframe(df, intent_to_desc, k_rep)
        * search(query, top_k_per_collection)
    """

    def __init__(
        self,
        chroma_dir: Path | str,
        model_name: str,
        intents_collection_name: str,
        utterances_collection_name: str,
    ) -> None:

        self.chroma_dir = Path(chroma_dir)
        self.chroma_dir.mkdir(parents=True, exist_ok=True)

        # Persistent client
        self.client = chromadb.PersistentClient(path=str(self.chroma_dir))

        # Embedding function (SentenceTransformers)
        self.embedding_fn = SentenceTransformerEmbeddingFunction(model_name=model_name)

        # Collection names
        self.intents_collection_name = intents_collection_name
        self.utterances_collection_name = utterances_collection_name

        # Get or create collections with the same embedding function
        self.intents_coll = self.client.get_or_create_collection(
            name=self.intents_collection_name,
            embedding_function=self.embedding_fn,
        )
        self.utterances_coll = self.client.get_or_create_collection(
            name=self.utterances_collection_name,
            embedding_function=self.embedding_fn,
        )

    def build_from_dataframe(
        self,
        df: pd.DataFrame,
        intent_to_desc: Dict[str, str],
        k_rep: int = 20,
    ) -> None:
        """
        Build/refresh both collections from:
        - df: pandas DataFrame with columns [intent, utterance]
        - intent_to_desc: {intent: description}
        - k_rep: number of representative utterances per intent (MMR)
        """

        # Filter DF to intents we actually have descriptions for
        df = df[df["intent"].isin(intent_to_desc.keys())].copy()

        # 1) Compute representative utterances per intent using MMR
        print("[ChromaDBHandler] Selecting representative utterances with MMR...")
        rep_map = self._compute_representative_utterances(df, k_rep=k_rep)

        # 2) Upsert the intent + description docs
        print("[ChromaDBHandler] Upserting intent-description collection...")
        self._upsert_intent_descriptions(intent_to_desc, rep_map)

        # 3) Upsert the representative utterances docs
        print("[ChromaDBHandler] Upserting representative utterances collection...")
        self._upsert_representative_utterances(rep_map, intent_to_desc)

        print("[ChromaDBHandler] Build complete.")

    def _compute_representative_utterances(
        self,
        df: pd.DataFrame,
        k_rep: int,
    ) -> Dict[str, List[str]]:
        """
        Returns:
            {intent_name: [rep_utt_1, ..., rep_utt_k]}
        """
        result: Dict[str, List[str]] = {}
        all_intents = sorted(df["intent"].unique())

        for intent in all_intents:
            subset = df[df["intent"] == intent]
            utterances = subset["text"].tolist()
            if not utterances:
                result[intent] = []
                continue

            # embed and run MMR
            embeddings = np.array(self.embedding_fn(utterances), dtype=np.float32)
            k = min(k_rep, len(utterances))
            idxs = mmr_select(embeddings, k=k, lambda_mult=0.5)

            reps = [utterances[i] for i in idxs]
            result[intent] = reps
            print(f"  - intent '{intent}': selected {len(reps)} reps out of {len(utterances)}")

        return result

    def _upsert_intent_descriptions(
        self,
        intent_to_desc: Dict[str, str],
        rep_map: Dict[str, List[str]],
    ) -> None:
        ids, docs, metas = [], [], []

        for intent, desc in intent_to_desc.items():
            reps = rep_map.get(intent, [])
            doc_id = f"intent::{intent}"
            text = f"{intent}\n{desc}"
            ids.append(doc_id)
            docs.append(text)
            metas.append(
                {
                    "intent_name": intent,
                    "description": desc,
                    "representative_utterances": "<utt>".join(reps),
                }
            )

        if ids:
            self.intents_coll.upsert(
                ids=ids,
                documents=docs,
                metadatas=metas,
            )
        print(f"→ Upserted {len(ids)} intent-description docs.")

    def _upsert_representative_utterances(
        self,
        rep_map: Dict[str, List[str]],
        intent_to_desc: Dict[str, str],
    ) -> None:
        ids, docs, metas = [], [], []

        for intent, reps in rep_map.items():
            desc = intent_to_desc.get(intent, "")
            for j, utt in enumerate(reps):
                doc_id = f"utt::{intent}::{j}"
                ids.append(doc_id)
                docs.append(utt)
                metas.append(
                    {
                        "intent_name": intent,
                        "description": desc,
                        "representative_utterances": "<utt>".join(reps),
                    }
                )

        if ids:
            self.utterances_coll.upsert(
                ids=ids,
                documents=docs,
                metadatas=metas,
            )
        print(f"→ Upserted {len(ids)} representative utterance docs.")
        
    def search(
        self,
        query: str,
        top_k_per_collection: int = 20,
        top_k_intents: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve candidates from both collections, then aggregate by intent_name.

        Fusion rule:
        - Count how many times each intent_name appears in the retrieved results
            across both collections.
        - Sort intents by descending count (and by best rank as a tie-breaker).
        - Return top_k_intents entries with metadata.

        Returns a list of dicts:
            {
                "intent_name": str,
                "count": int,  # how many hits across both collections
                "description": str,
                "representative_utterances": List[str],
                "collections_hit": List[str]  # e.g. ["intents_desc", "intent_utterances_repr"]
            }
        """

        # 1) Query intents_desc
        res_intents = self.intents_coll.query(
            query_texts=[query],
            n_results=top_k_per_collection,
        )

        # 2) Query intent_utterances_repr
        res_utts = self.utterances_coll.query(
            query_texts=[query],
            n_results=top_k_per_collection,
        )

        # Helper to flatten chroma results into (collection, rank, metadata)
        def _extract(res, coll_name: str) -> List[Dict[str, Any]]:
            items: List[Dict[str, Any]] = []
            if not res or not res.get("ids"):
                return items

            ids = res["ids"][0]
            docs = res["documents"][0]
            metas = res["metadatas"][0]

            for rank, (doc_id, doc, meta) in enumerate(zip(ids, docs, metas), start=1):
                items.append(
                    {
                        "collection": coll_name,
                        "rank": rank,
                        "id": doc_id,
                        "document": doc,
                        "metadata": meta,
                    }
                )
            return items

        flat_results: List[Dict[str, Any]] = []
        flat_results.extend(_extract(res_intents, "intents_desc"))
        flat_results.extend(_extract(res_utts, "intent_utterances_repr"))

        # 3) Aggregate by intent_name with counts
        intent_stats: Dict[str, Dict[str, Any]] = {}

        for item in flat_results:
            meta = item["metadata"] or {}
            intent_name = meta.get("intent_name")
            if not intent_name:
                continue

            if intent_name not in intent_stats:
                intent_stats[intent_name] = {
                    "intent_name": intent_name,
                    "count": 0,
                    "description": meta.get("description", ""),
                    "representative_utterances": meta.get("representative_utterances", []),
                    "best_rank": item["rank"],                  # best (smallest) rank seen
                    "collections_hit": {item["collection"]},    # set to avoid duplicates
                }

            stats = intent_stats[intent_name]
            stats["count"] += 1
            stats["best_rank"] = min(stats["best_rank"], item["rank"])
            stats["collections_hit"].add(item["collection"])

        # 4) Convert sets to lists and sort intents by:
        #    - descending count
        #    - ascending best_rank (tie-breaker)
        results: List[Dict[str, Any]] = []
        for intent_name, stats in intent_stats.items():
            stats["collections_hit"] = sorted(stats["collections_hit"])
            results.append(stats)

        results.sort(key=lambda s: (-s["count"], s["best_rank"]))

        # 5) Return top_k_intents
        return results[:top_k_intents]