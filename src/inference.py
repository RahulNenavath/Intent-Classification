import json
import tomllib
from typing import Any, Dict, List, Tuple, Optional

from config import Config
from mlx_lm import load
from vector_db.chromadb_manager import ChromaDBManager
from model.llm_manager import BaseLLMHandler
from pydantic import BaseModel, Field, ValidationError, field_validator

config = Config()


class IntentResponse(BaseModel):
	intent: str = Field(..., description=f"Chosen intent name or '{config.unknown_intent_name}'")

	@field_validator("intent")
	@classmethod
	def _strip_and_require(cls, v: str) -> str:
		v = (v or "").strip()
		if not v:
			raise ValueError("intent cannot be empty")
		return v


class IntentClassifier:
    def __init__(
        self,
        cfg: Config,
        model: Any,
        tokenizer: Any,
        system_prompt: str,
        user_template: str,
        candidate_item_template: Optional[str] = None,
        example_bullet_template: Optional[str] = None,
        max_tokens: int = 128,
        temperature: float = 0.2,
        ) -> None:
        
        self.cfg = cfg or Config()
        self.model = model
        self.tokenizer = tokenizer
        self.system_prompt = system_prompt.format(unknown_intent_name=self.cfg.unknown_intent_name)
        self.user_template = user_template
        self.candidate_item_template = candidate_item_template
        self.example_bullet_template = example_bullet_template
        self.llm = BaseLLMHandler(
            system_prompt=self.system_prompt,
			model=model,
			tokenizer=tokenizer,
			max_tokens=max_tokens,
			temperature=temperature,
		)
    
    def _parse_json_object(self, raw: str) -> Dict[str, Any]:
        text = raw.strip()
        if text.startswith("```"):
            lines = text.splitlines()
            if lines and lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].startswith("```"):
                lines = lines[:-1]
            text = "\n".join(lines).strip()
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end < start:
            raise ValueError("No JSON object found in model output")
        return json.loads(text[start : end + 1])
    
    def _render_candidate_blocks(self, hits: List[Dict[str, Any]]) -> str:
        if not self.candidate_item_template or not self.example_bullet_template:
            return "\n".join(h.get("intent_name", "") for h in hits)
        blocks: List[str] = []
        for h in hits:
            name = h.get("intent_name", "")
            desc = h.get("description", "")
            reps_raw = h.get("representative_utterances", "")
            if isinstance(reps_raw, list):
                examples = [str(x).strip() for x in reps_raw if str(x).strip()]
            else:
                examples = [s.strip() for s in str(reps_raw).split("<utt>") if s.strip()]
            examples_block = "\n".join(self.example_bullet_template.format(example=e) for e in examples[:10])
            block = self.candidate_item_template.format(intent_name=name, description=desc, examples_block=examples_block)
            blocks.append(block.strip())
        return "\n".join(blocks)
    
    def _format_user_prompt(self, user_utterance: str, hits: List[Dict[str, Any]]) -> str:
        candidates_block = self._render_candidate_blocks(hits)
        return self.user_template.format(
            user_utterance=user_utterance, 
            candidates_block=candidates_block,
            unknown_intent_name=self.cfg.unknown_intent_name
            )
    
    def classify_intent(self, user_utterance: str) -> str:
        db = ChromaDBManager(
			chroma_dir=self.cfg.vector_db_dir,
			model_name=self.cfg.encoder_model_name,
			intents_collection_name=self.cfg.intent_collection_name,
			utterances_collection_name=self.cfg.utterance_collection_name,
		)
        
        hits = db.search(
			query=user_utterance,
			top_k_per_collection=self.cfg.candidate_pool,
			top_k_intents=self.cfg.top_k,
		)
        
        user_prompt = self._format_user_prompt(user_utterance, hits)
        raw = self.llm.invoke(user_prompt)
        try:
            obj = self._parse_json_object(raw)
            parsed = IntentResponse(**obj)
            intent = parsed.intent
        except (ValidationError, Exception):
            intent = self.cfg.unknown_intent_name
        return intent
    
def get_inference_classifier() -> IntentClassifier:

    cfg = Config()
    model_path = cfg.model_dir / cfg.mlx_model_repo_id.split("/")[-1]
    model, tokenizer = load(model_path)
    
    prompts_path = cfg.model_dir / "prompts.toml"
    with open(prompts_path, "rb") as f:
        prompts_data = tomllib.load(f)
        rag_prompts = prompts_data.get('rag_intent_classification', {})
    
    system_prompt = rag_prompts.get("system", "").strip()
    user_template = rag_prompts.get("user_template", "").strip()
    candidate_item_template = rag_prompts.get("candidate_item_template", "").strip()
    example_bullet_template = rag_prompts.get("example_bullet_template", "").strip()
    
    classifier = IntentClassifier(
        cfg=cfg,
        model=model,
        tokenizer=tokenizer,
        system_prompt=system_prompt,
        user_template=user_template,
        candidate_item_template=candidate_item_template,
        example_bullet_template=example_bullet_template,
    )
    
    return classifier
    
    
if __name__ == "__main__":

    classifier = get_inference_classifier()
    test_utterance = "i am looking to buy the new ipad air with 512gb storage but looks like it is out of stock all the time! can you tell me when it will be available?"
    intent = classifier.classify_intent(test_utterance)
    print(f"Predicted intent: {intent}")