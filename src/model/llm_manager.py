from config import Config
from typing import List, Dict
from mlx_lm import load, generate

class BaseLLMHandler:
    """
    Base wrapper around a local MLX LLM.
    - please first use setup_llm.py to download the MLX LLM locally
    - builds a chat-style prompt via tokenizer's chat template when available
    - generates text locally using mlx_lm.generate
    """

    def __init__(
        self,
        system_prompt: str,
        model: object,
        tokenizer: object,
        max_tokens: int = 512,
        temperature: float = 0.7
    ) -> None:
        self.system_prompt = system_prompt
        self.max_tokens = max_tokens
        self.temperature = temperature
        
        self.model = model
        self.tokenizer = tokenizer

    def _build_prompt(self, user_prompt: str) -> str:
        messages: List[Dict[str, str]] = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        # Prefer chat template if available
        apply_tmpl = getattr(self.tokenizer, "apply_chat_template", None)
        if callable(apply_tmpl):
            try:
                # Use tokenizer's template if set; else fall back below
                return apply_tmpl(messages, add_generation_prompt=True)
            except Exception:
                pass
        # Fallback to simple concatenation
        return (
            f"System: {self.system_prompt}\n"
            f"User: {user_prompt}\n"
            f"Assistant:"
        )

    def invoke(self, user_prompt: str) -> str:
        prompt = self._build_prompt(user_prompt)
        out_text = generate(
            self.model,
            self.tokenizer,
            prompt=prompt,
            max_tokens=self.max_tokens,
            verbose=False,
        )
        return out_text
    
if __name__ == "__main__":
    config = Config()
    model_path = config.model_dir / config.mlx_model_repo_id.split("/")[-1]
    model, tokenizer = load(model_path)
    
    handler = BaseLLMHandler(
        system_prompt="You are a concise assistant.",
        model=model,
        tokenizer=tokenizer,
        max_tokens=32,
        temperature=0.7
    )
    print(handler.invoke("Say hello in one short sentence."))