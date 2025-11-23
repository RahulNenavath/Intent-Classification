from typing import List, Dict
from pydantic import BaseModel, Field, ValidationError

class BaseLLMHandler:
    """
    Base wrapper around a Groq LLM.
    - loads model + tokenizer in constructor
    - provides .format(user_prompt) to build chat-style prompt
    - provides .invoke(user_prompt) to actually generate text
    """

    def __init__(
        self,
        system_prompt: str,
        model_name: str,
        max_tokens: int = 512,
        temperature: float = 0.7,
        api_key: str | None = None,
    ) -> None:
        self.system_prompt = system_prompt
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.temperature = temperature
        
        self.client = Groq(api_key=api_key or os.getenv("GROQ_API_KEY"))

    def format(self, user_prompt: str) -> List[Dict[str, str]]:
        """
        Build a chat prompt for Groq (system + user).
        """
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        return messages

    def invoke(self, user_prompt: str) -> str:
        """
        Call Groq chat completions and return raw text.
        """
        messages = self.format(user_prompt)
        resp = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            stream=False,
        )
        return resp.choices[0].message.content