import os, json, re, asyncio
from typing import Optional, Type

from openai import OpenAI
from pydantic import BaseModel
from deepeval.models.base_model import DeepEvalBaseLLM


def _extract_first_json(text: str) -> Optional[str]:
    """모델이 JSON 앞뒤로 말을 섞어도 첫 번째 { ... } 덩어리만 뽑아냅니다."""
    if not text:
        return None
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    cand = text[start:end + 1].strip()
    try:
        json.loads(cand)
        return cand
    except Exception:
        return None


class GroqChatLLM(DeepEvalBaseLLM):
    """
    DeepEval이 schema(BaseModel)를 넘겨줄 때:
    - Groq JSON mode(response_format=json_object)로 '유효 JSON'을 강제
    - JSON 추출/리트라이
    - schema(**data)로 검증 후 반환
    """
    def __init__(
        self,
        model_name: str = "llama-3.3-70b-versatile",
        api_key: Optional[str] = None,
        base_url: str = "https://api.groq.com/openai/v1",
        max_tokens: int = 900,
    ):
        self.model_name = model_name
        self.max_tokens = max_tokens

        api_key = api_key or os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY 환경변수를 설정해주세요.")

        self.client = OpenAI(api_key=api_key, base_url=base_url)

        self.system_prompt = (
            "You are a strict evaluation engine.\n"
            "You MUST output ONLY valid JSON.\n"
            "- No prose, no markdown, no code fences.\n"
            "- Output must start with { and end with }.\n"
            "- Use double quotes for all keys and strings.\n"
        )

    def load_model(self):
        return self.client

    def _call_chat(self, messages, response_format=None, temperature=0.0):
        kwargs = dict(
            model=self.model_name,
            messages=messages,
            temperature=temperature,
            max_tokens=self.max_tokens,
        )
        if response_format is not None:
            kwargs["response_format"] = response_format  # Groq supports json_object/json_schema :contentReference[oaicite:2]{index=2}

        # openai 라이브러리 버전에 따라 response_format이 막히면 fallback
        try:
            return self.client.chat.completions.create(**kwargs)
        except TypeError:
            kwargs.pop("response_format", None)
            return self.client.chat.completions.create(**kwargs)

    def generate(self, prompt: str, schema: Optional[Type[BaseModel]] = None):
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt},
        ]

        if schema is not None:
            # 1) JSON mode (valid JSON) :contentReference[oaicite:3]{index=3}
            try:
                resp = self._call_chat(messages, response_format={"type": "json_object"}, temperature=0.0)
                raw = (resp.choices[0].message.content or "").strip()
            except Exception:
                raw = ""

            json_str = _extract_first_json(raw) or raw

            # 2) 파싱 실패 시 리트라이(더 강하게)
            try:
                data = json.loads(json_str)
                return schema(**data)
            except Exception:
                retry_messages = [
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": prompt + "\n\nREMINDER: Output ONLY valid JSON."},
                ]
                resp2 = self._call_chat(retry_messages, response_format={"type": "json_object"}, temperature=0.0)
                raw2 = (resp2.choices[0].message.content or "").strip()
                json_str2 = _extract_first_json(raw2) or raw2
                data2 = json.loads(json_str2)
                return schema(**data2)

        resp = self._call_chat(messages, response_format=None, temperature=0.0)
        return (resp.choices[0].message.content or "").strip()

    async def a_generate(self, prompt: str, schema: Optional[Type[BaseModel]] = None):
        return await asyncio.to_thread(self.generate, prompt, schema)

    def get_model_name(self) -> str:
        return f"Groq({self.model_name})"


class GroqDeepEvalLLM(DeepEvalBaseLLM):
    def __init__(
        self,
        model: str = "llama-3.3-70b-versatile",
        api_key: str | None = None,
        max_completion_tokens: int = 2048,
        temperature: float = 0.0,
        use_json_object_mode: bool = True,
    ):
        self.client = OpenAI(
            api_key=api_key or os.getenv("GROQ_API_KEY"),
            base_url="https://api.groq.com/openai/v1",
        )
        self.model = model
        self.max_completion_tokens = max_completion_tokens
        self.temperature = temperature
        self.use_json_object_mode = use_json_object_mode

        self.system = (
            "You are a strict evaluator. Follow instructions exactly. "
            "If JSON is requested, output ONLY valid JSON (no extra text)."
        )

    def load_model(self):
        return self  # DeepEval 요구사항 충족용 (실제 로드는 client가 담당)

    def generate(self, prompt: str) -> str:
        kwargs = dict(
            model=self.model,
            messages=[
                {"role": "system", "content": self.system},
                {"role": "user", "content": prompt},
            ],
            temperature=self.temperature,
            max_completion_tokens=self.max_completion_tokens,
        )

        if self.use_json_object_mode:
            kwargs["response_format"] = {"type": "json_object"}

        resp = self.client.chat.completions.create(**kwargs)
        return resp.choices[0].message.content

    async def a_generate(self, prompt: str) -> str:
        return await asyncio.to_thread(self.generate, prompt)

    def get_model_name(self) -> str:
        return f"groq:{self.model}"