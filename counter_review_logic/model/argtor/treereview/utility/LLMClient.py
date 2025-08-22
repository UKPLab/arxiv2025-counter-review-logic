import os

import datetime
import json
import sqlite3
import time
from typing import Optional, Dict, List, Any
import openai
import tiktoken
from dotenv import load_dotenv
from langchain.llms.base import LLM
from langchain_core.callbacks import CallbackManagerForLLMRun

load_dotenv()

class BaseClient:
    def __init__(
            self,
            base_url: str = "https://generativelanguage.googleapis.com/v1beta/openai/",
            api_key: str = None,
            default_model: str = "gemini-2.0-flash",
            default_temperature: Optional[float] = 0.0,
            default_top_p: Optional[float] = 1.0,
            default_max_tokens: Optional[int] = 32768,
            default_frequency_penalty: Optional[float] = 0.0,
            default_presence_penalty: Optional[float] = 0.0,
            system_prompt: Optional[str] = None,
            cache_db_path: Optional[str] = None,
            max_retires: Optional[int] = 4,
            fingerprint: Optional[str] = None
    ):
        self.base_url = base_url
        self.api_key = api_key
        self.openai_client = openai.OpenAI(base_url=base_url, api_key=api_key)
        self.system_prompt = system_prompt
        self.fingerprint = fingerprint or datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

        if cache_db_path:
            self.cache_db = sqlite3.connect(cache_db_path)
            self._init_db()
        else:
            self.cache_db = None

        self.default_params = {
            "model": default_model or "gpt-4o",
            "temperature": default_temperature,
            "top_p": default_top_p,
            "max_tokens": default_max_tokens,
            "frequency_penalty": default_frequency_penalty,
            "presence_penalty": default_presence_penalty,
        }
        self.max_retires = max(0, max_retires)
        self.tokenizers: Dict[str, tiktoken.Encoding] = {}
        self.reset()

    @staticmethod
    def from_config(config):
        return LLMClient(
            base_url=config["base_url"],
            api_key=config["api_key"],
            default_model=config["default_model"] if "default_model" in config else "gpt-4o",
            default_temperature=config["default_temperature"] if "default_temperature" in config else 0.0,
            default_top_p=config["default_top_p"] if "default_top_p" in config else 1.0,
            default_max_tokens=config["default_max_tokens"] if "default_max_tokens" in config else 1024,
            default_frequency_penalty=config["default_frequency_penalty"] if "default_frequency_penalty" in config else 0.0,
            default_presence_penalty=config["default_presence_penalty"] if "default_presence_penalty" in config else 0.0,
            cache_db_path=config["cache_db_path"] if "cache_db_path" in config else None,
            max_retires=config["max_retires"] if "max_retires" in config else 4,
            fingerprint=config["fingerprint"] if "fingerprint" in config else None,
        )

    def reset(self):
        self.messages = []
        if self.system_prompt is not None:
            self.messages.append({"role": "system", "content": self.system_prompt})

    def _init_db(self):
        cur = self.cache_db.cursor()

        cur.execute("""
        CREATE TABLE IF NOT EXISTS chat_cache (
            fingerprint TEXT,
            model TEXT NOT NULL,
            messages_json TEXT NOT NULL,
            temperature REAL NOT NULL,
            top_p REAL NOT NULL,
            max_tokens INTEGER NOT NULL,
            frequency_penalty REAL NOT NULL,
            presence_penalty REAL NOT NULL,
            response_json TEXT NOT NULL,
            prompt_tokens INTEGER NOT NULL,
            completion_tokens INTEGER NOT NULL,
            total_tokens INTEGER NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )""")

        cur.execute("""
        CREATE TABLE IF NOT EXISTS token_usage (
            fingerprint TEXT,
            model TEXT NOT NULL,
            prompt_tokens INTEGER NOT NULL,
            completion_tokens INTEGER NOT NULL,
            total_tokens INTEGER NOT NULL,
            is_cached INTEGER NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )""")

        self.cache_db.commit()

    def _get_tokenizer(self, model: str) -> tiktoken.Encoding:
        if model not in self.tokenizers:
            try:
                self.tokenizers[model] = tiktoken.encoding_for_model(model)
            except KeyError:
                self.tokenizers[model] = tiktoken.get_encoding("cl100k_base")
        return self.tokenizers[model]

    def estimate_messages_num_tokens(self, messages: List[Dict], model: str) -> int:
        tokenizer = self._get_tokenizer(model)
        tokens_per_message = 3
        tokens_per_name = 1
        num_tokens = 0
        for message in messages:
            num_tokens += tokens_per_message
            for key, value in message.items():
                if key in ["role", "content"]:
                    num_tokens += len(tokenizer.encode(value))
                if key == "name" or key == "role":
                    num_tokens += tokens_per_name
        num_tokens += 3
        return num_tokens

    def chat(self,
             prompt: str,
             model: Optional[str] = None,
             temperature: Optional[float] = None,
             top_p: Optional[float] = None,
             max_tokens: Optional[int] = None,
             frequency_penalty: Optional[float] = None,
             presence_penalty: Optional[float] = None,
             use_cache: bool = True,
             max_retries: Optional[int] = None
             ):

        self.messages.append({"role": "user", "content": prompt})
        response = self._run_prompt_basic(
            messages=self.messages,
            model=model,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            use_cache=use_cache,
            max_retries=max_retries
        )
        response_message = response["choices"][0]["message"]
        self.messages.append(response_message.copy())
        return response_message["content"]

    def generate(
            self,
            prompt: str,
            use_system_prompt: bool = False,
            system_prompt: Optional[str] = None,
            model: Optional[str] = None,
            temperature: Optional[float] = None,
            top_p: Optional[float] = None,
            max_tokens: Optional[int] = None,
            frequency_penalty: Optional[float] = None,
            presence_penalty: Optional[float] = None,
            use_cache: bool = True,
            max_retries: Optional[int] = None
    ) -> str:
        system_prompt = system_prompt or self.system_prompt
        messages = []
        if use_system_prompt:
            assert system_prompt is not None
            messages.append({"role": "system", "content": system_prompt})

        messages.append({"role": "user", "content": prompt})
        response = self._run_prompt_basic(
            messages=messages,
            model=model,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            use_cache=use_cache,
            max_retries=max_retries
        )
        return response["choices"][0]["message"]["content"]

    def _run_prompt_basic(
            self,
            messages: List[Dict],
            model: Optional[str] = None,
            temperature: Optional[float] = None,
            top_p: Optional[float] = None,
            max_tokens: Optional[int] = None,
            frequency_penalty: Optional[float] = None,
            presence_penalty: Optional[float] = None,
            use_cache: bool = True,
            max_retries: Optional[int] = None
    ) -> Dict:
        params = {
            "model": model or self.default_params["model"],
            "temperature": temperature or self.default_params["temperature"],
            "top_p": top_p or self.default_params["top_p"],
            "max_tokens": max_tokens or self.default_params["max_tokens"],
            "frequency_penalty": frequency_penalty or self.default_params["frequency_penalty"],
            "presence_penalty": presence_penalty or self.default_params["presence_penalty"],
        }

        max_retries = max_retries or self.max_retires

        db_keyvals = params.copy()
        messages_json = json.dumps(messages, sort_keys=True)
        db_keyvals["messages_json"] = messages_json
        cache_json = None
        token_usage_keyvals = None
        if use_cache and self.cache_db and params["temperature"] == 0:
            cur = self.cache_db.cursor()
            select_keyvals = db_keyvals.copy()
            select_keyvals["messages_token_count"] = self.estimate_messages_num_tokens(messages, params["model"])
            dbrecs = cur.execute(
                """select response_json, model, prompt_tokens, completion_tokens, total_tokens from chat_cache
                where
                model = :model and
                messages_json = :messages_json and
                temperature = :temperature and
                ((:messages_token_count+max_tokens) > total_tokens or max_tokens = :max_tokens) and
                total_tokens <= (:messages_token_count+:max_tokens) and
                top_p = :top_p and
                frequency_penalty = :frequency_penalty and
                presence_penalty = :presence_penalty
                """,
                select_keyvals
            ).fetchall()

            if len(dbrecs) >= 1:
                cache_json = dbrecs[0][0]
                token_usage_keyvals = {
                    "fingerprint": self.fingerprint,
                    "model": dbrecs[0][1],
                    "prompt_tokens": dbrecs[0][2],
                    "completion_tokens": dbrecs[0][3],
                    "total_tokens": dbrecs[0][4],
                    "is_cached": 1,
                    "created_at": datetime.datetime.timestamp(datetime.datetime.utcnow())
                }

        if cache_json is None:
            model_keyvals = db_keyvals.copy()
            del model_keyvals["messages_json"]
            model_keyvals["messages"] = messages
            call_params = model_keyvals.copy()
            resp = None
            if max_retries > 0:
                while resp is None and max_retries >= 0:
                    max_retries -= 1
                    try:
                        resp = self.openai_client.chat.completions.create(**call_params).model_dump()
                    except openai.RateLimitError:
                        time.sleep(60)
                    except openai.APIError as e:
                        if e.code == 502:
                            time.sleep(10)
                        elif e.code in ["RequestTimeOut", 400]:
                            time.sleep(10)
                        else:
                            time.sleep(10)
                    except openai.APITimeoutError as e:
                        time.sleep(10)
                    except openai.BadRequestError as e:
                        pass
            else:
                resp = self.openai_client.chat.completions.create(**call_params).model_dump()

            insert_keyvals = db_keyvals.copy()
            cache_json = json.dumps(resp)
            insert_keyvals["fingerprint"] = self.fingerprint
            insert_keyvals["response_json"] = cache_json
            insert_keyvals["created_at"] = datetime.datetime.timestamp(datetime.datetime.utcnow())
            insert_keyvals["prompt_tokens"] = resp["usage"]["prompt_tokens"]
            insert_keyvals["completion_tokens"] = resp["usage"]["completion_tokens"]
            insert_keyvals["total_tokens"] = resp["usage"]["total_tokens"]
            token_usage_keyvals = {
                "fingerprint": self.fingerprint,
                "model": params["model"],
                "prompt_tokens": resp["usage"]["prompt_tokens"],
                "completion_tokens": resp["usage"]["completion_tokens"],
                "total_tokens": resp["usage"]["total_tokens"],
                "is_cached": 0,
                "created_at": datetime.datetime.timestamp(datetime.datetime.utcnow()),
            }
            if use_cache and self.cache_db:
                cur = self.cache_db.cursor()
                cur.execute(
                    """INSERT INTO chat_cache ( fingerprint, model,  messages_json,  temperature,  top_p,  max_tokens,  frequency_penalty,  presence_penalty,  response_json,  created_at,  prompt_tokens, completion_tokens, total_tokens)
                       VALUES                      (:fingerprint, :model, :messages_json, :temperature, :top_p, :max_tokens, :frequency_penalty, :presence_penalty, :response_json, :created_at, :prompt_tokens, :completion_tokens, :total_tokens)""",
                    insert_keyvals,
                )
                self.cache_db.commit()
        if token_usage_keyvals is not None:
            if use_cache and self.cache_db:
                cur = self.cache_db.cursor()
                cur.execute(
                    """INSERT INTO token_usage ( fingerprint, model,  prompt_tokens,  completion_tokens,  total_tokens,  is_cached, created_at)
                       VALUES                      (:fingerprint, :model,  :prompt_tokens,  :completion_tokens,  :total_tokens, :is_cached, :created_at)""",
                    token_usage_keyvals,
                )
                self.cache_db.commit()
        resp = json.loads(cache_json)
        return resp

    def get_token_usage(self, model: Optional[str] = None) -> Dict:
        cur = self.cache_db.cursor()

        query = "SELECT SUM(prompt_tokens), SUM(completion_tokens) FROM token_usage"
        params = ()

        if model:
            query += " WHERE model = ?"
            params = (model,)

        result = cur.execute(query, params).fetchone()
        return {
            "prompt_tokens": result[0] or 0,
            "completion_tokens": result[1] or 0,
            "total_tokens": (result[0] or 0) + (result[1] or 0)
        }

    def __enter__(self):
        self.cache_db.__enter__()
        return self

    def __exit__(self, *args, **kwargs):
        self.cache_db.__exit__(*args, **kwargs)

    def close(self):
        self.cache_db.close()

    def _call(
            self,
            prompt: str,
            stop: Optional[List[str]] = None,
            run_manager: Optional[CallbackManagerForLLMRun] = None,
            **kwargs: Any,
    ) -> str:
        return self.generate(prompt)

# LangChain Decorator
class LLMClient(LLM):
    @property
    def _llm_type(self) -> str:
        pass

    def __init__(self):
        super().__init__()

    def main(self, prompt):
        client = BaseClient(
            api_key=os.environ["GEMINI_API_KEY"],
            max_retires=10
        )
        resp = client.generate(prompt=prompt, use_system_prompt=False, temperature=0, max_tokens=32768)
        return resp
    def _call(
            self,
            prompt: str,
            stop: Optional[List[str]] = None,
            run_manager: Optional[CallbackManagerForLLMRun] = None,
            **kwargs: Any,
    ) -> str:
        return self.main(prompt)
