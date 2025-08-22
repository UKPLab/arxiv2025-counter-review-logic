import json
import logging
import re
from pathlib import Path
from typing import Callable

import tiktoken
from langchain_core.prompts import ChatPromptTemplate


class ChatLLM():
    def __init__(self, name):
        self.name = name
        self.prompt_template = None
        self.model = None

    @staticmethod
    def _load_chat_prompt(fp:str|Path):
        if isinstance(fp, str):
            fp = Path(fp)

        try:
            is_json = True
            with fp.open("r") as f:
                msgs = json.load(f)
        except Exception:
            is_json = False
            msgs = eval(fp.read_text())

        tmplt = []
        if is_json:
            for m in msgs:
                assert m["actor"] in ["human", "system"]
                tmplt += [(m["actor"], m["msg"])]
        else:
            for m in msgs:
                tmplt += [(m[0], m[1])]

        return ChatPromptTemplate.from_messages(tmplt)

    def load_prompt(self, prompt_fp: str|Path):
        self.prompt_template = self._load_chat_prompt(prompt_fp)

    def set_prompt(self, prompt: ChatPromptTemplate):
        self.prompt_template = prompt

    def set_config(self, config):
        raise NotImplementedError("Abstract base class does not implement set_config method")

    def add_config(self, config):
        raise NotImplementedError("Abstract base class does not implement add_config method")

    def set_pre_processor(self, fun: Callable):
        self.pre_processor = fun

    def set_post_processor(self, fun: Callable):
        self.post_processor = fun

    def __call__(self, *args, **kwargs):
        raise NotImplementedError("Abstract base class does not implement call method")

    def batch(self, *args, **kwargs):
        raise NotImplementedError("Abstract base class does not implement batch method")


def parse_llm_output_as_single_json(text):
    parsed_results = []

    def clean_json_string(json_str):
        # Regex pattern to match valid escape sequences
        valid_escapes = re.compile(r'\\["\\n]|\\u[0-9a-fA-F]{4}')

        def replace_invalid_escapes(match):
            text = match.group(0)
            # If the matched text is a valid escape, return it unchanged
            if valid_escapes.match(text):
                return text
            # Otherwise, replace single \ with \\ (escaped backslash)
            return text.replace('\\', '\\\\')

        # Replace all standalone \ not followed by a valid escape
        return re.sub(r'\\.', replace_invalid_escapes, json_str)

    try:
        match = clean_json_string(text)
        json_data = json.loads(match)

        return text, json_data
    except json.JSONDecodeError:
        pass

    pattern = r'```json\n(.*?)```'
    matches = re.findall(pattern, text, re.DOTALL)

    if not matches:
        # Recovery mode: Try to guess JSON boundaries based on {}
        guessed_pattern = r'\{.*?\}'
        matches = re.findall(guessed_pattern, text, re.DOTALL)

    for match in matches:
        try:
            match = clean_json_string(match)
            json_data = json.loads(match)

            parsed_results.append(json_data)
        except json.JSONDecodeError as e:
            print("Invalid JSON found:", match, e)
            continue

    if len(parsed_results) == 0:
        return text, None

    return text, parsed_results[0]



def get_num_ctx_of_llm(model_name):
    """
    Based on ruler https://github.com/NVIDIA/RULER
    :param model_name:
    :return:
    """
    if "deepseek-r1:14b" in model_name:
        return 128 * 1000
    elif "deepseek-r1:70b" in model_name:
        return 64 * 1000 # technically it's 128k, but we get a memory error
    elif "deepseekv3" == model_name:
        return 64 * 1000
    elif "llama" in model_name:
        return 64 * 1000
    elif "qwen" in model_name:
        return 32 * 1000
    elif "gemma" in model_name:
        return 64 * 1000
    elif "phi4" in model_name:
        return 16 * 1000
    elif "gpt-4o-mini" in model_name:
        return 128*1000
    elif "gpt-4o" in model_name:
        return 64*1000
    elif "gpt-4.1" in model_name:
        return 30*1000
    else:
        return None


def approximately_truncate(text:str, llm:ChatLLM, margin=2048):
    """
    Truncate the text to fit within the context window of the LLM, considering a margin. The number of tokens
    is approximately calculated using byte pair encoding. Approximately the margin number of tokens will
    remain in the estimated context window of the model.

    Do consider that the margin should cover: the rest of the prompt AND the response length.

    :param text:
    :param llm:
    :param margin:
    :return:
    """
    # get context size of model
    max_tokens = get_num_ctx_of_llm(llm.model)
    max_tiktoken_char_chunksize = 500*1000
    if max_tokens is None:
        logging.info(f"No truncation; model {llm.model.name} has no defined context size limit.")
        print(f"No truncation; model {llm.model.name} has no defined context size limit.")
        return text

    # heuristic based on chars, no need to tokenize
    if len(text) < max_tokens - margin:
        logging.info(f"No truncation; by heuristic, text is shorter than {max_tokens - margin} characters.")
        print(f"No truncation; by heuristic, text is shorter than {max_tokens - margin} characters.")
        return text

    # otherwise tokenize
    enc = tiktoken.get_encoding("o200k_base")

    if len(text) > max_tiktoken_char_chunksize:
        tokens = []
        for i in range(0, len(text), max_tiktoken_char_chunksize):
            chunk = text[i:i + max_tiktoken_char_chunksize]
            tokens += enc.encode(chunk)
    else:
        tokens = enc.encode(text)

    token_count = len(tokens)

    # if tokens are within max tokens plus margin, leave the text
    if token_count <= max_tokens - margin:
        print(f"No truncation necessary since it has only {token_count} tokens on {len(text)} chars, which is less than the limit of {max_tokens - margin} tokens.")
        return text

    truncated_text = enc.decode(tokens[:max_tokens - margin])

    logging.info(f"Truncated text from {len(text)} to {len(truncated_text)} characters.")
    print(f"Truncated text from {len(text)} to {len(truncated_text)} characters.")

    return truncated_text