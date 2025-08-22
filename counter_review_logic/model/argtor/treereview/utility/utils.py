import re
import json
import tiktoken
from langchain_core.output_parsers import JsonOutputParser

def count_token(text:str, model_name="gpt-4o", encoding_name="o200k_base"):
    if model_name:
        encoding_name = tiktoken.encoding_name_for_model(model_name)
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(text))
    return num_tokens

def load_json_object(text):
    try:
        json_text = text[text.index("{"):text.rindex("}") + 1]
        obj = json.loads(json_text, strict=False)
    except json.JSONDecodeError:
        try:
            json_text = text[text.index("{"):text.rindex("}") + 1]
            pattern = r'(?<!\\)\\(?![\\"])'
            processed_text = re.sub(pattern, r'\\\\', json_text)
            obj = json.loads(processed_text, strict=False)
        except json.JSONDecodeError:
            parser = JsonOutputParser()
            obj = parser.parse(text)
    return obj

def load_json_array(text):
    try:
        json_text = text[text.index("["):text.rindex("]") + 1]
        arr = json.loads(json_text, strict=False)
    except json.JSONDecodeError:
        try:
            json_text = text[text.index("["):text.rindex("]") + 1]
            pattern = r'(?<!\\)\\(?![\\"])'
            processed_text = re.sub(pattern, r'\\\\', json_text)
            arr = json.loads(processed_text, strict=False)
        except json.JSONDecodeError:
            parser = JsonOutputParser()
            arr = parser.parse(text)
    return arr