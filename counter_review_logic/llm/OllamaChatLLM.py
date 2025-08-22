import copy
import os
import time
import logging
from typing import Callable

from httpx import ConnectTimeout
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

from . import ChatLLM


class OllamaChatLLM(ChatLLM):
    def __init__(self,
                 name,
                 model: str,
                 prompt_fp: str | None = None,
                 prompt: ChatPromptTemplate | None = None,
                 post_processor: Callable | None = None,
                 pre_processor: Callable | None = None,
                 config: dict = None):
        super().__init__(name)

        # logging
        self.logger = logging.getLogger(name)

        # check prompts
        assert prompt_fp is None or prompt is None, "cannot provide both a prompt and a file path for a prompt at the same time"

        # model name
        self.model = model
        self.config = config
        self.default_config = {
            "num_ctx": 8192  # default value that should be covered by all models
        }

        self.post_processor = lambda x: x if post_processor is None else post_processor
        self.pre_processor = lambda x: x if pre_processor is None else pre_processor

        self._setup(prompt_fp, prompt)

        # rate limiting
        self.last_call = None
        self.min_interval_ms = 3000

    def set_config(self, config):
        self.config = config

        if self.llm:
            self.llm = ChatOllama(
                model=self.model,
                base_url=os.environ["OLLAMA_CHAT_ENDPOINT"].replace('"', ''),
                **self.config
            )

    def add_config(self, config):
        self.config = {**self.config, **config}

        self.set_config(self.config)

    def _setup(self, prompt_fp, prompt):
        self.logger.info("Connecting to Ollama Chat models:")
        self.logger.info(f"MODEL = {self.model}")
        self.logger.info(f"ENDPOINT = {os.environ['OLLAMA_CHAT_ENDPOINT']}")

        # config client
        if self.config is None:
            self.config = copy.deepcopy(self.default_config)

        self.llm = ChatOllama(
            model=self.model,
            base_url=os.environ["OLLAMA_CHAT_ENDPOINT"].replace('"', ''),
            **self.config
        )

        # load prompt
        if prompt_fp:
            self.load_prompt(prompt_fp)
        elif prompt:
            self.set_prompt(prompt)
        else:
            self.prompt_template = None

    def _post_process(self, outputs):
        return [self.post_processor(o) for o in outputs]

    def _pre_process(self, inputs):
        return [self.pre_processor(i) for i in inputs]

    def _call_ollama(self, inputs, config=None):
        # set config for these calls if provided
        if config:
            # create new client with new config
            llm = ChatOllama(
                model=self.model,
                base_url=os.environ["OLLAMA_CHAT_ENDPOINT"].replace('"', ''),
                **config
            )
        else:
            llm = self.llm
            config = self.config

        chain = self.prompt_template | llm

        single_instance = type(inputs) != list
        if single_instance:
            inputs = [inputs]

        if len(inputs) == 0:
            return []

        results = []

        max_retries = 10
        delay_increment = 5

        # pre process
        inputs = self._pre_process(inputs)

        # set batch size
        batch_size = min(20, len(inputs))

        for i in range(0, len(inputs), batch_size):
            batch = inputs[i: i + batch_size]

            retries = 0
            while retries <= max_retries:
                # to avoid rate issues, wait for minimal amount of time needed
                current_time = time.time() * 1000
                elapsed = (current_time - self.last_call) if self.last_call else self.min_interval_ms

                if elapsed < self.min_interval_ms:
                    logging.debug(f"Waiting for {self.min_interval_ms - elapsed} ms before next ollama call...")
                    time.sleep((self.min_interval_ms - elapsed) / 1000)

                # make the actual call
                try:
                    result = chain.batch(batch)
                    results.extend(map(lambda x: x.content, result))

                    self.last_call = time.time() * 1000

                    break  # Exit the retry loop once successful
                except (ConnectionError, ConnectTimeout) as connection_error:
                    self.last_call = time.time() * 1000

                    delay = (retries + 1) * delay_increment
                    self.logger.warning(f"OLLAMA WARNING: {connection_error}. Retrying in {delay} seconds...")
                    time.sleep(delay)
                    retries += 1

                    if retries > max_retries:
                        self.logger.error(
                            f"ERROR: Max retries reached for batch starting at index {i}. Raising Error!")
                        results.extend([None for i in range(len(batch))])

        results = self._post_process(results)

        if single_instance:
            return results[0]
        else:
            return results

    def __call__(self, *args, **kwargs):
        return self._call_ollama(args[0], kwargs)

    def batch(self, *args, **kwargs):
        raise NotImplementedError("not yet implemented")

    def get_llm(self):
        return self.llm
