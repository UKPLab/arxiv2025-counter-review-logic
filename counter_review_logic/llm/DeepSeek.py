import json
import os
import time
from datetime import datetime
import logging
from typing import Callable

import openai
from langchain_community.callbacks import get_openai_callback
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import ConfigurableField
from langchain_openai import ChatOpenAI

from . import ChatLLM


class DeepSeekChatLLM(ChatLLM):
    def __init__(self,
                 name,
                 model: str,
                 cost_cache_dir: str|None = None,
                 prompt_fp: str | None = None,
                 prompt: ChatPromptTemplate | None = None,
                 post_processor: Callable | None = None,
                 pre_processor: Callable | None = None,
                 multimodal=False,
                 config:dict=None):
        super().__init__(name)

        # logging
        self.logger = logging.getLogger(name)

        # check prompts
        assert prompt_fp is None or prompt is None, "cannot provide both a prompt and a file path for a prompt at the same time"

        # model name
        self.model = model
        self.config = config

        # cost logging
        if cost_cache_dir is None and "COST_CACHE_DIR" in os.environ:
            cost_cache_dir = os.environ["COST_CACHE_DIR"].replace("\"", "")

        assert cost_cache_dir is not None and os.path.exists(
            cost_cache_dir), f"the provided cache dir does not exist: {cost_cache_dir}"
        self._cost_cache_file = cost_cache_dir + "/" + self.model.replace("/", "_") + "_" + \
                                datetime.now().strftime("%Y-%m-%d-%S-%f")[:-3] + ".json"
        self.costs = {
            "total_tokens": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_cost_usd": 0
        }

        self.multimodal = multimodal

        self.post_processor = lambda x: x if post_processor is None else post_processor
        self.pre_processor = lambda x: x if pre_processor is None else pre_processor

        self._setup(prompt_fp, prompt)

    def _setup(self, prompt_fp, prompt):
        self.logger.info("Connecting to API in chat mode with:")
        self.logger.info(f"DEPLOYMENT = {self.model}")
        self.logger.info(f"ENDPOINT = https://api.deepseek.com")

        # config client
        if self.config is None:
            self.config = {
                "max_tokens": 16384 # default max toknes
            }

        self.llm = ChatOpenAI(
            model_name="deepseek-chat",
            api_key=os.environ["DEEPSEEK_API_KEY"],
            base_url="https://api.deepseek.com",
            **self.config
        )
        self.llm.configurable_fields(
            temperature=ConfigurableField(id="temperature", name="LLM Temperature", description="The temp of the LLM"),
            max_tokens=ConfigurableField(id="max_tokens", name="LLM Max Tokens", description="The max tokens of the LLM"),
        )

        # load prompt
        if prompt_fp:
            self.load_prompt(prompt_fp)
        elif prompt:
            self.set_prompt(prompt)
        else:
            self.prompt_template = None

    def _log_costs(self, cb_result):
        self.costs["total_tokens"] += cb_result.total_tokens
        self.costs["prompt_tokens"] += cb_result.prompt_tokens
        self.costs["completion_tokens"] += cb_result.completion_tokens
        self.costs["total_cost_usd"] += cb_result.total_cost

        # print
        self.logger.debug("COSTS:" + str(self.costs["total_cost_usd"]))
        self.logger.debug("PROMPTED:" + str(self.costs["prompt_tokens"]))
        self.logger.debug("COMPLETED:" + str(self.costs["completion_tokens"]))

        # cache
        with open(self._cost_cache_file, "w+") as f:
            json.dump(self.costs, f)

    def set_config(self, config):
        self.config = config

        if self.llm:
            self.llm = self.llm.with_config(**config)

    def add_config(self, config):
        self.config = {**self.config, **config}

        self.set_config(self.config)

    def _post_process(self, outputs):
        return [self.post_processor(o) for o in outputs]

    def _pre_process(self, inputs):
        return [self.pre_processor(i) for i in inputs]

    def call_openai_fixed_input(self, inputs, config=None):
        # set config for these calls if provided
        if config:
            llm = self.llm.with_config(**config)
        else:
            llm = self.llm

        single_instance = type(inputs) != list
        if single_instance:
            inputs = [inputs]

        with get_openai_callback() as cb:
            # https://medium.com/@hey_16878/efficient-batch-processing-with-langchain-and-openai-overcoming-ratelimiterror-daa9de4bbd8b
            results = []

            max_retries = 10
            delay_increment = 5

            if len(inputs) == 0:
                return []

            # pre process
            inputs = self._pre_process(inputs)

            # set batch size
            batch_size = min(20, len(inputs))

            for i in range(0, len(inputs), batch_size):
                batch = inputs[i: i + batch_size]

                retries = 0
                while retries <= max_retries:
                    try:
                        result = llm.batch(batch)
                        results.extend(map(lambda x: x.content, result))

                        break  # Exit the retry loop once successful
                    except (openai.RateLimitError, openai.APIConnectionError) as connection_error:
                        delay = (retries + 1) * delay_increment
                        self.logger.warning(f"OPENAI WARNING: {connection_error}. Retrying in {delay} seconds...")
                        time.sleep(delay)
                        retries += 1

                        if retries > max_retries:
                            self.logger.error(
                                f"ERROR: Max retries reached for batch starting at index {i}. Raising Error!")
                            results.extend([None for i in range(len(batch))])

                self._log_costs(cb)

        results = self._post_process(results)

        if single_instance:
            return results[0]
        else:
            return results

    def _call_openai(self, inputs, config=None, prompt=None):
        # set config for these calls if provided
        if config:
            llm = self.llm.with_config(**config)
        else:
            llm = self.llm

        if prompt:
            prompt_template = prompt
        else:
            prompt_template = self.prompt_template

        chain = prompt_template | llm

        single_instance = type(inputs) != list
        if single_instance:
            inputs = [inputs]

        with get_openai_callback() as cb:
            # https://medium.com/@hey_16878/efficient-batch-processing-with-langchain-and-openai-overcoming-ratelimiterror-daa9de4bbd8b
            results = []

            max_retries = 10
            delay_increment = 5

            if len(inputs) == 0:
                return []

            # pre process
            inputs = self._pre_process(inputs)

            # set batch size
            batch_size = min(20, len(inputs))

            for i in range(0, len(inputs), batch_size):
                batch = inputs[i: i + batch_size]

                retries = 0
                while retries <= max_retries:
                    try:
                        result = chain.batch(batch)
                        results.extend(map(lambda x: x.content, result))

                        break  # Exit the retry loop once successful
                    except (openai.RateLimitError, openai.APIConnectionError) as connection_error:
                        delay = (retries + 1) * delay_increment
                        self.logger.warning(f"OPENAI WARNING: {connection_error}. Retrying in {delay} seconds...")
                        time.sleep(delay)
                        retries += 1

                        if retries > max_retries:
                            self.logger.error(
                                f"ERROR: Max retries reached for batch starting at index {i}. Raising Error!")
                            results.extend([None for i in range(len(batch))])

                self._log_costs(cb)

        results = self._post_process(results)

        if single_instance:
            return results[0]
        else:
            return results

    async def _call_openai_async(self, inputs, config=None, prompt=None):
        # set config for these calls if provided
        if config:
            llm = self.llm.with_config(**config)
        else:
            llm = self.llm

        if prompt:
            prompt_template = prompt
        else:
            prompt_template = self.prompt_template

        chain = prompt_template | llm

        single_instance = type(inputs) != list
        if single_instance:
            inputs = [inputs]

        with get_openai_callback() as cb:
            # https://medium.com/@hey_16878/efficient-batch-processing-with-langchain-and-openai-overcoming-ratelimiterror-daa9de4bbd8b
            results = []

            max_retries = 10
            delay_increment = 5

            if len(inputs) == 0:
                return []

            # pre process
            inputs = self._pre_process(inputs)

            # set batch size
            batch_size = min(20, len(inputs))

            for i in range(0, len(inputs), batch_size):
                batch = inputs[i: i + batch_size]

                retries = 0
                while retries <= max_retries:
                    try:
                        result = await chain.abatch(batch)
                        results.extend(map(lambda x: x.content, result))

                        break  # Exit the retry loop once successful
                    except (openai.RateLimitError, openai.APIConnectionError) as connection_error:
                        delay = (retries + 1) * delay_increment
                        self.logger.warning(f"OPENAI WARNING: {connection_error}. Retrying in {delay} seconds...")
                        time.sleep(delay)
                        retries += 1

                        if retries > max_retries:
                            self.logger.error(
                                f"ERROR: Max retries reached for batch starting at index {i}. Raising Error!")
                            results.extend([None for i in range(len(batch))])

                self._log_costs(cb)

        results = self._post_process(results)

        if single_instance:
            return results[0]
        else:
            return results

    def __call__(self, *args, **kwargs):
        return self._call_openai(args[0], kwargs, prompt=kwargs["prompt"] if "prompt" in kwargs else None)

    def async_call(self, *args, **kwargs):
        return self._call_openai_async(args[0], kwargs, prompt=kwargs["prompt"] if "prompt" in kwargs else None)

    def batch(self, *args, **kwargs):
        raise NotImplementedError("not yet implemented")

    def get_llm(self):
        return self.llm