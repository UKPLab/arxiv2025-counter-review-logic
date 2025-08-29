import math
import os
import re
import sys
from pathlib import Path

import torch
import transformers


from ...data import Paper, Review
from ...framework import AutomaticReviewGenerator
from ...llm import ChatLLM, parse_llm_output_as_single_json, approximately_truncate

class Reviewer2ARGtor(AutomaticReviewGenerator):
    """
     Uses the Reviewer2 proposed in below work.

     @misc{gao2024reviewer2,
      title={Reviewer2: Optimizing Review Generation Through Prompt Generation},
      author={Zhaolin Gao and Kianté Brantley and Thorsten Joachims},
      year={2024},
      eprint={2402.10886},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
    }
    """

    def __init__(self,
                 name: str,
                 llm: ChatLLM=None,
                 prompt_base_path: str | Path = None):
        super().__init__(name)

        self.formatting_llm = llm

        self.prompt_base_path = prompt_base_path
        if prompt_base_path is None and "PROMPT_DIR" in os.environ:
            self.prompt_base_path = Path(os.environ["PROMPT_DIR"].replace("\"", "")) / "argtor"
        elif prompt_base_path is None:
            raise ValueError("Prompt base path is not provided.")

        assert self.prompt_base_path.exists(), f"the prompt base path needs to exist. It does not: {self.prompt_base_path}."

        self.formatting_llm.load_prompt(self.prompt_base_path / "reviewer2" / "refine_format.txt")

        self._setup()

    def build_generator(self,
            model, tokenizer, temperature=0.7, top_p=0.7, top_k=50, max_new_tokens=1024, min_new_tokens=64,
            repetition_penalty=1.13
    ):
        def response(prompt):
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

            output = model.generate(
                **inputs,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                max_new_tokens=max_new_tokens,
                min_new_tokens=min_new_tokens,
                repetition_penalty=repetition_penalty,
                do_sample=True,
            )

            out = tokenizer.decode(output[0], skip_special_tokens=True)

            try:
                out = out.split(prompt.lstrip("<s>"))[1].strip()
            except:
                out = []

            return out

        return response

    def _setup(self):
        # based on https://github.com/ZhaolinGao/Reviewer2/blob/main/demo.py
        SEQ_LEN = 32768

        # not replacing llama attention (skipped that part)

        # ==========Prompt Model==========
        # Set RoPE scaling factor
        config = transformers.AutoConfig.from_pretrained('GitBag/Reviewer2_Mp')
        orig_rope_scaling = getattr(config, "rope_scaling", None)
        if orig_rope_scaling is None:
            orig_rope_scaling = {"factor": 1}
        orig_rope_scaling_factor = orig_rope_scaling["factor"] if "factor" in orig_rope_scaling.keys() else 1
        orig_ctx_len = getattr(config, "max_position_embeddings", None)
        if orig_ctx_len:
            orig_ctx_len *= orig_rope_scaling_factor
            if SEQ_LEN > orig_ctx_len:
                scaling_factor = float(math.ceil(SEQ_LEN / orig_ctx_len))
                config.rope_scaling = {"type": "linear", "factor": scaling_factor}
                print(f'rope scaling factor {scaling_factor}')

        # prep model
        prompt_tokenizer = transformers.AutoTokenizer.from_pretrained(
            'GitBag/Reviewer2_Mp',
            model_max_length=SEQ_LEN if SEQ_LEN > orig_ctx_len else orig_ctx_len,
            padding_side="right",
            use_fast=False,
        )

        prompt_model = transformers.AutoModelForCausalLM.from_pretrained(
            'GitBag/Reviewer2_Mp',
            config=config,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )

        prompt_model.resize_token_embeddings(32001)
        prompt_model.eval()
        if torch.__version__ >= "2" and sys.platform != "win32":
            prompt_model = torch.compile(prompt_model)

        self.prompt_generator = self.build_generator(prompt_model, prompt_tokenizer)

        # Set RoPE scaling factor
        config = transformers.AutoConfig.from_pretrained('GitBag/Reviewer2_Mr')
        orig_rope_scaling = getattr(config, "rope_scaling", None)
        if orig_rope_scaling is None:
            orig_rope_scaling = {"factor": 1}
        orig_rope_scaling_factor = orig_rope_scaling["factor"] if "factor" in orig_rope_scaling.keys() else 1
        orig_ctx_len = getattr(config, "max_position_embeddings", None)
        if orig_ctx_len:
            orig_ctx_len *= orig_rope_scaling_factor
            if SEQ_LEN > orig_ctx_len:
                scaling_factor = float(math.ceil(SEQ_LEN / orig_ctx_len))
                config.rope_scaling = {"type": "linear", "factor": scaling_factor}
                print(f'rope scaling factor {scaling_factor}')

        # prep model
        reviewer_tokenizer = transformers.AutoTokenizer.from_pretrained(
            'GitBag/Reviewer2_Mr',
            model_max_length=SEQ_LEN if SEQ_LEN > orig_ctx_len else orig_ctx_len,
            padding_side="right",
            use_fast=False,
        )
        self.reviewer_tokenizer = reviewer_tokenizer

        review_model = transformers.AutoModelForCausalLM.from_pretrained(
            'GitBag/Reviewer2_Mr',
            config=config,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )

        review_model.resize_token_embeddings(32001)
        review_model.eval()
        if torch.__version__ >= "2" and sys.platform != "win32":
            review_model = torch.compile(review_model)

        # ==========Review Generation==========
        review_generator = self.build_generator(review_model, reviewer_tokenizer)
        self.review_generator = review_generator

    def prompt_review_gen_model(self, paper_content:str, gen_prompt:str):
        prompt_Llama_2 = (
            "[INST] <<SYS>>\n"
            "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n"
            "If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.\n"
            "<</SYS>>\nRead the following paper carefully:\n{paper_content}\n\n\n"
            "Your task is to compose a high-quality review of the paper submitted to a top-tier conference.\n"
            "Your review should contain the answers to the following questions:\n{prompt_gen}\n"
            "\nWrite your review into following section:\n{format}\n"
            "[/INST]"
        )
        prompt_dict = {
            'paper_content': paper_content,
            'prompt_gen': gen_prompt,
            'format': '\n'.join(['Summary Of The Paper', 'Strengths And Weaknesses', 'Questions', 'Limitations'])
        }
        prompt = prompt_Llama_2.format_map(prompt_dict)

        return prompt

    def prompt_prompt_gen_model(self, paper_content: str):
        prompt_Llama_2 = (
            "[INST] <<SYS>>\n"
            "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n"
            "If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.\n"
            "<</SYS>>\nRead the following paper carefully:\n{paper_content}\n\n\n"
            "Your task is to construct a list of questions about the paper for the reviewer to answer.\n"
            "\nThe reviewer should answer in the following format:\n{format}\n"
            "[/INST]"
        )
        prompt_dict = {
            'paper_content': paper_content,
            'format': '\n'.join(['Summary Of The Paper', 'Strengths And Weaknesses'])
        }
        prompt = prompt_Llama_2.format_map(prompt_dict)

        return prompt

    def format(self, raw_output, venue_config):
        if self.formatting_llm is None:
            raise ValueError("No formatting LLM provided.")

        raw_output = str(raw_output)

        # deal with enormous amounts of whitespaces (appears commonly)
        pattern = r'[\s\u00A0\u1680\u180E\u2000-\u200A\u2028\u2029\u202F\u205F\u3000]{3,}'
        raw_output = re.sub(pattern, '  ', raw_output)

        truncated = approximately_truncate(raw_output, self.formatting_llm, margin=8192)
        for i in range(3):  # try up to 3 times to parse
            self.formatting_llm.load_prompt(self.prompt_base_path / "reviewer2" / "refine_format.txt")
            res = self.formatting_llm({
                "faulty_review_report": truncated,
                "template": venue_config["review_template"],
                "scores": venue_config["review_scores"]
            })
            parsed = parse_llm_output_as_single_json(res)[1]

            if parsed is not None:
                break

        return parsed

    def _truncate_paper(self, paper_md: str):
        tok = self.reviewer_tokenizer

        encoded = tok.encode(paper_md)
        max_len = int(70000 * 0.8)
        if len(encoded) > max_len:
            paper_md = tok.decode(encoded[:max_len])

        return paper_md

    def run(self, paper: Paper, **config) -> Review:
        paper_text = paper.without_appendix().md
        paper_text = self._truncate_paper(paper_text)

        venue_config = paper.meta["venue_config"]

        gen_prompt = self.prompt_generator(self.prompt_prompt_gen_model(paper_text))
        result = self.review_generator(self.prompt_review_gen_model(paper_text, gen_prompt))
        parsed = self.format(result, venue_config)

        if parsed is None or len(parsed) == 0:
            return Review(
                sections={"main": result},
                scores={"overall": 0},
                main_section="main",
                overall_score="overall",
                meta={"error": "failed parsing the review output"}
            )

        summary_field = venue_config["template_field_semantics"]["summary"]

        if parsed is not None and "review" in parsed and "scores" in parsed and parsed.get("review") and parsed.get("scores"):
            return Review(
                sections=parsed.get("review", {}),
                scores=parsed.get("scores", {}),
                overall_score=venue_config["overall_score_name"],
                main_section=summary_field if summary_field in parsed.get("review", {}) else list(parsed.get("review").keys())[0] if parsed.get("review", {}) else None,
                meta={
                    "raw_text": result
                }
            )

        return None