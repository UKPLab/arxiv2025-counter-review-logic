import itertools
import json
import logging
import os
from pathlib import Path

import tiktoken
from langchain_core.prompts import ChatPromptTemplate

from ....data import Review, Paper
from ....framework import AutomaticReviewGenerator
from ....llm import parse_llm_output_as_single_json, ChatLLM, OpenAiChatLLM
from .agents import MultiAgentGroup, LlmChatBot


class MARG(AutomaticReviewGenerator):
    """
    This class generates reviews according to the MARG paper.
    Paper: https://arxiv.org/pdf/2401.04259
    Code: https://github.com/allenai/marg-reviewer/tree/master

    Adapation from their code:

    * The function reviewgen_v26_specialized_multi_agent in reviewer_worker/run_reviewgen.py is the main function
    to replicate the MARG paper's method using multiple specialized agents.
    * It relies on passed "paper_chunks" retrieved from get_paper_chunks in the same file. This is based on grobid-parsed
      paper text structured into sections. We need to simulate this.

    * Prompts are provided under review_worker/review_prompts.json


    """

    def __init__(self,
                 name: str,
                 model_name: str|None = "gpt-4o-mini",
                 local_model: ChatLLM = None,
                 prompt_dir: str | Path | None = None,
                 review_format_prompt_name: str | Path | None = None,
                 config: dict = None):
        super().__init__(name)
        self.logger = logging.getLogger(name)

        self.marg_config = config
        self.model_name = model_name
        self.local_model = local_model

        assert local_model is not None or model_name is not None, "Either a model name or a local model must be provided"
        assert local_model is None or model_name is None, "Cannot provide both a model name and a local model"

        self.prompt_conf = None
        if prompt_dir is None:
            self.prompt_dir = Path(os.environ.get("PROMPT_DIR", Path(__file__).resolve().parent / "prompts")) / "argtor"
        elif isinstance(prompt_dir, str):
            self.prompt_dir = Path(prompt_dir)

        self.review_format_prompt_path = review_format_prompt_name
        if self.review_format_prompt_path is None:
            self.review_format_prompt_path = self.prompt_dir / "marg" / "format_default_review_prompt.json"

        assert self.prompt_dir.exists(), f"Prompt directory {self.prompt_dir} does not exist"

        self._setup()

    def _setup(self):
        # load marg_config
        if self.marg_config is None:
            self.marg_config = {
                "paper_chunk_size": 4096,
                "gpt_default_max_length": 2048,
                "master_chunk_type": "normal",
                "skip_refinement": False,
                "gpt_model": self.model_name
            }

        # load prompts
        self.prompt_conf = json.loads((self.prompt_dir / "marg" / "agent_prompts.json").read_text())
        self.aggregate_system_prompt = (self.prompt_dir / "marg" / "aggregate.txt").read_text()
        self.review_format_prompt = json.loads(self.review_format_prompt_path.read_text())

    def _format_paper_text(self, paper: Paper):
        """
        Based on get_paper_chunks in make_chunked_paper_diff in multi_agent.py

        :param paper: the paper to be formatted for MARG
        :return: the output text
        """

        def format_paragraph(text, paragraph_id, section_name="unknown"):
            res = ""
            res += "section: {}".format(section_name)
            res += "\nparagraph id: {}".format(paragraph_id)
            res += "\n" + text
            res += "\n\n"

            return res

        paper = paper.without_appendix()

        # add all paragraphs of paper
        res = ""
        for isec, section_name in enumerate(paper.get_section_names()):
            for ipar, paragraph_text in enumerate(paper.get_paragraphs(section_name)):
                paragraph_id = f"{ipar}"
                res += format_paragraph(paragraph_text, paragraph_id, section_name)

        # add abstract
        res = format_paragraph(paper.get_abstract().replace("######", ""), 9999, "Abstract") + res

        # split into suitable chunks
        encoding = tiktoken.encoding_for_model(self.model_name)

        diff_chunks = []
        cur_chunk = []
        cur_chunk_len = 0

        para_chunks = res.split("\n\n")
        for para_chunk in para_chunks:
            # Add 2 for the stripped \n\n
            new_chunk_len = len(encoding.encode(para_chunk)) + 2
            if cur_chunk_len + new_chunk_len > self.marg_config["paper_chunk_size"]:
                diff_chunks.append("\n\n".join(cur_chunk))
                cur_chunk = []
                cur_chunk_len = 0
            cur_chunk.append(para_chunk)
            cur_chunk_len += new_chunk_len

        if len(cur_chunk) != 0:
            diff_chunks.append("\n\n".join(cur_chunk))

        return diff_chunks

    def _marg(self, chunks, config, prompts):
        """
        Based on reviewgen_v24_multi_agent in reviewer_worker/run_reviewgen.py

        :return:
        """
        extra_agents = config["experts"]

        # run swarm
        swarm = MultiAgentGroup(
            config,
            None,
            config["gpt_model"],
            paper_chunk_size=config["paper_chunk_size"],
            prompts=prompts,
            max_tokens=config["gpt_default_max_length"],
            quiet=False,
            use_history_pruning=True,
            taxonomy="",
            master_chunk_type=config["master_chunk_type"],
            extra_bots=extra_agents,
            raw_paper_chunks=chunks,
            local_model=self.local_model
        )
        expert_name_substitutions = {x["name"]: swarm.extra_bot_experts[eidx].agent_name for eidx, x in
                                     enumerate(config["experts"])}
        master_task_prompt = prompts["task_prompt_set1_v1"].format(comment_type="",
                                                                   **expert_name_substitutions)
        rt = swarm.ask_swarm_question(master_task_prompt, pre_prompt="")

        # post process swarm response
        tmpbot = LlmChatBot(
            model_name=self.model_name,
            system_prompt=self.aggregate_system_prompt,
            max_tokens=2048,
            local_model=self.local_model
        )

        rt = tmpbot.chat("Output:\n" + rt)

        # failed to generate a review
        if rt.strip().strip('"') == "No comments.":
            # run with the second version
            rt = swarm.ask_swarm_question(prompts["task_prompt_set1_v2"], pre_prompt="")
            try:
                _ = json.loads(rt)
            except json.decoder.JSONDecodeError:
                # encourage JSON formatting
                rt = swarm.ask_swarm_question(
                    prompts["task_prompt_set1_v2"] + "\n\nMake sure to use the specified JSON format.", pre_prompt="")
        else:
            print(rt)

        # parse the output
        try:
            review1 = rt[rt.index("["):].strip().strip("`")
        except Exception as e:
            self.logger.exception("FAILED TO PARSE MODEL JSON OUTPUT")
            review1 = "null"

        # if without refinement, just return the review as is
        if config.get("skip_refinement"):
            return review1

        # refinement stage
        final_comments = []

        loaded_review1 = parse_llm_output_as_single_json(review1)[1]
        if loaded_review1 is None:
            return review1

        for comment in loaded_review1:
            # for each comment,
            swarm = MultiAgentGroup(
                config,
                None,
                config["gpt_model"],
                paper_chunk_size=config["paper_chunk_size"],
                prompts=prompts,
                max_tokens=config["gpt_default_max_length"],
                quiet=False,
                use_history_pruning=True,
                master_chunk_type=config["master_chunk_type"],
                taxonomy="",
                raw_paper_chunks=chunks,
                local_model=self.local_model
            )
            rt = swarm.ask_swarm_question(
                prompts["task_prompt_set2_v1"].format(comment_type="", review_comments=comment),
                pre_prompt="",
            )
            rt = swarm.ask_swarm_question(prompts["task_prompt_set2_v2"], pre_prompt="")

            comment = parse_llm_output_as_single_json(rt)[1]

            if comment is not None:
                if "revised_comments" in comment:
                    assert "revised_comment" not in comment
                    for c in comment["revised_comments"]:
                        final_comments.append({"revised_comment": c})
                else:
                    final_comments.append(comment)
        return json.dumps([x["revised_comment"] for x in final_comments if x["revised_comment"] is not None])

    def _marg_specialists(self, chunks):
        """
        Based on reviewgen_v26_specialized_multi_agent in reviewer_worker/run_reviewgen.py

        :param chunks: the chunked paper
        :param marg_config: marg_config for execution
        :return:
        """
        comments_by_type = {}
        default_prompts = self.prompt_conf["default_prompts"]
        for pset in self.prompt_conf["prompt_sets"]:
            pset = pset.copy()
            for k, v in default_prompts.items():
                if k not in pset:
                    pset[k] = v

            new_config = self.marg_config.copy()
            new_config["experts"] = pset.get("experts", [])

            # run the review generation
            rev = self._marg(chunks=chunks, config=new_config, prompts=pset)
            comments_by_type[pset["name"]] = json.loads(rev)

        comments_by_type['all'] = list(itertools.chain(*comments_by_type.values()))
        final_rev = json.dumps(comments_by_type)

        return final_rev

    def post_process_review(self, marg_review, paper_title, paper_abstract):
        """
        Post process the review generated by MARG

        :param marg_review: the review generated by MARG
        :return: the post-processed review
        """

        if marg_review is None:
            return None

        # get revision llm
        if not self.local_model:
            llm = OpenAiChatLLM(
                f"marg-revise-review",
                model=self.model_name,
                multimodal=False,
                config=dict(
                    max_tokens=4096
                )
            )
        else:
            llm = self.local_model
            llm.set_config({"max_tokens": 4096})

        # apply the revision prompt
        llm.set_prompt(ChatPromptTemplate.from_messages([(m["role"], m["content"]) for m in self.review_format_prompt]))

        # run the revision
        revised_review = llm({"marg_review": marg_review, "paper_title": paper_title, "paper_abstract": paper_abstract})

        # parse as json
        parsed =  parse_llm_output_as_single_json(revised_review)

        if "sections" not in parsed or "scores" not in parsed:
            return None
        else:
            return parsed

    def run(self, paper: Paper, **config) -> Review:
        chunks = self._format_paper_text(paper)

        raw_review = self._marg_specialists(chunks)
        review = self.post_process_review(raw_review, paper.get_title(), paper.get_abstract())

        if review is None:
            return Review(sections={"main": raw_review},
                          scores=None,
                          meta={"original_review": raw_review, "parsing_failure": True})
        else:
            return Review(sections=review["sections"], scores=review["scores"], meta={"original_review": raw_review})
