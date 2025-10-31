import hashlib
import os
import re
import secrets
from pathlib import Path
from typing import Optional, List, Any

from langchain_core.callbacks import CallbackManagerForLLMRun
from pydantic import PrivateAttr

from ...data import Paper, Review
from ...llm import OpenAiChatLLM
from ...framework import AutomaticReviewGenerator
from ...llm import parse_llm_output_as_single_json, approximately_truncate

from .treereview.agents.answer_synthesizer import AnswerSynthesizer
from .treereview.agents.question_generator import QuestionGenerator
from .treereview.utility.LLMClient import LLMClient
from .treereview.utility.context_ranker import ContextRanker
from .treereview.models.paper import Paper as PaperTR
from .treereview.utility.text_chunker import TextChunker
from .treereview.core import PipelineConfig, ReviewPipeline


class Wrapper(LLMClient):
    _chat_llm: OpenAiChatLLM = PrivateAttr(True)

    def __init__(self, llm: OpenAiChatLLM):
        super().__init__()
        self._chat_llm = llm

    def _llm_type(self):
        return self._chat_llm.name

    def main(self, prompt):
        config = dict(
            temperature=0,
            max_tokens=32768
        )
        resp = self._chat_llm.call_openai_fixed_input(prompt, config=config)

        return resp

    def _call(
            self,
            prompt: str,
            stop: Optional[List[str]] = None,
            run_manager: Optional[CallbackManagerForLLMRun] = None,
            **kwargs: Any,
    ) -> str:
        return self.main(prompt)


class TreeReviewARGtor(AutomaticReviewGenerator):
    """
   Implements TreeReview copied from

   https://github.com/YuanChang98/tree-review
    """

    def __init__(self,
                 name: str,
                 llm: OpenAiChatLLM,
                 prompt_base_path: str | Path = None):
        super().__init__(name)

        if isinstance(prompt_base_path, str):
            prompt_base_path = Path(prompt_base_path)

        self.prompt_base_path = prompt_base_path
        if prompt_base_path is None and "PROMPT_DIR" in os.environ:
            self.prompt_base_path = Path(os.environ["PROMPT_DIR"].replace("\"", ""))
        elif prompt_base_path is None:
            self.prompt_base_path = Path(__file__).resolve().parent / "prompts" # default path

        assert self.prompt_base_path.exists(), f"the prompt base path needs to exist. It does not: {self.prompt_base_path}."

        # model name
        self.llm = llm
        assert isinstance(self.llm, OpenAiChatLLM), "TreeReviewARGtor requires an OpenAiChatLLM instance."

        self.default_conf = {
            "max_depth": 4,
            "retrieval_top_k": 3
        }

    def review(self, paper, conf=None):
        if conf is None:
            conf = self.default_conf

        pipe_config = PipelineConfig(
            max_depth=conf["max_depth"],
            retrieval_top_k=conf["retrieval_top_k"]
        )
        question_gen, context_ranker, answer_syn = self.initialize_agents()

        pipeline = ReviewPipeline(
            paper=paper,
            question_generator=question_gen,
            context_ranker=context_ranker,
            answer_synthesizer=answer_syn,
            config=pipe_config,
            state_file=None
        )

        result = pipeline.run()

        return result

    def initialize_agents(self):
        llm = Wrapper(llm=self.llm)

        question_gen = QuestionGenerator(llm=llm)
        context_ranker = ContextRanker()
        answer_syn = AnswerSynthesizer(llm=llm)

        print("INitialized agents")

        return question_gen, context_ranker, answer_syn

    def format(self, raw_output, venue_config):
        raw_output = str(raw_output)

        truncated = approximately_truncate(raw_output, self.llm, margin=8192)
        for i in range(3):  # try up to 3 times to parse
            self.llm.load_prompt(self.prompt_base_path / "treereview" / "refine_format.txt")
            res = self.llm({
                "faulty_review_report": truncated,
                "template": venue_config["review_template"],
                "scores": venue_config["review_scores"]
            })
            parsed = parse_llm_output_as_single_json(res)[1]

            if parsed is not None:
                break

        return parsed

    def _parse_as_review(self, output: str, venue_config: dict):
        """
        Output format:

        **Summary:**
        Summary content
        **Strengths:**
        Strengths result
        **Weaknesses:**
        Weaknesses result
        **Questions:**
        Questions result
        **Soundness:**
        Soundness result
        **Presentation:**
        Presentation result
        **Contribution:**
        Contribution result
        **Rating:**
        Rating result
        **Confidence:**
        Confidence result
        """
        overall_score_name = venue_config["overall_score_name"]
        summary_section_name = venue_config["template_field_semantics"]["summary"]
        strengths_section_name = venue_config["template_field_semantics"]["strengths"]
        weaknesses_section_name = venue_config["template_field_semantics"]["weaknesses"]
        suggestions_section_name = venue_config["template_field_semantics"]["suggestions"]

        fields = ["Summary",
                  "Strengths",
                  "Weaknesses",
                  "Questions"]
        scores = ["Soundness",
                  "Presentation",
                  "Contribution",
                  "Rating",
                  "Confidence"]

        field_matches = []
        for field in fields:
            pattern = rf"\*\*{field}:?\*\*"
            match = re.search(pattern, output, re.DOTALL)
            if match:
                field_matches.append((field, match.start(), match.end()))

        score_matches = []
        for score in scores:
            pattern = rf"\*\*{score}:?\*\*"
            match = re.search(pattern, output, re.DOTALL)
            if match:
                score_matches.append((score, match.start(), match.end()))

        if score_matches:
            start_score = score_matches[0][1]
        else:
            start_score = len(output)

        sections = {}
        for i in range(len(field_matches)):
            field, start, end = field_matches[i]
            next_start = field_matches[i + 1][1] if i + 1 < len(field_matches) else start_score
            sections[field] = output[end:next_start].strip()

        scores = {}
        for i in range(len(score_matches)):
            score, start, end = score_matches[i]
            next_start = score_matches[i + 1][1] if i + 1 < len(score_matches) else len(output)
            score_content = output[end:next_start].strip()
            try:
                scores[score] = float(score_content)
            except ValueError:
                # parse a number from the text
                number_match = re.search(r"([\d.]+)", score_content)
                if number_match:
                    scores[score] = float(number_match.group(1))
                else:
                    scores[score] = None

        # create the review object
        if "Summary" in sections:
            sections[summary_section_name] = sections["Summary"]
            del sections["Summary"]
            ssection_name = summary_section_name
        else:
            ssection_name = list(sections.keys())[0] if sections else None

        if "Strengths" in sections:
            sections[strengths_section_name] = sections["Strengths"]
            del sections["Strengths"]

        if "Weaknesses" in sections:
            sections[weaknesses_section_name] = sections["Weaknesses"]
            del sections["Weaknesses"]

        if "Questions" in sections:
            sections[suggestions_section_name] = sections["Questions"]
            del sections["Questions"]

        if "Rating" in scores:
            scores[overall_score_name] = scores["Rating"]
            del scores["Rating"]
            oscore_name = overall_score_name
        else:
            oscore_name = list(scores.keys())[0] if scores else None

        if len(sections) == 0:
            sections = {"main": output}
            ssection_name = "main"
        if len(scores) == 0:
            scores = {"overall": None}
            oscore_name = "overall"

        try:
            return Review(
                sections=sections,
                scores=scores,
                overall_score=oscore_name,
                main_section=ssection_name,
                meta={"original_review": output}
            )
        except:
            return Review(
                sections={"main": output},
                scores={"overall": None},
                overall_score="overall",
                main_section="main",
                meta={"original_review": output, "error": True}
            )

    def run(self, paper: Paper, **config) -> Review:
        paper_text = paper.without_appendix().md
        paper_text = approximately_truncate(paper_text, self.llm)

        # discard references
        if "## References" in paper_text:
            paper_text = paper_text.split("## References")[0].strip()

        chunker = TextChunker()
        chunks = chunker.chunk(paper_text)

        # get around caching by tree reviewer
        random_bytes = secrets.token_bytes(32)
        pseudo_unique_id = hashlib.sha256(random_bytes)
        pseudo_unique_hash = pseudo_unique_id.hexdigest()

        paper_tr = PaperTR(id=pseudo_unique_hash,
                           title=paper.get_title(),
                           abstract=paper.get_abstract(),
                           toc="\n".join("##" + k for k in paper.get_sections().keys()),
                           chunks=chunks,
                           content=paper_text
                           )

        try:
            review = self.review(paper_tr, conf=config if config else self.default_conf)
        except Exception as e:
            print("Error during review generation. Returning None.", e)
            return None

        if review is not None:
            review_text = review["full_review"]
        else:
            return None

        print("REVIEW")
        print(review_text)

        parsed = self._parse_as_review(review_text, paper.meta["venue_config"])

        print("PARSED")
        print(parsed)

        return parsed
