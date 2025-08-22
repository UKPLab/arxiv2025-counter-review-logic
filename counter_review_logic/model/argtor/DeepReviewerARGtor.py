import os
import re
from pathlib import Path

from ....data import Paper, Review
from ....framework.argtor import AutomaticReviewGenerator
from ....llm import ChatLLM, parse_llm_output_as_single_json, approximately_truncate

from ai_researcher import DeepReviewer

class DeepReviewerARGtor(AutomaticReviewGenerator):
    """
     Uses the DeepReviewer proposed in

     @misc{zhu2025deepreviewimprovingllmbasedpaper,
      title={DeepReview: Improving LLM-based Paper Review with Human-like Deep Thinking Process},
      author={Minjun Zhu and Yixuan Weng and Linyi Yang and Yue Zhang},
      year={2025},
      eprint={2503.08569},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2503.08569},
    }
    """

    def __init__(self,
                 name: str,
                 llm: ChatLLM = None,
                 prompt_base_path: str | Path = None):
        super().__init__(name)

        # Initialize DeepReviewer with 14B model
        self.deep_reviewer = DeepReviewer(model_size="14B")

        if isinstance(prompt_base_path, str):
            prompt_base_path = Path(prompt_base_path)

        self.formatting_llm = llm
        if self.formatting_llm is not None:
            self.prompt_base_path = prompt_base_path
            if prompt_base_path is None and "PROMPT_DIR" in os.environ:
                self.prompt_base_path = Path(os.environ["PROMPT_DIR"].replace("\"", "")) / "argtor"
            elif prompt_base_path is None:
                raise ValueError("Prompt base path is not provided.")

            assert self.prompt_base_path.exists(), f"the prompt base path needs to exist. It does not: {self.prompt_base_path}."

            self.formatting_llm.load_prompt(self.prompt_base_path / "deepreviewer" / "refine_format.txt")

    @staticmethod
    def _parse_as_review(result, venue_config, strict=True):
        overall_score_name = venue_config["overall_score_name"]
        summary_section_name = venue_config["template_field_semantics"]["summary"]
        strengths_section_name = venue_config["template_field_semantics"]["strengths"]
        weaknesses_section_name = venue_config["template_field_semantics"]["weaknesses"]
        suggestions_section_name = venue_config["template_field_semantics"]["suggestions"]

        def _as_float(s):
            try:
                return float(s)
            except:
                return None

        # catch error on output structure
        if type(result) != list or (type(result) == list and len(result) < 1):
            return None

        result = result[0]

        # take all sections as presented and fill-in sections by conference names of predefined fields
        # skipping "content" field, since it subsumes all other fields
        sections = {k: v for k, v in result["meta_review"].items()
                    if k not in ["rating", "presentation", "contribution", "soundness",
                                 "summary", "strengths", "weaknesses", "suggestions", "content"]}

        if strict and len(result["meta_review"]) == 0:
            return None

        sections[summary_section_name] = result["meta_review"]["summary"] if "summary" in result["meta_review"] else ""
        sections[strengths_section_name] = result["meta_review"]["strengths"] if "strengths" in result[
            "meta_review"] else ""
        sections[weaknesses_section_name] = result["meta_review"]["weaknesses"] if "weaknesses" in result[
            "meta_review"] else ""
        sections[suggestions_section_name] = result["meta_review"]["suggestions"] if "suggestions" in result[
            "meta_review"] else ""

        overall_score = _as_float(result["meta_review"]["rating"]) if "rating" in result["meta_review"] else None
        other_scores = {k: _as_float(v) for k, v in result["meta_review"].items() if
                        k in ["presentation", "contribution", "soundness"]}

        if strict and overall_score is None or sections[summary_section_name] == "":
            return None

        return Review(
            sections=sections,
            scores={overall_score_name: overall_score, **other_scores},
            overall_score=overall_score_name,
            main_section=summary_section_name,
            meta={"original_review": result["raw_text"]}
        )

    def format(self, raw_output, venue_config):
        if self.formatting_llm is None:
            raise ValueError("No formatting LLM provided.")

        # deal with enormous amounts of whitespaces (appears commonly)
        pattern = r'[\s\u00A0\u1680\u180E\u2000-\u200A\u2028\u2029\u202F\u205F\u3000]{3,}'
        raw_output = re.sub(pattern, '  ', raw_output)

        truncated = approximately_truncate(raw_output, self.formatting_llm, margin=8192)
        res = self.formatting_llm({
            "faulty_review_report": truncated
        })
        parsed = parse_llm_output_as_single_json(res)[1]

        if parsed is None:
            return None

        to_process= [
            {
                "meta_review": parsed,
                "raw_text": res
            }
        ]

        return self._parse_as_review(to_process, venue_config)

    def _truncate_paper(self, paper_md:str):
        tok = self.deep_reviewer.model.get_tokenizer()

        encoded = tok.encode(paper_md)
        max_len = int(70000 * 0.8)
        if len(encoded) > max_len:
            paper_md = tok.decode(encoded[:max_len])

        return paper_md

    def run(self, paper: Paper, **config) -> Review:
        paper_text = paper.without_appendix().md
        paper_text = self._truncate_paper(paper_text)

        venue_config = paper.meta["venue_config"]

        result = self.deep_reviewer.evaluate(
            paper_text,
            mode="Best Mode",  # Options: "Fast Mode", "Standard Mode", "Best Mode"
            reviewer_num=4  # Simulate 4 different reviewers
        )

        parsed = self._parse_as_review(result, venue_config)
        if parsed is None and self.formatting_llm:
            parsed = self.format(result[0]["raw_text"], venue_config)
            if parsed is None:
                parsed = self._parse_as_review(result, venue_config, strict=False)
        else:
            # parse under relaxed conditions
            parsed = self._parse_as_review(result, venue_config, strict=False)

        return parsed
