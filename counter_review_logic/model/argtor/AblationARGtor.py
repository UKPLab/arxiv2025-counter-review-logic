import json
import logging
import os
import random
from collections.abc import Callable
from pathlib import Path

from ...data import Paper, Review
from ...framework import AutomaticReviewGenerator, AutomaticReviewDataset, PaperCounterfactualDataset
from ...llm import ChatLLM, parse_llm_output_as_single_json


class AblationARGtor(AutomaticReviewGenerator):
    """
    This class is meant for ablating an existing arg
    """
    def __init__(self, simulated_argtor_name:str, llm: ChatLLM, underlying_original_review_dataset: AutomaticReviewDataset, prompt_base_path: str | Path = None):
        super().__init__("ablator_" + simulated_argtor_name)

        self.simulated_argtor_name = simulated_argtor_name

        self.llm = llm
        self.max_parsing_attempts = 3
        self.cf_dataset = None
        self.underlying_original_review_dataset = underlying_original_review_dataset

        if isinstance(prompt_base_path, str):
            prompt_base_path = Path(prompt_base_path)

        self.prompt_base_path = prompt_base_path
        if prompt_base_path is None and "PROMPT_DIR" in os.environ:
            self.prompt_base_path = Path(os.environ["PROMPT_DIR"].replace("\"", "")) / "ablations"
        elif prompt_base_path is None:
            raise ValueError("Prompt base path is not provided.")

        assert self.prompt_base_path.exists(), f"the prompt base path needs to exist. It does not: {self.prompt_base_path}."

    def set_cf_dataset(self, cf_dataset:PaperCounterfactualDataset):
        self.cf_dataset = cf_dataset

    def load_prompt(self, prompt_name: str):
        if prompt_name == "extend":
            self.llm.load_prompt(self.prompt_base_path / "extend_review.txt")
        elif prompt_name == "revise":
            self.llm.load_prompt(self.prompt_base_path / "revise_review.txt")
        elif prompt_name == "revise_valid_json":
            self.llm.load_prompt(self.prompt_base_path / "revise_valid_json.txt")
        else:
            raise ValueError(f"Unknown prompt name: {prompt_name}. Available: 'extend', 'revise'.")

    def call_llm_unstructured_output(self, prompt, **params):
        self.load_prompt(prompt)
        return self.llm(params)

    def call_llm_structured_output(self, prompt, json_template: str, validate: Callable = None, **params):
        if validate is None:
            validate = lambda x: True

        self.load_prompt(prompt)
        response = self.llm(params)
        parsed = parse_llm_output_as_single_json(response)[1]

        if parsed is not None and not validate(parsed):
            parsed = None

        response2 = response
        for i in range(self.max_parsing_attempts):
            if parsed is not None:
                return parsed

            logging.info(f"Retrying after bad format {i + 1} of {self.max_parsing_attempts} times")

            # If parsing fails, try again with the revise prompt
            self.load_prompt("revise_valid_json")
            response2 = self.llm({
                "output": response2,
                "expected_json_format": json_template,
            })
            parsed = parse_llm_output_as_single_json(response2)[1]

            if parsed is not None and not validate(parsed):
                parsed = None

            if response2.strip() == "INVALID":
                break

        return parsed

    def run(self, paper: Paper, **config) -> Review:
        paper_id = paper.id

        assert self.cf_dataset is not None, "Counterfactual dataset is not set. Please set it using set_cf_dataset method."

        if paper_id not in self.cf_dataset:
            logging.info(f"Cannot generate review for paper {paper_id}, it is not in the counterfactual dataset.")
            return None

        # load cf information
        orig_paper, cf_paper = self.cf_dataset[paper_id]

        # get the original review
        original_review = self.underlying_original_review_dataset.get_review(paper_id, self.simulated_argtor_name)
        if original_review is None:
            logging.error(f"Original review for paper {paper_id} not found in the underlying dataset.")
            return None

        original_scores = original_review.scores

        # paraphrase the original review text (maintain scores)
        review_text_fields = original_review.to_json()["sections"]

        revised_review_text_fields = self.call_llm_structured_output(
            "revise",
            json_template=json.dumps({k: "..." for k in review_text_fields}, indent=4),
            validate=lambda x: isinstance(x, dict) and all(k in x for k in review_text_fields.keys()),
            review_report=json.dumps(review_text_fields, indent=4)
        )

        # abort if failed
        if revised_review_text_fields is None:
            logging.error(f"Failed to generate revised review for paper {paper_id} using {self.simulated_argtor_name}.")
            return None

        main_section_title = original_review.main_section_title
        main_sec_candidates = ["paper summary", "paper topic", "summary"]
        for k in revised_review_text_fields.keys():
            if any(c in k.lower() for c in main_sec_candidates):
                main_section_title = k
                break

        oscore_name = original_review.overall_score_title
        o_score_candidates= ["overall", "rating"]
        for k in original_scores.keys():
            if any(c in k.lower() for c in o_score_candidates):
                oscore_name = k
                break


        changes = cf_paper.changes
        if "operation" not in changes: # soundness neutral
            # merge everything into one output review
            review = Review(
                sections=revised_review_text_fields,
                scores=original_review.scores,
                main_section=main_section_title,
                overall_score=oscore_name,
                meta={
                    "oracle": True,
                    "simulated_argtor_name": self.simulated_argtor_name,
                    "type": "soundness neutral rewrite"
                }
            )
        else:
            operation = changes["operation"]

            cf_type = self.cf_dataset.meta["counterfactual_type"]
            break_target = changes["break_target"]
            sum_field = None
            if "finding" in cf_type:
                sum_field = "claim_summary"
            elif "result" in cf_type:
                sum_field = "result_summary"
            elif "conclusion" in cf_type:
                sum_field = "conclusion_summary"

            target_summary = break_target[sum_field]

            ex_field = None
            if "finding" in cf_type:
                ex_field = "explanation"
            elif "result" in cf_type:
                ex_field = "conclusion_verification"
            elif "conclusion" in cf_type:
                ex_field = "logical_relation_revised_conclusion_to_old_conclusion"

            explanation = changes["break_result"][ex_field]

            change_summary = f"{operation}:\n{explanation}"

            edits = changes["edits"]
            edits_summary = [
                f"{t} #{n}: {s}" for t, s, n in edits
            ]
            edits_summary = "\n".join(edits_summary)

            # add a comment on the soundness using oracle information
            soundness_comment = self.call_llm_unstructured_output(
                "extend",
                original_claims = target_summary,
                soundness_observations = change_summary,
                problematic_passages = edits_summary
            )

            # check for a valid weakness field
            weakness_field_indicators = ["weakness", "reject"]
            added = False
            for k, v in revised_review_text_fields.items():
                if any(indicator in k.lower() for indicator in weakness_field_indicators):
                    if random.random() < 0.3:
                        revised_review_text_fields[k] = v + "\n\n" + soundness_comment
                    else:
                        revised_review_text_fields[k] = soundness_comment + "\n\n" + v
                    added = True

            # add to any field as a fallback
            if not added:
                for k, v in revised_review_text_fields.items():
                    if random.random() < 0.3:
                        revised_review_text_fields[k] = v + "\n\n" + soundness_comment
                    else:
                        revised_review_text_fields[k] = soundness_comment + "\n\n" + v
                    break

            # adapt the overall score with randomness to decrease
            revised_scores = {k:v for k,v in original_review.scores.items()}

            if oscore_name in revised_scores:
                revised_scores[oscore_name] = max(1, revised_scores[oscore_name] - random.choice([0, 0.5, 0.5, 1.0, 1.0, 1.0, 1.5, 1.5, 2.0, 2.0]))
            else:
                #as a fallback, reduce all scores
                for k, v in revised_scores.items():
                    revised_scores[k] = max(1, v - random.choice([0, 0.5, 0.5, 1.0, 1.0, 1.0, 1.5, 1.5, 2.0, 2.0]))

            # merge everything into one output review
            review = Review(
                sections=revised_review_text_fields,
                scores=revised_scores,
                main_section=main_section_title,
                overall_score=oscore_name,
                meta={
                    "oracle": True,
                    "simulated_argtor_name": self.simulated_argtor_name,
                    "type": "rewrite and added comment"
                }
            )

        return review