import logging
import os
from pathlib import Path

from ...data import Review
from ...framework import ReviewChangeDetector, ReviewDelta
from ...llm import ChatLLM, parse_llm_output_as_single_json
from .utils import get_review_text_data


class PolarityRCD(ReviewChangeDetector):
    """
    Determines sentiment of key talking points of a review. The sentiment is stored per point.
    """

    def __init__(self, llm: ChatLLM, prompt_base_path: str | Path = None, with_feedback=True):
        super().__init__(f"polarity_rcd_{llm.name}")

        if isinstance(prompt_base_path, str):
            prompt_base_path = Path(prompt_base_path)

        self.prompt_base_path = prompt_base_path
        if prompt_base_path is None and "PROMPT_DIR" in os.environ:
            self.prompt_base_path = Path(os.environ["PROMPT_DIR"].replace("\"", "")) / "rcd"
        elif prompt_base_path is None:
            raise ValueError("Prompt base path is not provided.")

        assert self.prompt_base_path.exists(), f"the prompt base path needs to exist. It does not: {self.prompt_base_path}."
        self.llm = llm

        self.with_feedback = with_feedback

        # setup (test loading both prompts)
        if with_feedback:
            self._load_feedback_prompt()

        self._load_revise_extract_prompt()
        self._load_extract_prompt()

    def _load_extract_prompt(self):
        return ChatLLM._load_chat_prompt(self.prompt_base_path / "point_overlap" / f"extract_prompt.txt")

    def _load_revise_extract_prompt(self):
        return ChatLLM._load_chat_prompt(self.prompt_base_path / "point_overlap" / "refine_extract.txt")

    def _load_feedback_prompt(self):
        return ChatLLM._load_chat_prompt(self.prompt_base_path / "point_overlap" / "feedback_prompt.txt")

    def align_points(self, review_text1, review_text2):
        points = await self.identify_points(review_text1, review_text2)
        if points is None:
            return None

        for p in points["points_review1"]:
            p["id"] = "r1_" + str(p["id"])

        for p in points["points_review2"]:
            p["id"] = "r2_" + str(p["id"])

        return {
            "points_review1": points["points_review1"],
            "points_review2": points["points_review2"],
        }

    def identify_points(self, review_text1, review_text2):
        def validate(parsed_output):
            if parsed_output is None:
                return False

            if type(parsed_output) != dict:
                return False

            all_fields = all(f in parsed_output for f in ["points_review1", "points_review2"])
            if not all_fields:
                return False

            all_lists = all([type(f) == list for f in parsed_output.values()])
            if not all_lists:
                return False

            for p in parsed_output["points_review1"] + parsed_output["points_review2"]:
                if type(p) != dict:
                    return False

                if "id" not in p or "span" not in p:
                    return False

                if "summary" not in p or "evidence" not in p or "conclusion" not in p or "sentiment" not in p:
                    return False

            return True

        # initial extraction
        raw_res = await self.llm({
            "report1": review_text1,
            "report2": review_text2
        }, prompt=self._load_extract_prompt())

        # parse the result
        parsed = parse_llm_output_as_single_json(raw_res)[1]
        valid = validate(parsed)

        max_it = 5
        raw_res2 = raw_res
        for i in range(max_it):
            if valid:
                break

            print(f"Revising extraction due to invalid output. {i + 1}/{max_it}")

            raw_res2 = await self.llm({
                "faulty_response": raw_res2
            }, prompt=self._load_revise_extract_prompt())

            parsed = parse_llm_output_as_single_json(raw_res2)[1]
            valid = validate(parsed)

        if not valid:
            return None

        # no feedback loop used
        if not self.with_feedback:
            return parsed

        # give feedback
        def get_spans_without_coverage(review_text, spans):
            missed = review_text
            for span in spans:
                missed = missed.replace(span, "")

            return missed

        missed_spans1 = get_spans_without_coverage(review_text1,
                                                   [p["span"] for p in parsed["points_review1"]])
        missed_spans2 = get_spans_without_coverage(review_text2,
                                                   [p["span"] for p in parsed["points_review2"]])

        # skip feedback if nearly everything is included
        if len(missed_spans1) < 0.3 * len(review_text1) and len(missed_spans2) < 0.3 * len(review_text2):
            return parsed

        raw_res_fb = await self.llm({
            "initial_result": raw_res,
            "missing_spans1": missed_spans1,
            "missing_spans2": missed_spans2,
            "report1": review_text1,
            "report2": review_text2
        }, prompt=self._load_feedback_prompt())

        parsed = parse_llm_output_as_single_json(raw_res_fb)[1]
        valid = validate(parsed)

        max_it = 5
        raw_res3 = raw_res_fb
        for i in range(max_it):
            if valid:
                break

            print(f"Revising extraction due to invalid output. {i + 1}/{max_it}")

            raw_res3 = await self.llm({
                "faulty_response": raw_res3
            }, prompt=self._load_revise_extract_prompt())

            parsed = parse_llm_output_as_single_json(raw_res3)[1]
            valid = validate(parsed)

        if not valid:
            return None

        return parsed

    def run(self, review1: Review, review2: Review, **config) -> ReviewDelta:
        result = {"by_section": {}}

        data = get_review_text_data(review1, review2)

        r1_full, r2_full = data["r1"], data["r2"]
        r1_by_section, r2_by_section = data["r1_sections"], data["r2_sections"]

        # identify the points using the llm
        identified_points = self.align_points(r1_full, r2_full)
        if identified_points is None:
            logging.warning("Failed to fetch identified points. Returning empty delta.")
            return ReviewDelta(review1, review2, {"error": "failed to identify points"})

        # add the identified points to the result
        result.update(identified_points)

        # add the points by section
        def _find_points_in_section(section, points):
            return [point for point in points if point["span"] in str(section)]

        result["by_section"]["points_review1"] = {
            section_name: _find_points_in_section(section_text, identified_points["points_review1"])
            for section_name, section_text in r1_by_section.items()
        }

        result["by_section"]["points_review2"] = {
            section_name: _find_points_in_section(section_text, identified_points["points_review2"])
            for section_name, section_text in r2_by_section.items()
        }

        delta = ReviewDelta(review1, review2, result)

        return delta
