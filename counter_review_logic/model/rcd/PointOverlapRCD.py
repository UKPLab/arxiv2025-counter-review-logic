import logging
import os
from pathlib import Path


from ...data import Review
from ...framework import ReviewChangeDetector, ReviewDelta
from ...llm import ChatLLM, parse_llm_output_as_single_json
from .utils import get_review_text_data


class PointOverlapRCD(ReviewChangeDetector):
    """
    This class determines the delta between two reviews by first extracting the key assessments
    per review and then checking if they align, contradict or are independent. This leans on to prior
    literature on review changes. The detection is done using an LLM.
    """

    def __init__(self, llm: ChatLLM, prompt_base_path: str | Path = None, with_feedback=True, multi_match=True):
        super().__init__(f"pointoverlap_rcd_{llm.name}")

        if isinstance(prompt_base_path, str):
            prompt_base_path = Path(prompt_base_path)

        self.prompt_base_path = prompt_base_path
        if prompt_base_path is None and "PROMPT_DIR" in os.environ:
            self.prompt_base_path = Path(os.environ["PROMPT_DIR"].replace("\"", "")) / "rcd"
        elif prompt_base_path is None:
            raise ValueError("Prompt base path is not provided.")

        assert self.prompt_base_path.exists(), f"the prompt base path needs to exist. It does not: {self.prompt_base_path}."

        assert multi_match, "non multi match is legacy and not supported anymore"

        self.llm = llm

        self.with_feedback = with_feedback
        self.multi_match = multi_match

        # setup (test loading both prompts)
        if with_feedback:
            self._load_feedback_prompt()

        self._load_revise_extract_prompt()
        self._load_revise_match_prompt()
        self._load_match_prompt()
        self._load_extract_prompt()

    def _load_extract_prompt(self):
        return ChatLLM._load_chat_prompt(self.prompt_base_path / "point_overlap" / f"extract_prompt.txt")

    def _load_match_prompt(self):
        return ChatLLM._load_chat_prompt(self.prompt_base_path / "point_overlap" / f"match_multi_prompt.txt")

    def _load_revise_extract_prompt(self):
        return ChatLLM._load_chat_prompt(self.prompt_base_path / "point_overlap" / "refine_extract.txt")

    def _load_feedback_prompt(self):
        return ChatLLM._load_chat_prompt(self.prompt_base_path / "point_overlap" / "feedback_prompt.txt")

    def _load_revise_match_prompt(self):
        return ChatLLM._load_chat_prompt(self.prompt_base_path / "point_overlap" / "refine_match_multi.txt")

    async def align_points(self, review_text1, review_text2):
        points = await self.identify_points(review_text1, review_text2)
        if points is None:
            return None

        for p in points["points_review1"]:
            p["id"] = "r1_" + str(p["id"])

        for p in points["points_review2"]:
            p["id"] = "r2_" + str(p["id"])

        aligned_points = await self.match_points(points["points_review1"], points["points_review2"])
        if aligned_points is None:
            return None

        return {
            "points_review1": points["points_review1"],
            "points_review2": points["points_review2"],
            "shared_points": aligned_points["shared_points"],
            "contradictory_points": aligned_points["contradictory_points"]
        }

    async def identify_points(self, review_text1, review_text2):
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
        raw_res = await self.llm.async_call({
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

            raw_res2 = await self.llm.async_call({
                "faulty_response": raw_res2
            }, prompt=self._load_revise_match_prompt())

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

        raw_res_fb = await self.llm.async_call({
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

            raw_res3 = await self.llm.async_call({
                "faulty_response": raw_res3
            }, prompt=self._load_revise_match_prompt())

            parsed = parse_llm_output_as_single_json(raw_res3)[1]
            valid = validate(parsed)

        if not valid:
            return None

        return parsed

    async def match_points(self, review1_points, review2_points):
        def validate(parsed_output):
            if parsed_output is None:
                return False

            if type(parsed_output) != dict:
                return False

            all_fields = all(f in parsed_output for f in ["label", "explanation"])
            if not all_fields:
                return False

            return parsed_output["label"] in ["align", "contradict", "neutral"]

        def validate_multi(parsed_output):
            if parsed_output is None:
                return False

            if type(parsed_output) != list:
                return False

            all_fields = all({"id1", "id2", "label"}.issubset(set(f.keys())) for f in parsed_output)
            if not all_fields:
                return False

            all_labels = all(f["label"] in ["align", "contradict", "neutral"] for f in parsed_output)
            return all_labels

        aligned, contradict, neutral = [], [], []

        def to_md(points):
            return "\n\n".join(["\n".join([f"#### Point {p['id']}",
                                           f"**Content**: {p['span']}",
                                           f"**Summary**: {p['summary']}",
                                           f"**Evidence**: {p['evidence']}",
                                           f"**Conclusion** {p['conclusion']}",
                                           f"**Sentiment**: {p['sentiment']}"]) for p in points])

        raw_res = await self.llm.async_call({
            "points1": to_md(review1_points),
            "points2": to_md(review2_points)
        }, prompt=self._load_match_prompt())
        parsed = parse_llm_output_as_single_json(raw_res)[1]
        valid = validate_multi(parsed)

        max_it = 3
        raw_res2 = raw_res
        for i in range(max_it):
            if valid:
                break

            raw_res2 = await self.llm.async_call({
                "faulty_response": raw_res2
            }, prompt=self._load_revise_match_prompt())

            parsed = parse_llm_output_as_single_json(raw_res2)[1]
            valid = validate(parsed)

        if valid:
            for alignment in parsed:
                id1 = str(alignment["id1"])
                id2 = str(alignment["id2"])

                # skip self-referential comparisons
                if id1 not in [str(point["id"]) for point in review1_points] or id2 not in [str(point["id"]) for point
                                                                                            in review2_points]:
                    continue

                if alignment["label"] == "align":
                    aligned.append({
                        "id1": alignment["id1"],
                        "id2": alignment["id2"],
                        "explanation": alignment["explanation"]
                    })
                elif alignment["label"] == "contradict":
                    contradict.append({
                        "id1": alignment["id1"],
                        "id2": alignment["id2"],
                        "explanation": alignment["explanation"]
                    })
                elif alignment["label"] == "neutral":
                    neutral.append({
                        "id1": alignment["id1"],
                        "id2": alignment["id2"],
                        "explanation": alignment["explanation"]
                    })

        return {
            "shared_points": aligned,
            "contradictory_points": contradict,
            "neutral_points": neutral
        }

    def run(self, review1: Review, review2: Review, **config) -> ReviewDelta:
        raise NotImplementedError("Only async version support. Call arun()")

    async def arun(self, review1: Review, review2: Review, **config) -> ReviewDelta:
        result = {"by_section": {}}

        data = get_review_text_data(review1, review2)

        r1_full, r2_full = data["r1"], data["r2"]
        r1_by_section, r2_by_section = data["r1_sections"], data["r2_sections"]

        # identify the points using the llm
        identified_points = await self.align_points(r1_full, r2_full)
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

        # get shared points in common sections
        result["by_section"]["shared_points"] = []

        for shared_point in identified_points["shared_points"]:
            id1 = shared_point["id1"]
            id2 = shared_point["id2"]

            section_shared_point1 = next(
                (section for section, points in result["by_section"]["points_review1"].items() for point in points if
                 point["id"] == id1), None)
            section_shared_point2 = next(
                (section for section, points in result["by_section"]["points_review2"].items() for point in points if
                 point["id"] == id2), None)

            if section_shared_point1 is None or section_shared_point2 is None:
                continue

            result["by_section"]["shared_points"] += [[section_shared_point1, section_shared_point2]]

        delta = ReviewDelta(review1, review2, result)

        return delta
