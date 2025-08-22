import os
import re
from pathlib import Path

from ....data import Paper, Review
from ....framework.argtor import AutomaticReviewGenerator
from ....llms import ChatLLM, approximately_truncate


class SystematicARGtor(AutomaticReviewGenerator):
    """
   This class implements the new prompts designed in this work that enable three levels of detail
   during assessment.
    """

    def __init__(self,
                 name: str,
                 llm: ChatLLM,
                 prompt_type: str,
                 prompt_base_path: str | Path = None):
        super().__init__(name)

        if isinstance(prompt_base_path, str):
            prompt_base_path = Path(prompt_base_path)

        self.prompt_base_path = prompt_base_path
        if prompt_base_path is None and "PROMPT_DIR" in os.environ:
            self.prompt_base_path = Path(os.environ["PROMPT_DIR"].replace("\"", "")) / "argtor"
        elif prompt_base_path is None:
            raise ValueError("Prompt base path is not provided.")

        assert self.prompt_base_path.exists(), f"the prompt base path needs to exist. It does not: {self.prompt_base_path}."

        assert prompt_type in ["generic", "guided", "guideline", "workflow"], f"Prompt type {prompt_type} is not supported."
        self.prompt_type = prompt_type

        # model name
        self.llm = llm

        # setup (test loading both prompts)
        self._load_revise_prompt()
        self._load_review_prompt()

    def _load_review_prompt(self):
        self.llm.load_prompt(self.prompt_base_path / "systematic" / f"{self.prompt_type}.txt")

    def _load_revise_prompt(self):
        self.llm.load_prompt(self.prompt_base_path / "systematic" / "refine_format.txt")

    @staticmethod
    def _parse_as_review(output, venue_config):
        # preamble
        template_fields = venue_config["template_field_semantics"]
        overall_score_name = venue_config["overall_score_name"]

        # get relevant field names
        summary_field_name = template_fields["summary"]
        strengths_field_name = template_fields["strengths"]
        weaknesses_field_name = template_fields["weaknesses"]
        suggestions_field_name = template_fields["suggestions"]
        other_fields = [k for k in venue_config["review_template"].keys() if k not in template_fields.values()]

        # make sure the venue config adds up
        assert summary_field_name in venue_config["review_template"]
        assert strengths_field_name in venue_config["review_template"]
        assert weaknesses_field_name in venue_config["review_template"]
        assert suggestions_field_name in venue_config["review_template"]

        fields = [summary_field_name,
                  strengths_field_name,
                  weaknesses_field_name,
                  suggestions_field_name] + \
                 other_fields

        scores = list(venue_config["review_scores"].keys())

        # clean for deepseek versions
        if "<think>" in output:
            sthink = output.find("<think>")
            ethink = output.find("</think>")
            output = output[:sthink] + output[ethink + len("</think>"):]

        # first find start of review by -----
        start = output.find("-----")
        alternative_start = output.find("**Review Report**")

        if start == -1 or alternative_start < start:
            start = alternative_start
            if alternative_start > -1:
                end_start = start + len("**Review Report**")
            else:
                end_start = -1
        else:
            end_start = start + len("-----")

        if start > -1:
            report = output[end_start:].strip()
        else:
            report = output.strip()

        # pre-clean the output md styling
        report = report.replace("**", "")
        report = report.replace("__", "")

        # construct regex pattern
        section_pattern = rf"###\s*({'|'.join(map(re.escape, fields + ['Scores']))})"
        score_pattern = rf"[\*-]\s*({'|'.join(map(re.escape, scores))}):\s*([\d.]+)"

        matches = list(re.finditer(section_pattern, report, re.IGNORECASE))

        extracted_data = {}
        for i, match in enumerate(matches):
            name = match.group(1)

            # fix name if needed (matching lowercase, too)
            if name not in fields:
                for f in fields:
                    if name.lower().strip() == f.lower().strip():
                        name = f
                        break

            start = match.end()  # Start after section header

            # Determine end of the section (start of the next header or end of text)
            end = matches[i + 1].start() if i + 1 < len(matches) else len(report)

            text = report[start:end].strip()
            if text.startswith(":"):
                text = text[1:].strip()

            if text.endswith("###"):
                text = text[:-3].strip()

            # Extract content and clean up
            extracted_data[name] = text

        if "Scores" in extracted_data:
            score_text = extracted_data["Scores"]
            del extracted_data["Scores"]
        else:
            score_text = report

        score_matches = list(re.finditer(score_pattern, score_text, re.IGNORECASE))
        for i, match in enumerate(score_matches):
            name = match.group(1)

            if name is None:
                extracted_data[name] = None

            # fix name if needed (matching lowercase, too)
            if name not in scores:
                for f in scores:
                    if name.lower().strip() == f.lower().strip():
                        name = f
                        break

            # Extract content and clean up
            try:
                extracted_data[name] = match.group(2).strip()
            except IndexError:
                extracted_data[name] = None

        # if no overall score, try more liberal parsing
        if overall_score_name not in extracted_data:
            recovery = re.search(rf"{overall_score_name}(?: Score)?:?\s*([\d.]+)", score_text, re.IGNORECASE)
            if recovery is not None:
                extracted_data[overall_score_name] = recovery.group(1).strip()

        for score in scores:
            if score not in extracted_data or extracted_data[score] is None:
                continue

            try:
                extracted_data[score] = float(extracted_data[score])
            except ValueError:
                extracted_data[score] = None

        parsing_failure = summary_field_name not in extracted_data or strengths_field_name not in extracted_data or weaknesses_field_name not in extracted_data
        parsing_failure = parsing_failure or overall_score_name not in extracted_data or extracted_data[overall_score_name] is None

        if parsing_failure:
            return None

        return Review(sections={sec: text for sec, text in extracted_data.items() if sec in fields},
                      scores={score: val for score, val in extracted_data.items() if score in scores},
                      main_section=summary_field_name,
                      overall_score=overall_score_name,
                      meta={"original_review": output})

    def select_prompt(self, prompt_type: str):
        assert prompt_type in ["generic", "guided", "guideline", "workflow"], f"Prompt type {prompt_type} is not supported."
        self.prompt_type = prompt_type

    def construct_review_template(self, venue_config):
        fields = venue_config["review_template"]
        scores = venue_config["review_scores"]

        res = []
        example = []
        for k,v in fields.items():
            res += [f"### {k}:\n{v}"]
            example += [f"### {k}:\nThis is my assessment on {k}."]

        res += ["\n### Scores"]
        example += ["\n### Scores"]

        for k, v in scores.items():
            rubrics = [f"{kk}: {vv}" for kk, vv in v["scores"].items()]
            rubrics = '\n'.join(rubrics)
            ex_value = list(v["scores"].keys())[-1]

            res += [f"* {k}:\n{v['meaning']} Select from:\n{rubrics}"]
            example += [f"* {k}: {ex_value}"]

        return "\n".join(res), "\n".join(example)

    def run(self, paper: Paper, **config) -> Review:
        paper_text = paper.without_appendix().md
        paper_text = approximately_truncate(paper_text, self.llm)

        paper_title = paper.meta["title"] if "title" in paper.meta else "SEE BELOW"

        venue_config = paper.meta["venue_config"]
        venue_name = venue_config["type"]
        venue_desc = venue_config["description"]
        template, example = self.construct_review_template(venue_config)

        params = {
            "title": paper_title,
            "paper_content": paper_text,
            "template": template,
            "venue": venue_name,
            "venue_description": venue_desc
        }
        # additional venue specific parameters depending on prompt type
        if self.prompt_type == "guideline":
            params["venue_guidelines"] = venue_config["guidelines"]
        elif self.prompt_type == "guided":
            params["guide_instructions"] = venue_config["guidelines_detailed"]

        # run review generation with LLM
        self._load_review_prompt()
        result = self.llm(params)
        parsed = self._parse_as_review(result, venue_config)

        if parsed is None:
            # revise (up to 3 times)
            self._load_revise_prompt()
            result2 = result
            for i in range(3):
                result2 = self.llm({
                    "faulty_review_report": result2,
                    "template": template,
                    "example": example
                })
                # validate revised version by length, otherwise treat as error
                if len(result2) <= len(result) * 0.5:
                    parsed = None
                    continue

                parsed = self._parse_as_review(result2, venue_config)
                if parsed is not None:
                    break

            # still parsing error
            if parsed is None:
                return Review(sections={"main": result2},
                              scores=None,
                              meta={"original_review": result, "parsing_failure": True})

        return parsed
