import logging
import os
import random
import re
from pathlib import Path

from ...data import Paper
from ...framework import PaperCounterfactualGenerator, PaperCounterfactual
from ...llm import ChatLLM, parse_llm_output_as_single_json


class LanguageErrorCF(PaperCounterfactualGenerator):
    """
    This class generates counterfactuals by adding language errors to the text.

    It is a manually designed counterfactual generator.
    """

    def __init__(self, llm: ChatLLM, prompt_base_path: Path | str = None, error_ratio=0.2):
        super().__init__(name=f"language_error_{error_ratio:.2f}")

        if prompt_base_path is None and "PROMPT_DIR" in os.environ:
            prompt_base_path = Path(os.environ["PROMPT_DIR"].replace("\"", "")) / "cfgen"
        elif prompt_base_path is None:
            prompt_base_path = Path(__file__).resolve().parent / "prompts" # default path

        if isinstance(prompt_base_path, str):
            prompt_base_path = Path(prompt_base_path)

        self.prompt_base_path = prompt_base_path

        self.error_ratio = error_ratio

        self.llm = llm
        self.llm.set_post_processor(parse_llm_output_as_single_json)
        self.use_conversion_prompt()

    def use_conversion_prompt(self):
        self.llm.load_prompt(self.prompt_base_path / "manual/style_change/language_errors.txt")

    def use_revision_prompt(self):
        self.llm.load_prompt(self.prompt_base_path / "manual/style_change/revise_response.txt")

    def _revise_paper_passage(self, paper, lines, revision):
        revised_section = []
        for lix in range(lines[0], lines[-1]):
            text = None
            for rev in revision:
                if lix == rev["line"]:
                    text = rev["changed_line"]

            if text is None:
                text = paper._indexes_to_text(lix, lix)

            revised_section += [text]

        return "\n".join(revised_section)

    def revise_section(self, paper, section_name, revision):
        section_lines = paper._structured_md["sections"][section_name]

        return self._revise_paper_passage(paper, section_lines, revision)

    def revise_abstract(self, paper, revision):
        abstract_lines = paper._structured_md["abstract"]

        return self._revise_paper_passage(paper, abstract_lines, revision)

    def _validate_revision_suggestion(self, revision_suggestion, text_to_revise):
        if revision_suggestion is None:
            return False

        # validate the output
        valid_output = isinstance(revision_suggestion, list) and \
                       all("line" in rev for rev in revision_suggestion) and \
                       all(type(rev["line"]) in [int, float] for rev in revision_suggestion) and \
                       all("changed_line" in rev for rev in revision_suggestion)

        if valid_output:
            affected_lines = [rev["line"] for rev in revision_suggestion]

            p_lines = {int(re.findall(r"^line (\d+):", l)[0]): l for l in text_to_revise.split("\n")}
            p_line_numbers = list(p_lines.keys())

            valid_output = valid_output and set(affected_lines).issubset(set(p_line_numbers))
            if not valid_output:
                return False

            # revisions of more than 0.3 of the original length deviation are most likely an error
            original_lines = [p_lines[int(ix)] for ix in affected_lines]
            original_length = sum([len(ol) for ol in original_lines])
            revised_length = sum([len(r["changed_line"]) for r in revision_suggestion])

            valid_output = abs(original_length - revised_length) < 0.2 * original_length

        return valid_output

    def single_run(self, paper: Paper, **config) -> PaperCounterfactual:
        paper = paper.without_appendix()
        paper_sections = paper.get_sections()
        section_names = [section_name for section_name in paper_sections.keys()]

        altered_sections = {}
        changes = {}

        # gather batch inputs
        inputs = []
        input_index = []
        for section_name in section_names:
            if section_name.lower().strip() == "references":  # skip references
                continue

            paragraphs = paper.get_paragraphs(section_name, with_line_numbers=True)

            for p_i, p_text in enumerate(paragraphs):
                inputs += [{
                    "paper": p_text
                }]
                input_index += [(section_name, p_i, p_text)]

        # at random subselect the input to make up roughly error ratio% of the paper paragraphs
        subset_inputs = random.sample(range(len(inputs)), int(len(inputs) * self.error_ratio))
        inputs = [inputs[i] for i in subset_inputs]
        input_index = [input_index[i] for i in subset_inputs]

        # run the llm
        self.use_conversion_prompt()
        revision_suggestions = self.llm(inputs)

        # check outputs
        needs_revision = []
        suggestions = {}
        for i, revision_suggestion in enumerate(revision_suggestions):
            section_name, p_i, p_text = input_index[i]
            parsed_llm_suggestion = revision_suggestion[1]

            if parsed_llm_suggestion is not None:
                valid_output = self._validate_revision_suggestion(parsed_llm_suggestion, p_text)

                # if invalid, try to revise the output once more
                if not valid_output:
                    needs_revision += [i]
                else:
                    suggestions[section_name] = suggestions.get(section_name, []) + parsed_llm_suggestion
            else:
                logging.debug(
                    f"Paper {paper.id} section {section_name} could not be revised from passive to active. Leaving as is...")

        # revise the ones that need revision
        self.use_revision_prompt()
        revised = self.llm([{
            "incorrect_response": revision_suggestions[i][0]
            # pass the raw answer, not the failed parse (which is none)
        } for i in needs_revision])

        for i in range(len(needs_revision)):
            section_name, p_i, p_text = input_index[i]
            revision_suggestion = revised[i][1]

            if revision_suggestion is not None:
                valid_output = self._validate_revision_suggestion(revision_suggestion, p_text)

                if valid_output:
                    suggestions[section_name] = suggestions.get(section_name, []) + revision_suggestion
                else:
                    logging.debug(
                        f"Paper {paper.id} section {section_name} could not be revised to add language errors. Output error...")

        # apply changes
        for section_name, section_suggestions in suggestions.items():
            revised_section = self.revise_section(paper, section_name, section_suggestions)
            altered_sections[section_name] = revised_section
            changes[section_name] = section_suggestions

        # revise abstract
        abstract = paper.get_abstract(with_line_numbers=True)

        self.use_conversion_prompt()
        revision_suggestion = self.llm({
            "paper": abstract
        })

        revised_abstract = None
        abstract_changes = None
        if revision_suggestion is not None:
            valid_output = self._validate_revision_suggestion(revision_suggestion, abstract)
            # if invalid, try to revise the output once more
            if not valid_output:
                # try with revision prompt
                self.use_revision_prompt()
                revision_suggestion = self.llm({
                    "incorrect_response": revision_suggestion[0]  # use raw output
                })

            # check again
            valid_output = self._validate_revision_suggestion(revision_suggestion[1], abstract)

            if not valid_output:
                logging.info(f"Paper {paper.id} abstract could not be revised from passive to active. Output error...")
            else:
                revised_abstract = self.revise_abstract(paper, revision_suggestion[1])
                abstract_changes = revision_suggestion

        # create new paper from revisions
        overrides = [["sections", altered_sections[section], section] for section in altered_sections]
        if revised_abstract is not None:
            overrides.append(["abstract", revised_abstract])

        cf_paper = paper.override_multi(overrides)

        return PaperCounterfactual(
            o_paper=paper,
            counterfactual_type=self.name,
            cf_paper=cf_paper,
            changes={"highlevel": [{"target": "sec_" + section_name, "change": change} for section_name, change in
                                   changes.items()] \
                                  + [{"target": "abstract", "change": abstract_changes}],
                     "edits": overrides
                     }
        )
