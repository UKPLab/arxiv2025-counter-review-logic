import logging
import os
import re
from pathlib import Path
from typing import Callable

from cerg.data import Paper
from cerg.framework.blueprint import PaperBlueprint, PaperSiteEngineer
from cerg.framework.cfg import PaperCounterfactual
from cerg.framework.picfg import PerturbatorInspectorCF, Perturbator
from cerg.llms import ChatLLM, parse_llm_output_as_single_json
from cerg.llms.utils import approximately_truncate


class BlueprintBasedPerturbator(Perturbator):
    def __init__(self,
                 name: str,
                 llm: ChatLLM,
                 blueprint_dir: str | Path | None = None,
                 prompt_base_path: str | Path | None = None,
                 config=None):
        super().__init__(name=name)
        self.llm = llm
        self.max_parsing_attempts = 5

        # check blueprint dir
        if blueprint_dir is None:
            if "BLUEPRINT_DIR" in os.environ:
                blueprint_dir = Path(os.environ["BLUEPRINT_DIR"].replace("\"", ""))
            else:
                raise ValueError("Blueprint path is not provided.")

        if type(blueprint_dir) == str:
            blueprint_dir = Path(blueprint_dir)

        if not blueprint_dir.exists():
            raise ValueError(f"Blueprint path {blueprint_dir} does not exist.")

        self.config = config if config is not None else {}

        self.blueprint_dir = blueprint_dir
        self.bp_cache = {}

        self.prompt_base_path = prompt_base_path
        if prompt_base_path is None and "PROMPT_DIR" in os.environ:
            self.prompt_base_path = Path(os.environ["PROMPT_DIR"].replace("\"", "")) / "cfgen" / "blueprint_based_cfgen"
        elif prompt_base_path is None:
            raise ValueError("Prompt base path is not provided.")
        elif type(prompt_base_path) == str:
            self.prompt_base_path = Path(self.prompt_base_path)

        assert self.prompt_base_path.exists(), f"the prompt base path needs to exist. It does not: {self.prompt_base_path}."

        self.prompt_paths = {
            "redact": {
                "edit": Path("redact") / "surgical_edit.txt",
                "coreference": Path("redact") / "coreference_edit.txt",
                "apply": Path("redact") / "apply_edit.txt",
                "apply_table": Path("redact") / "apply_edit_table.txt",
                "check_edits_constants": Path("redact") / "check_edits_constants.txt",
                "revise_edit": Path("redact") / "revise_edit.txt",
                "validate_paper": Path("redact") / "validate_redaction.txt",
                "filter_edits": Path("redact") / "filter_edits.txt",
            }
        }
        self.template_paths = {
            "redact": {
                "edit": Path("redact") / "surgical_edit.json",
                "coreference": Path("redact") / "coreference_edit.json",
                "check_edits_constants": Path("redact") / "check_edits_constants.json",
                "filter_edits": Path("redact") / "filter_edits.json",
            }
        }

    def add_prompt_paths(self, prompt_hierarchy: dict[str, str | Path]):
        """
        Add prompt paths to the existing prompt paths. This is used to add additional prompts to the perturbator.
        :param prompt_hierarchy: A dictionary with the prompt hierarchy.
        """
        assert type(prompt_hierarchy) == dict, "Prompt hierarchy needs to be a dictionary."
        self.prompt_paths.update(prompt_hierarchy)

    def add_template_paths(self, template_hierarchy: dict[str, str | Path]):
        """
        Add template paths to the existing template paths. This is used to add additional templates to the perturbator.
        :param template_hierarchy: A dictionary with the template hierarchy.
        """
        assert type(template_hierarchy) == dict, "Template hierarchy needs to be a dictionary."
        self.template_paths.update(template_hierarchy)

    def load_prompt(self, prompt_name: str):
        if prompt_name == "revise_valid_json":
            prompt_path = "revise_valid_json.txt"
        else:
            compounds = prompt_name.split(":")

            hierarchy = self.prompt_paths
            for c in compounds:
                if type(hierarchy) == dict and c in hierarchy:
                    hierarchy = hierarchy[c]
                else:
                    raise ValueError(f"Invalid prompt name {prompt_name}. Does not exist in {self.prompt_paths}.")

            if hierarchy is None or type(hierarchy) == dict:
                raise ValueError(
                    f"Invalid prompt name {prompt_name}. Does not terminate in a single prompt, but {hierarchy}.")

            prompt_path = hierarchy

        self.llm.load_prompt(self.prompt_base_path / prompt_path)

    def load_json_template(self, template_name: str):
        compounds = template_name.split(":")

        hierarchy = self.template_paths
        for c in compounds:
            if type(hierarchy) == dict and c in hierarchy:
                hierarchy = hierarchy[c]
            else:
                raise ValueError(f"Invalid template name {template_name}. Does not exist in {self.template_paths}.")

        if hierarchy is None or type(hierarchy) == dict:
            raise ValueError(
                f"Invalid prompt name {template_name}. Does not terminate in a single prompt, but {hierarchy}.")

        json_template_path = hierarchy

        with open(self.prompt_base_path / json_template_path, 'r', encoding='utf-8') as f:
            json_template = f.read()

        return json_template

    def load_blueprint(self, paper: Paper):
        if paper.id in self.bp_cache:
            return self.bp_cache[paper.id]

        # otherwise load from disc
        files = list(self.blueprint_dir.glob(f"{paper.id}.json"))
        if len(files) == 0 or len(files) > 1:
            raise ValueError(f"Blueprint for {paper.id} not found or duplicated in {self.blueprint_dir}.")
        self.bp_cache[paper.id] = PaperBlueprint.load_json(files[0], paper)

        return self.bp_cache[paper.id]

    def call_llm_unstructured_output(self, **params):
        return self.llm(params)

    def call_llm_structured_output(self, json_template: str, validate: Callable = None, **params):
        if validate is None:
            validate = lambda x: True

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

    def revise_cf(self, paper: Paper, feedback: str, previous: PaperCounterfactual) -> PaperCounterfactual:
        raise NotImplementedError("This method should be implemented in the subclass.")  # todo implement this here

    def new_cf(self, paper: Paper) -> PaperCounterfactual:
        raise NotImplementedError("This method should be implemented in the subclass.")

    def __call__(self,
                 paper: Paper,
                 feedback: str = None,
                 previous_solution: PaperCounterfactual = None) -> PaperCounterfactual:
        # if you get feedback, go into revision mode
        if feedback is not None:
            return self.revise_cf(paper, feedback, previous_solution)

        # otherwise generate a new CF
        return self.new_cf(paper)

    def apply_changes_to_paper(self, overrides: list[dict],
                               detailed_changes: list[dict],
                               blueprint: PaperBlueprint) -> PaperCounterfactual:
        paper = blueprint.paper.without_appendix()
        cf_paper = paper.override_multi(overrides)

        return PaperCounterfactual(
            o_paper=paper,
            counterfactual_type=f"{self.name}",
            cf_paper=cf_paper,
            changes=detailed_changes
        )

    def merge_paper_changes(self, redactions: list[dict]) -> tuple[list, list, list]:
        # sync all changes to the same location into one edit
        changes_per_location = {}
        for red in redactions:
            revised_text = red["revision"]
            original_text = red["original_text"]
            etype = red["loc_type"]
            enumber = red["loc_id"]

            if revised_text.lower().strip() == original_text.lower().strip():
                continue

            if f"{etype}.{enumber}" not in changes_per_location:
                changes_per_location[".".join([str(etype), str(enumber)])] = []

            changes_per_location[".".join([str(etype), str(enumber)])] += [red]

        # apply the changes to the original text
        changes = []
        overrides = []
        originals = []
        for loc, reds in changes_per_location.items():
            etype, enumber = loc.split(".")

            original_text = reds[0]["original_text"]
            revisions = [{"revision": red["revision"], "detailed_edits": red["detailed_edits"]} for red in reds]

            # apply either with the redact:apply prompt or the redact:apply_table prompt
            if etype == "table":
                self.load_prompt("redact:apply_table")
                revised_text = None
                answer = self.call_llm_unstructured_output(
                    passage_to_change=original_text,
                    changes_to_apply=revisions
                )
                match = re.search("```markdown(.*)```", answer, re.DOTALL)
                if match:
                    revised_text = match.group(1).strip()
                else:
                    match = re.search(r"\* \*Revised Table\*:(.*)", answer, re.DOTALL)
                    if match:
                        revised_text = match.group(1).strip()
                if revised_text is None:
                    logging.debug("ERROR: Could not parse the revised table from the LLM output.")
                    continue
            else:
                self.load_prompt("redact:apply")
                revised_text = self.call_llm_unstructured_output(
                    passage_to_change=original_text,
                    changes_to_apply=revisions
                )

            originals += [(etype, original_text, enumber)]
            overrides += [(etype, revised_text, enumber)]
            changes += [{
                "target": f"{etype}_{enumber}",
                "change": {
                    "original_text": original_text,
                    "revised_text": revised_text,
                    "detailed_edits": reds
                }
            }]

        return overrides, changes, originals

    def suggest_paper_changes_for_bb_revision(self,
                                              bb_type: str,
                                              original_bb: dict,
                                              revised_bb: dict,
                                              locations: list[dict],
                                              constant_bb_types: list[str],
                                              blueprint: PaperBlueprint,
                                              constant_bbs: list[dict] = None,
                                              ensure_constant=False):
        paper = blueprint.paper

        # verify input format
        assert bb_type in ["finding contribution", "conclusion", "result",
                           "method"], f"Invalid building block type {bb_type}."
        assert "summary" in original_bb, "Original building block does not contain a summary."
        assert "evidences" in original_bb or "evidence" in original_bb, "Original building block does not contain an evidence span."
        assert "coreferences" in original_bb, "Original building block does not contain coreferences."

        assert all(bb in ["finding contributions", "conclusions", "results", "methods"] for bb in constant_bb_types), \
            f"Invalid building block types {constant_bb_types}. Only finding, conclusion, and result are allowed."

        if constant_bbs:
            assert all(bb["type"] in ["finding contribution", "conclusion", "result"] for bb in constant_bbs)
            assert all("summary" in bb for bb in constant_bbs), "Constant building blocks do not contain a summary."
            assert all("location" in bb for bb in constant_bbs), "Constant building blocks do not contain a location."
            assert all("relation" in bb for bb in constant_bbs), "Constant building blocks do not contain a relation."
            assert all("id" in bb for bb in constant_bbs), "Constant building blocks do not contain a relation."

        passages_to_change = []
        evidences = original_bb["evidences"] if "evidences" in original_bb else original_bb["evidence"]
        for i, loc in enumerate(locations):
            if loc[0] == "paragraph":
                passages_to_change += [
                    f"Passage to revise no. {i} (paragraph {loc[1]}): {loc[2]}.\n-> Claim spans: '{evidences}'"]
            elif loc[0] == "table":
                passages_to_change += [f"Passage to revise no. {i} (Table {loc[1]}): {loc[2]}."]
            elif loc[0] == "figure":
                passages_to_change += [f"Passage to revise no. {i} (Figure {loc[1]}): {loc[2]}."]

        constant_bb_type_str = ", ".join(constant_bb_types)

        self.load_prompt("redact:edit")
        redaction = self.call_llm_structured_output(
            json_template=self.load_json_template("redact:edit"),
            validate=lambda x: type(x) == list and len(x) > 0 and all(
                type(y) == dict and "passage_id" in y and "revised_passage" in y and "detailed_edits" in y for y in x),
            title=paper.get_title(),
            abstract=paper.get_abstract().replace("#", ""),
            research_goal=blueprint.research_goal["research_goal"],
            bb_type=bb_type,
            bb_original=original_bb["summary"],
            bb_revised=revised_bb,
            constant_bb_types=constant_bb_type_str,
            constat_bbs=constant_bbs,
            passages_to_change=passages_to_change
        )

        if redaction is None:
            logging.error("Redaction failed. No changes suggested.")
            return None

        redactions = []
        locations_numbered = dict(enumerate(locations))
        for red in redaction:
            if red["passage_id"] not in locations_numbered:  # erroneous passage_id, skip
                continue

            location = locations_numbered[red["passage_id"]]

            redactions += [{
                "loc_type": location[0],
                "loc_id": location[1],
                "original_text": location[2],
                "revision": red["revised_passage"],
                "detailed_edits": red["detailed_edits"]
            }]

        # handle coreferences
        coreferences = []
        for co in original_bb["coreferences"]:
            co_location = co["location"]
            if type(co_location) == str:
                try:
                    co_location = int(co_location)
                except ValueError:
                    continue

            aligned = paper.get_paragraph_by_number(co_location, with_line_numbers=False)[1]
            if aligned is not None:
                coreferences += [(aligned, co)]

        coreferences_to_change = [
            f"Coreference {i} with span '{l[1]['span']}': {l[0]}" for i, l in enumerate(coreferences)
        ]

        self.load_prompt("redact:coreference")
        coreference_redaction = self.call_llm_structured_output(
            json_template=self.load_json_template("redact:coreference"),
            validate=lambda x: type(x) == list and all(
                type(y) == dict and "coreference_id" in y and "revised_passage" in y and "detailed_edits" in y for y in
                x),
            title=paper.get_title(),
            abstract=paper.get_abstract().replace("#", ""),
            research_goal=blueprint.research_goal["research_goal"],
            bb_type=bb_type,
            bb_original=original_bb["summary"],
            bb_revised=revised_bb,
            constant_bb_types=constant_bb_type_str,
            constant_bbs=constant_bbs,
            changed_passages=redaction,
            coreferences_to_change=coreferences_to_change
        )
        if coreference_redaction is None:
            coreference_redaction = []

        # merge into the change log
        coreferences_numbered = dict(enumerate(coreferences))
        for red in coreference_redaction:
            try:
                coref = coreferences_numbered[red["coreference_id"]]
            except KeyError:
                continue

            redactions += [{
                "loc_type": "paragraph",
                "loc_id": coref[1]["location"],
                "original_text": coref[0],
                "revision": red["revised_passage"],
                "detailed_edits": red["detailed_edits"]
            }]

        # if the constant building blocks are provided, subselect or fix the redactions
        if ensure_constant and constant_bbs is not None:  # fixme this prompt renders useless results since it detects inconsistencies in the first place, which is what we want.
            self.load_prompt("redact:check_edits_constants")
            feedback = self.call_llm_structured_output(
                json_template=self.load_json_template("redact:check_edits_constants"),
                validate=lambda x: type(x) == list and all(
                    type(y) == dict and "modification_id" in y and "violation" in y for y in x),
                title=paper.get_title(),
                abstract=paper.get_abstract().replace("#", ""),
                research_goal=blueprint.research_goal["research_goal"],
                bb_type=bb_type,
                bb_original=original_bb,
                bb_revised=revised_bb,
                constant_bb_types=constant_bb_type_str,
                constant_bbs=constant_bbs,
                changed_passages=[{"id": i, **red} for i, red in enumerate(redactions)],
            )

            if feedback is not None and len(feedback) > 0:
                reds_to_revise = []
                for f in feedback:
                    if not f["violation"]:
                        continue

                    ired, red = next((i for i, r in enumerate(redactions) if str(i) == str(f["modification_id"])), None)
                    if red is None:
                        continue

                    reds_to_revise += [(ired, red, f)]

                reds_to_replace = []
                for ired, red, feedback in reds_to_revise:
                    self.load_prompt("redact:revise_edit")
                    revised_redaction = self.call_llm_unstructured_output(
                        title=paper.get_title(),
                        abstract=paper.get_abstract().replace("#", ""),
                        research_goal=blueprint.research_goal["research_goal"],
                        original_passage=red["original_text"],
                        revised_passage=red["revised_passage"],
                        feedback=feedback
                    )

                    if revised_redaction is not None and revised_redaction.lower().strip() != red[
                        "revised_passage"].lower().strip():
                        reds_to_replace += [(ired, {
                            "loc_type": red["loc_type"],
                            "loc_id": red["loc_id"],
                            "original_text": red["original_text"],
                            "revision": revised_redaction,
                            "detailed_edits": red["detailed_edits"]
                        })]

                for ired, red in reds_to_replace:
                    redactions[ired] = red

        if len(redactions) == 0:
            return None

        return redactions

    def validate_and_realize_redactions(self,
                                        blueprint: PaperBlueprint,
                                        overrides: list,
                                        originals: list,
                                        detailed_changes: list[dict],
                                        original_bb: dict,
                                        revised_bb: dict,
                                        bb_graph: dict,
                                        constant_bbs: dict = None,
                                        cascaded_redactions: list[dict] = None,
                                        with_filtering=False
                                        ) -> tuple[PaperCounterfactual|None, list]:
        if with_filtering:
            filtered = []
            for ov, org in zip(overrides, originals):
                self.load_prompt("redact:filter_edits")
                label = self.call_llm_structured_output(
                    json_template=self.load_json_template("redact:filter_edits"),
                    validate=lambda x: type(x) == dict and "is_appropriate" in x and type(x["is_appropriate"]) == bool,
                    original_text=org[1],
                    revised_text=ov[1]
                )

                if label["is_appropriate"]:
                    filtered += [ov]

            # recover from supposedly too restrictive filtering
            if len(filtered) == 0:
                filtered = overrides
        else:
            filtered = overrides

        # revise the paper
        revised_paper = self.apply_changes_to_paper(
            overrides=filtered,
            detailed_changes=detailed_changes,
            blueprint=blueprint
        )

        self.load_prompt("redact:validate_paper")
        feedback = self.call_llm_unstructured_output(
            original_paper=approximately_truncate(blueprint.paper.without_appendix().md,self.llm, margin=120*1000),
            revised_paper=approximately_truncate(revised_paper.cf_paper.without_appendix().md, self.llm, margin=120*1000),
            edit_intention={
                "goal": "Break the logical chain from results to conclusions to finding contribution claims",
                "underlying_logical_graph": f"We find '{bb_graph['finding']}' as we show by our interpretation '{bb_graph['conclusion']}' based on our results that show '{bb_graph['result']}'. The results are produced by following these steps: '{bb_graph['method']}'.",
                "point_of_attack": original_bb,
                "revision_to_point_of_attack": revised_bb
            },
            specific_changes=detailed_changes,
            protected_elements=constant_bbs if constant_bbs is not None else "None",
        )

        verdict_match = re.search(r"### Final Verdict: (.+)\n", feedback, re.DOTALL)
        if verdict_match is None:
            return revised_paper, filtered

        verdict = verdict_match.group(1).strip().lower()
        if "reject" in verdict:
            return None, filtered

        # use feedback to revise
        if "accept" in verdict:
            return revised_paper, filtered

        # todo implement feedback cycle (for now, treat revision needs as accept)

        return revised_paper, filtered


class BlueprintBasedCF(PerturbatorInspectorCF):
    def __init__(self, name, config=None):
        super().__init__(name=name, perturbator=None, inspector=None, config=config)

        self._setup()

    def set_perturbator(self, pert: Perturbator):
        self.perturbator = pert
