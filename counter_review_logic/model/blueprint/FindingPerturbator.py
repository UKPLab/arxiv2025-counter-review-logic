import logging
from pathlib import Path

from ...data import Paper
from ...framework import PaperBlueprint, PaperSiteEngineer
from ...framework import PaperCounterfactual
from ...llm import ChatLLM
from ..cfgen import BlueprintBasedPerturbator


class FindingPerturbator(BlueprintBasedPerturbator):
    """
    Finding perturbator for generating counterfactuals that perturbs only the finding contributions of a paper while
    leaving everything else constant.

    """

    def __init__(self,
                 name: str,
                 llm: ChatLLM,
                 blueprint_dir: str | Path | None = None,
                 prompt_base_path: str | Path | None = None,
                 config=None):
        super().__init__(name="finding_perturbator." + name,
                         llm=llm,
                         blueprint_dir=blueprint_dir,
                         prompt_base_path=prompt_base_path,
                         config=config)

        self.add_prompt_paths({
            "classify": Path("deform") / "finding" / "classify.txt",
            "causation": Path("deform") / "finding" / "deform_causation.txt",
            "correlation": Path("deform") / "finding" / "deform_correlation.txt",
            "condition": Path("deform") / "finding" / "deform_condition.txt",
        })

        self.add_template_paths({
            "classify": Path("deform") / "finding" / "classify_template.json",
            "deform": Path("deform") / "finding" / "deform_template.json",
        })

    def new_cf(self, paper: Paper) -> PaperCounterfactual:
        blueprint = self.load_blueprint(paper)

        return self.deform(blueprint)

    def classify_findings(self, findings: list[dict], title: str, research_goal: dict) -> list:
        classifications = []
        for fin in findings:
            fin_info = {
                "finding_claim": fin["claim_summary"],
                "id": fin["id"],
                "literal_claim": fin["evidence"],
                "evidence_spans": [
                    co["span"] for co in fin["coreferences"]
                ] if "coreferences" in fin else []
            }

            # classify the conclusions and pick the most appropriate option for deforming the claim
            self.load_prompt("classify")
            classified = self.call_llm_structured_output(
                json_template=self.load_json_template("classify"),
                validate=lambda x: type(x) == dict and "claim_types" in x and "modifiers" in x,
                finding=fin_info,
                title=title,
                research_goal=research_goal
            )
            classifications += [classified]

        return classifications

    def list_possible_deformations(self, findings: list[dict], finding_components: list, title: str,
                                   research_goal: str) -> list[dict]:
        output = []

        def modify(mod_type, mod_name, fin_info, fin_comp, title, research_goal):
            self.load_prompt(mod_type)
            deform = self.call_llm_structured_output(
                json_template=self.load_json_template("deform"),
                validate=lambda x: type(x) == dict and "modified_finding" in x and "modification_steps" in x,
                finding=fin_info,
                finding_components=fin_comp,
                title=title,
                research_goal=research_goal
            )
            if deform is not None and type(deform) == dict and len(deform) > 0:
                deform["modification"] = mod_name
            else:
                return None

            return deform

        for fin, fin_comp in zip(findings, finding_components):
            fin_info = {
                "finding_claim": fin["claim_summary"],
                "id": fin["id"],
                "literal_claim": fin["evidence"],
                "evidence_spans": [
                    co["span"] for co in fin["coreferences"]
                ] if "coreferences" in fin else []
            }

            # check for modifiers, they are the easiest to deform
            if "modifiers" in fin_comp and len(fin_comp["modifiers"]) > 0:
                mod = [m.lower().strip() for m in fin_comp["modifiers"]]

                # apply deformations in sequence of their complexity whichever applies first
                if "correlation relation" in mod:
                    mod_type = "correlation"
                    mod_name = "Turn a correlation into a causation"

                    deform = modify(mod_type, mod_name, fin_info, fin_comp, title, research_goal)
                    if deform is not None:
                        output.append({**fin, **deform})
                        continue

                if "causal relation" in mod:
                    mod_type = "causation"
                    mod_name = "Invert the order of causation"

                    deform = modify(mod_type, mod_name, fin_info, fin_comp, title, research_goal)
                    if deform is not None:
                        output.append({**fin, **deform})
                        continue

                if "condition" in mod:
                    mod_type = "condition"
                    mod_name = "Generalize a claim by discarding constraints"

                    deform = modify(mod_type, mod_name, fin_info, fin_comp, title, research_goal)
                    if deform is not None:
                        output.append({**fin, **deform})
                        continue

            # if there are no modifiers, do not apply deformation

        return output

    def deform(self, blueprint: PaperBlueprint) -> PaperCounterfactual | None:
        if not blueprint.is_complete():
            logging.warning(f"Blueprint for {blueprint.paper.id} is not complete, cannot deform conclusions without results and findings.")
            return None

        finding_contribution_claims = blueprint.findings_contributions
        finding_contribution_claims = list(sorted(finding_contribution_claims, key=lambda x: x["score"], reverse=True))

        # 1) Iterate over finding contributions as potential candidates and check if any of them fit a deformation rule
        classified_findings = self.classify_findings(finding_contribution_claims,
                                                     blueprint.paper.get_title(),
                                                     blueprint.research_goal["research_goal"])
        findings_to_deform = self.list_possible_deformations(finding_contribution_claims, classified_findings,
                                                             blueprint.paper.get_title(),
                                                             blueprint.research_goal["research_goal"])

        # 2) Select the finding with the highest score that can be deformed
        findings_to_deform = list(sorted(findings_to_deform, key=lambda x: x["score"], reverse=True))

        if len(findings_to_deform) == 0:
            return None

        # pick the top ranked finding to twist iterating over the ordered list
        for f in findings_to_deform:
            fin = f
            original_fin = next((f for f in finding_contribution_claims if f["id"] == fin["id"]), None)

            if original_fin is None or "modification_steps" not in fin or "modified_finding" not in fin:
                continue

            change_log = {
                "focus_on_finding": original_fin["id"],
                "operation": "Break the logical chain by extending the claim of the finding contribution",
                "break_target": original_fin,
                "break_result": fin
            }

            # 3) redact the finding claim in the paper
            # bring into format for redaction
            modified_content = {
                "modification_type": fin["modification"],
                "modification_steps": fin["modification_steps"],
                "modified_finding": fin["modified_finding"],
                "reasoning": {
                    "restriction_analysis": fin["restriction_analysis"] if "restriction_analysis" in fin else None,
                    "generalization": fin["generalization"] if "generalization" in fin else None,
                },
                "explanation": fin["explanation"] if "explanation" in fin else None
            }

            logging.debug(f"Identified finding to deform {original_fin['id']}")
            logging.debug(f"Deformation: {modified_content['modified_finding']}")

            _, underlying_conclusions, underlying_results, underlying_methods = blueprint.get_building_block(fin["id"])
            underlying_results = list({r["id"]: r for rs in underlying_results.values() for r in rs}.values())
            underlying_methods = list({r["step_number"]: r for rs in underlying_methods.values() for r in rs}.values())

            redactions = self.redact(original_fin,
                                     modified_content,
                                     underlying_results,
                                     underlying_conclusions,
                                     blueprint)

            logging.info(f"Redacted paper at {len(redactions)} locations" if redactions is not None else "Redaction failed!")
            if redactions is None:
                logging.warning(f"Redaction failed for {original_fin['id']}. No redactions were found.")
                continue

            change_log["core_redactions"] = redactions
            change_log["cascaded_redactions"] = []

            # merge redactions within itself
            overrides, detailed_changes, originals = self.merge_paper_changes(redactions=redactions)

            logging.info(f"Edited paper at {len(redactions)} passages")

            # no cascading necessary

            # 5) validate the deformation, if accepted return
            bb_graph = {
                "finding": original_fin,
                "conclusion": underlying_conclusions,
                "result": underlying_results,
                "method": underlying_methods
            }

            redacted_paper, edits = self.validate_and_realize_redactions(
                blueprint=blueprint,
                overrides=overrides,
                originals=originals,
                detailed_changes=detailed_changes,
                original_bb=original_fin,
                revised_bb=modified_content,
                bb_graph=bb_graph,
                constant_bbs=underlying_conclusions + underlying_results,
                cascaded_redactions=None
            )
            change_log["edits"] = edits
            change_log["considered_edits"] = overrides

            logging.debug(f"Redacted paper {redacted_paper} with changes {change_log}")

            if redacted_paper is not None:
                redacted_paper.changes = change_log
                return redacted_paper

        logging.error("Failed to create a finding perturbation. No version was acceptable.")

        return None

    def redact(self,
               original_bb: dict,
               revised_bb: dict,
               underlying_results: list[dict],
               underlying_conclusions: list[dict],
               blueprint: PaperBlueprint) -> list | None:
        paper = blueprint.paper.without_appendix()

        # 1) get inputs for redaction
        # get all relevant passages from the paper
        locations = []
        for loc in original_bb["location"]:
            aligned = PaperSiteEngineer.get_paragraphs_from_evidence_location(loc, paper)

            if aligned is not None:
                locations += aligned

        if len(locations) == 0:
            # try to find the location in the paper text by span as a fallback
            aligned = PaperSiteEngineer.get_paragraph_from_evidence_span(original_bb["evidence"], paper)
            if aligned is not None:
                locations += aligned

        if len(locations) == 0:
            return None

        # reformat the bb to adhere expected format
        original_bb_formatted = {**original_bb}
        original_bb_formatted["summary"] = original_bb["claim_summary"]
        del original_bb_formatted["claim_summary"]

        # produce a list of constant bbs (results + conclusions)
        constant_bbs = []
        for r in underlying_results:
            constant_bbs += [{
                "id": r["id"],
                "type": "result",
                "summary": r["result_summary"],
                "location": r["evidences"],
                "relation": r["relevance_to_conclusion"],
                #"mentions": [
                #    {
                #        "location": f"paragraph{co['location']}",
                #        "span": co["span"]
                #    } for co in r["coreferences"]
                #]
            }]

        for c in underlying_conclusions:
            constant_bbs += [{
                "id": c["id"],
                "type": "conclusion",
                "summary": c["conclusion_summary"],
                "location": c["evidences"],
                "relation": c["relevance_to_claim"],
                #"mentions": [
                #    {
                #        "location": f"paragraph{co['location']}",
                #        "span": co["span"]
                #    } for co in c["coreferences"]
                #]
            }]

        # 2) get redactions for the paper
        # get redactions of the paper
        redactions = self.suggest_paper_changes_for_bb_revision(
            bb_type="finding contribution",
            original_bb=original_bb_formatted,
            revised_bb=revised_bb,
            locations=locations,
            constant_bb_types=["conclusions", "results", "methods"],
            blueprint=blueprint,
            constant_bbs=constant_bbs,
            ensure_constant=False  # todo for now: no validation since the LLM is bad at it
        )

        return redactions
