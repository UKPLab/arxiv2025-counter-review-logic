import copy
import logging
from pathlib import Path

from cerg.data import Paper
from cerg.framework.blueprint import PaperBlueprint, PaperSiteEngineer
from cerg.framework.cfg import PaperCounterfactual
from cerg.llms import ChatLLM
from cerg.models.cfgens.blueprint.BlueprintBasedCF import BlueprintBasedPerturbator
from cerg.models.cfgens.blueprint.FindingPerturbator import FindingPerturbator


class ConclusionPerturbator(BlueprintBasedPerturbator):
    """
    Conclusion perturbator for generating counterfactuals that perturbs only the conclusion of a paper while
    adapting findings but not the underlying results.

    """

    def __init__(self,
                 name: str,
                 llm: ChatLLM,
                 blueprint_dir: str | Path | None = None,
                 prompt_base_path: str | Path | None = None,
                 config=None):
        super().__init__(name="conclusion_perturbator." + name,
                         llm=llm,
                         blueprint_dir=blueprint_dir,
                         prompt_base_path=prompt_base_path,
                         config=config)

        self.add_prompt_paths({
            "deform": Path("deform") / "conclusion" / "base_prompt.txt",
            "new_result": Path("deform") / "conclusion" / "new_result.txt",
            "rank": Path("deform") / "conclusion" / "rank_deform.txt",
            "cascade": Path("cascade") / "con_to_fins.txt"
        })

        self.add_template_paths({
            "deform": Path("deform") / "conclusion" / "base_template.json",
            "new_result": Path("deform") / "conclusion" / "new_result.json",
            "rank": Path("deform") / "conclusion" / "rank_template.json",
            "cascade": Path("cascade") / "con_to_fins.json"
        })

    def new_cf(self, paper: Paper) -> PaperCounterfactual:
        blueprint = self.load_blueprint(paper)

        return self.deform(blueprint)

    def list_possible_deformations(self,
                                   conclusions: list[dict],
                                   results: dict[list],
                                   blueprint: PaperBlueprint) -> list[dict]:
        output = []

        for conclusion in conclusions:
            relevant_results = results[conclusion["id"]]

            results_summary = [
                f"* Result {r['id']}:\n*Summary*: {r['result_summary']}\n*Relevance to conclusion*: {r['relevance_to_conclusion']}"
                for r in relevant_results
            ]

            self.load_prompt("new_result")
            new_result = self.call_llm_structured_output(
                json_template=self.load_json_template("new_result"),
                conclusion={
                    "conclusion_summary": conclusion["conclusion_summary"],
                    "context": conclusion["context"],
                    "evidences": conclusion["coreferences"]
                },
                results=results_summary,
                paper_title=blueprint.paper.get_title(),
                research_goal=blueprint.research_goal["research_goal"]
            )

            if new_result is None:
                output += [None]
                continue

            # produce deformed conclusion covering the new result
            self.load_prompt("deform")
            deformed_conclusion = self.call_llm_structured_output(
                json_template=self.load_json_template("deform"),
                validate=lambda x: type(
                    x) == dict and "conclusion_summary" in x and "changes" in x and "accounting_for_new_result" in x,
                conclusion={
                    "conclusion_summary": conclusion["conclusion_summary"],
                    "context": conclusion["context"],
                    "evidences": conclusion["coreferences"]
                },
                results=results_summary,
                new_result=new_result
            )

            if deformed_conclusion is not None:
                deformed_conclusion["new_result"] = new_result

            output += [deformed_conclusion]

        return output

    def rank_deformations(self,
                          conclusion_deformations: list[dict],
                          conclusions: list[dict],
                          results: dict[str, list],
                          blueprint: PaperBlueprint) -> dict | None:
        results_summary = [
            f"* Result {r['id']}:\n*Summary*: {r['result_summary']}\n*Relevance to conclusion*: {r['relevance_to_conclusion']}"
            for rs in results.values()
            for r in rs
        ]

        conclusions_with_modifications = []
        for original_conclusion, deformed_conclusion in zip(conclusions, conclusion_deformations):
            if deformed_conclusion is None:
                continue

            conclusions_with_modifications += [{
                "conclusion_id": original_conclusion["id"],
                "original_conclusion_summary": original_conclusion["conclusion_summary"],
                "modified_conclusion_summary": deformed_conclusion["conclusion_summary"],
                "changes": deformed_conclusion["changes"],
                "accounting_for_new_result": deformed_conclusion["accounting_for_new_result"]
            }]

        # rank
        self.load_prompt("rank")
        ranked = self.call_llm_structured_output(
            json_template=self.load_json_template("rank"),
            validate=lambda x: type(x) == list and len(x) == len(conclusions_with_modifications) and all(
                type(y) == dict and "score" in y and "conclusion_id" in y for y in x),
            conclusions_with_modifications=conclusions_with_modifications,
            results=results_summary,
            paper_title=blueprint.paper.get_title(),
            research_goal=blueprint.research_goal["research_goal"]
        )

        # take the top modification
        if ranked is None or len(ranked) == 0:
            return None

        best = list(sorted(ranked, key=lambda x: x["score"], reverse=True))[0]
        best_deformation = next(({"id": co["id"], **cm} for co, cm in zip(conclusions, conclusion_deformations) if
                                 co["id"] == best["conclusion_id"]), None)

        return best_deformation

    def redact(self,
               original_bb: dict,
               revised_bb: dict,
               underlying_results: list[dict],
               blueprint: PaperBlueprint):
        paper = blueprint.paper.without_appendix()

        # 1) get inputs for redaction
        # get all relevant passages from the paper
        locations = []
        for loc in original_bb["evidences"]:
            aligned = PaperSiteEngineer.get_paragraphs_from_evidence_location(loc["location"], paper)

            if aligned is None:
                aligned = PaperSiteEngineer.get_paragraph_from_evidence_span(loc["span"], paper)

            if aligned is not None:
                locations += aligned

        if len(locations) == 0:
            return None

        # reformat the bb to adhere expected format
        original_bb_formatted = {**original_bb}
        original_bb_formatted["summary"] = original_bb["conclusion_summary"]
        del original_bb_formatted["conclusion_summary"]

        # produce a list of constant bbs (results)
        constant_bbs = []
        for r in underlying_results:
            constant_bbs += [{
                "id": r["id"],
                "type": "result",
                "summary": r["result_summary"],
                "location": r["evidences"],
                "relation": r["relevance_to_conclusion"],
                "mentions": [
                    {
                        "location": f"paragraph{co['location']}",
                        "span": co["span"]
                    } for co in r["coreferences"]
                ]
            }]

        # 2) get redactions for the paper
        # get redactions of the paper
        redactions = self.suggest_paper_changes_for_bb_revision(
            bb_type="conclusion",
            original_bb=original_bb_formatted,
            revised_bb=revised_bb,
            locations=locations,
            constant_bb_types=["results", "methods"],
            blueprint=blueprint,
            constant_bbs=constant_bbs,
            ensure_constant=False  # todo for now: no validation since the LLM is bad at it
        )

        return redactions

    def cascade(self, original_conclusion: dict, revised_conclusion: dict, associated_findings: list[dict],
                blueprint: PaperBlueprint):
        modified_content = {
            "modification_type": "Adding a new virtual result and revising the conclusion accordingly",
            "modification_steps": revised_conclusion["changes"],
            "modified_conclusion": revised_conclusion["conclusion_summary"],
            "reasoning": revised_conclusion[
                "accounting_for_new_result"] if "accounting_for_new_result" in revised_conclusion else None,
            "explanation": revised_conclusion[
                "logical_relation_revised_conclusion_to_old_conclusion"] if "logical_relation_revised_conclusion_to_old_conclusion" in revised_conclusion else None
        }

        self.load_prompt("cascade")
        revised_findings = self.call_llm_structured_output(
            json_template=self.load_json_template("cascade"),
            validate=lambda x: type(x) == list and all(
                type(y) == dict and "contribution_claim_id" in y and "revised_claim" in y for y in x),
            original_conclusion=original_conclusion,
            revised_conclusion=modified_content,
            contribution_claims=associated_findings,
            paper_title=blueprint.paper.get_title(),
            research_goal=blueprint.research_goal["research_goal"]
        )

        # produce redactions for the findings with the findings perturbator
        fp = FindingPerturbator(
            name="finding_perturbator",
            llm=self.llm,
            blueprint_dir=self.blueprint_dir,
            prompt_base_path=self.prompt_base_path
        )
        blueprint_revised = copy.deepcopy(blueprint)
        blueprint_revised.conclusions[original_conclusion["id"]] = {
            "id": original_conclusion["id"],
            "conclusion_summary": revised_conclusion["conclusion_summary"],
            "context": original_conclusion["context"],
            "evidences": original_conclusion["evidences"],
            "associated_contribution_claims": original_conclusion["associated_contribution_claims"],
            "relevance_to_claim": original_conclusion["relevance_to_claim"],
            "coreferences": [{"location": r["location"], "span": r["revised_evidence"]} for r in
                             revised_conclusion["revisions_to_evidences"]]
        }

        output_redactions = []
        for rf in revised_findings:
            if rf["revised_claim"] is None:
                continue

            original_fin = next((f for f in associated_findings if f["id"] == rf["contribution_claim_id"]), None)
            if original_fin is None:
                continue

            # no changes needed, skip
            if rf["revised_claim"].lower().strip() == original_fin["claim_summary"].lower().strip():
                continue

            modified_fin = {
                "modification_type": "Update finding to align with revised conclusion",
                "modification_steps": rf["modification_steps"],
                "modified_finding": rf["revised_claim"],
                "reasoning": rf[
                    "effect_of_new_conclusion_on_claim"] if "effect_of_new_conclusion_on_claim" in rf else None,
                "explanation": rf["explanation"] if "explanation" in rf else None
            }
            _, underlying_conclusions, underlying_results, underlying_methods = blueprint_revised.get_building_block(
                original_fin["id"])
            underlying_results = list({r["id"]: r for rs in underlying_results.values() for r in rs}.values())

            redactions = fp.redact(
                original_bb=original_fin,
                revised_bb=modified_fin,
                underlying_results=underlying_results,
                underlying_conclusions=underlying_conclusions,
                blueprint=blueprint_revised
            )
            if redactions is not None:
                output_redactions += redactions

        return output_redactions

    def deform(self, blueprint: PaperBlueprint):
        if not blueprint.is_complete():
            logging.warning(f"Blueprint for {blueprint.paper.id} is not complete, cannot deform conclusions without results and findings.")
            return None

        finding_contribution_claims = blueprint.findings_contributions
        finding_contribution_claims = list(sorted(finding_contribution_claims, key=lambda x: x["score"], reverse=True))

        # 1) Iterate over findings in the order of their score
        for fin in finding_contribution_claims:
            fin_id = fin["id"]

            change_log = {
                "focus_on_finding": fin_id,
                "operation": "Break the logical chain by changing a conclusion to disalign with the underlying result. Cascade to findings."
            }

            # 2) get their underlying components
            _, conclusions, results, steps = blueprint.get_building_block(fin_id)

            # 3) for each conclusion and its results, develop a plan to deform it
            conclusion_deformations = self.list_possible_deformations(conclusions, results, blueprint)

            # 4) assess and select the most plausible plan
            best_conclusion_deformation = self.rank_deformations(conclusion_deformations,
                                                                 conclusions,
                                                                 results,
                                                                 blueprint)
            if best_conclusion_deformation is None:
                continue

            original_conclusion = next((c for c in conclusions if c["id"] == best_conclusion_deformation["id"]), None)

            if original_conclusion is None:
                continue

            change_log["break_target"] = original_conclusion
            change_log["break_result"] = best_conclusion_deformation

            # 5) redact the conclusion in the pape
            redactions = self.redact(original_conclusion,
                                     best_conclusion_deformation,
                                     results[best_conclusion_deformation["id"]],
                                     blueprint)

            if redactions is None or len(redactions) == 0:
                logging.warning(f"No redactions found for conclusion {original_conclusion['id']} in finding {fin_id}.")
                continue

            change_log["core_redactions"] = redactions

            # 6) cascade the changes to the findings in the paper
            cascaded_redactions = self.cascade(
                original_conclusion=original_conclusion,
                revised_conclusion=best_conclusion_deformation,
                associated_findings=[fin for fin in blueprint.findings_contributions if
                                     fin["id"] in original_conclusion["associated_contribution_claims"]],
                blueprint=blueprint
            )

            change_log["cascaded_redactions"] = cascaded_redactions

            # 7) merge within and across redactions and cascaded redactions
            overrides, detailed_changes, originals = self.merge_paper_changes(redactions=redactions + cascaded_redactions)

            # 8) validate and realize the redactions
            underlying_methods = list({r["step_number"]: r for rs in steps.values() for r in rs}.values())
            bb_graph = {
                "finding": fin,
                "conclusion": original_conclusion,
                "result": results[original_conclusion["id"]],
                "method": underlying_methods
            }

            modified_content = {
                "modification_type": "Adding a new virtual result and revising the conclusion accordingly",
                "modification_steps": best_conclusion_deformation["changes"],
                "modified_conclusion": best_conclusion_deformation["conclusion_summary"],
                "reasoning": best_conclusion_deformation[
                    "accounting_for_new_result"] if "accounting_for_new_result" in best_conclusion_deformation else None,
                "explanation": best_conclusion_deformation[
                    "logical_relation_revised_conclusion_to_old_conclusion"] if "logical_relation_revised_conclusion_to_old_conclusion" in best_conclusion_deformation else None
            }
            redacted_paper, edits = self.validate_and_realize_redactions(
                blueprint=blueprint,
                overrides=overrides,
                originals=originals,
                detailed_changes=detailed_changes,
                original_bb=original_conclusion,
                revised_bb=modified_content,
                bb_graph=bb_graph,
                constant_bbs=results,
                cascaded_redactions=cascaded_redactions
            )
            change_log["edits"] = edits
            change_log["considered_edits"] = overrides

            if redacted_paper is not None:
                redacted_paper.changes = change_log

                return redacted_paper

        # 9) if no deformation is accepted, return None
        return None
