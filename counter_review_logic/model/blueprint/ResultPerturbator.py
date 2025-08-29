import copy
import logging
from pathlib import Path

from ...data import Paper
from ...framework import PaperBlueprint, PaperSiteEngineer
from ...framework import PaperCounterfactual
from ...llm import ChatLLM
from ..cfgen import BlueprintBasedPerturbator


class ResultPerturbator(BlueprintBasedPerturbator):
    """
    Result perturbator for generating counterfactuals that perturbs a result of a paper while
    adapting the underlying methods if needed and leaving the conclusions and findings intact.
    """

    def __init__(self,
                 name: str,
                 llm: ChatLLM,
                 blueprint_dir: str | Path | None = None,
                 prompt_base_path: str | Path | None = None,
                 config=None):
        super().__init__(name="result_perturbator." + name,
                         llm=llm,
                         blueprint_dir=blueprint_dir,
                         prompt_base_path=prompt_base_path,
                         config=config)

        self.add_prompt_paths({
            "deform": Path("deform") / "result" / "base_prompt.txt",
            "cascade": Path("cascade") / "res_to_met.txt",
            "rank": Path("deform") / "result" / "rank_deform.txt",
            "check_media": Path("deform") / "result" / "check_media.txt",
        })

        self.add_template_paths({
            "deform": Path("deform") / "result" / "base_template.json",
            "cascade": Path("cascade") / "res_to_met.json",
            "rank": Path("deform") / "result" / "rank_template.json",
            "check_media": Path("deform") / "result" / "check_media.json",
        })

    def new_cf(self, paper: Paper) -> PaperCounterfactual:
        blueprint = self.load_blueprint(paper)

        return self.deform(blueprint)

    def rank_deformations(self,
                          result_deformations: list[dict],
                          original_results: list[dict],
                          associated_conclusions: list[dict],
                          blueprint: PaperBlueprint) -> tuple[dict, dict, dict]:
        results_with_modifications = []

        for original_conclusion, original_result, deformed_result in zip(associated_conclusions, original_results,
                                                                         result_deformations):
            if deformed_result is None:
                continue

            results_with_modifications += [{
                "modification_id": len(results_with_modifications),
                "conclusion_id": original_conclusion["id"],
                "conclusion_summary": original_conclusion["conclusion_summary"],
                "original_result_id": original_result["id"],
                "original_result_summary": original_result["result_summary"],
                "modified_result_summary": deformed_result["masked_version"],
                "changes": deformed_result["detailed_changes"],
                "realizing_negated_key_fact": deformed_result["negated_key_fact"]
            }]

        # rank
        self.load_prompt("rank")
        ranked = self.call_llm_structured_output(
            json_template=self.load_json_template("rank"),
            validate=lambda x: type(x) == list and len(x) == len(results_with_modifications) and all(
                type(y) == dict and "score" in y and "modification_id" in y for y in x),
            results_with_modifications=results_with_modifications,
            original_results=original_results,
            paper_title=blueprint.paper.get_title(),
            research_goal=blueprint.research_goal["research_goal"]
        )

        # take the top modification
        if ranked is None or len(ranked) == 0:
            return None, None, None

        best_idx = -1
        best = None
        ranked = list(sorted(ranked, key=lambda x: x["score"], reverse=True))
        while True:
            if len(ranked) == 0:
                break

            try:
                best = ranked[0]
                best_idx = int(best["modification_id"])
                break
            except:
                ranked = ranked[1:]

        if best_idx not in range(len(results_with_modifications)):
            return None, None, None

        original_result = next(
            (o for o in original_results if o["id"] == results_with_modifications[best_idx]["original_result_id"]),
            None)
        best_deformation = next((co for co in result_deformations if co["selected_result_id"] == original_result["id"]),
                                None)
        original_conclusion = next(
            (co for co in associated_conclusions if co["id"] == results_with_modifications[best_idx]["conclusion_id"]),
            None)

        return original_result, best_deformation, original_conclusion

    def list_possible_deformations(self, results: dict[str, list], conclusions: list[dict],
                                   blueprint: PaperBlueprint) -> \
            list[dict]:
        output = []

        for conclusion in conclusions:
            rs = results[conclusion["id"]]

            self.load_prompt("deform")
            deformed = self.call_llm_structured_output(
                json_template=self.load_json_template("deform"),
                validate=lambda x: type(
                    x) == dict and "selected_result_id" in x and "edited_result" in x and "masked_version" in x,
                conclusion={
                    "conclusion_summary": conclusion["conclusion_summary"],
                    "context": conclusion["context"],
                    "evidences": conclusion["evidences"]
                },
                results=rs,
                paper_title=blueprint.paper.get_title(),
                paper_abstract=blueprint.paper.get_abstract().replace("#", "")
            )

            output += [deformed]

        return output

    def find_relevant_media(self, original_result: dict, blueprint: PaperBlueprint) -> list:
        raise NotImplementedError("This method makes no sense. Dont call it.")
        """""
        title = blueprint.paper.get_title()
        abstract = blueprint.paper.get_abstract().replace("#", "")

        # result formatting
        res = {
            "result_summary": original_result["result_summary"],
            "evidences": original_result["evidences"],
            "mentions": original_result["coreferences"]
        }
        figs = [f["text"] for f in blueprint.paper.get_figures().values()]
        tbs = [table for table in blueprint.paper.get_tables().values()]

        self.load_prompt("check_media")
        media = self.call_llm_structured_output(
            json_template=self.load_json_template("check_media"),
            validate=lambda x: type(x) == dict and "relevant_figures" in x and "relevant_tables" in x and all(
                type(f) == dict and "figure_name" in f for f in x["relevant_figures"]) and all(
                type(t) == dict and "table_name" in t for t in x["relevant_tables"]),
            result=res,
            paper_title=title,
            paper_abstract=abstract,
            figures=figs,
            tables=tbs,
        )

        if media is None:
            return []

        return [f["figure_name"] for f in media["relevant_figures"]] + [t["table_name"] for t in
                                                                        media["relevant_tables"]]"""

    def deform(self, blueprint: PaperBlueprint) -> PaperCounterfactual | None:
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
                "operation": "Break the logical chain by changing a result to disalign with a conclusion underlying an important finding"
            }

            # 2) get their underlying components
            _, conclusions, results, steps = blueprint.get_building_block(fin_id)

            # 3) for each result and its findings, develop a plan to deform it
            result_deformations = self.list_possible_deformations(results, conclusions, blueprint)
            covered_results = [r["selected_result_id"] for r in result_deformations if r is not None]

            original_results = [blueprint.results[rid] for rid in covered_results if rid in blueprint.results]

            if len(original_results) == 0:
                logging.warning(f"No original results found for finding {fin_id}. Skipping deformation.")
                continue

            # 4) assess and select the most plausible plan
            original_result, best_deformation, original_conclusion = self.rank_deformations(result_deformations,
                                                                                            original_results,
                                                                                            conclusions,
                                                                                            blueprint)

            if best_deformation is None:
                continue

            # 4.2) results need extra attention to tables and figures, so we need to make sure that all relevant ones are included
            #additional_media = self.find_relevant_media(original_result, blueprint)
            #if additional_media:
            #    original_result = copy.deepcopy(original_result)
            #    for am in additional_media:
            #        if am.lower().strip() not in [f.lower().strip() for f in original_result["evidence_location"]]:
            #            original_result["evidence_location"] += [am]

            change_log["break_target"] = original_result
            change_log["break_result"] = best_deformation

            # 5) redact the result in the paper
            associated_conclusions = [con for con in blueprint.conclusions.values() if
                                      con["id"] in original_result["associated_conclusions"]]
            redactions = self.redact(original_result,
                                     {
                                         "edited": best_deformation["masked_version"],
                                         "detailed_changes": best_deformation["detailed_changes"],
                                         "underlying_reasoning": {
                                             "negated_key_fact": best_deformation["negated_key_fact"],
                                             "conclusion_verification": best_deformation["conclusion_verification"],
                                             "edited_result": best_deformation["edited_result"],
                                         },
                                     },
                                     associated_conclusions,
                                     blueprint)

            change_log["core_redactions"] = redactions

            # 6) cascade the changes downward to the methods in the paper
            # for now, skip it
            """cascaded_redactions = self.cascade(
                original_result=original_result,
                revised_result=best_deformation,
                constant_results=[r for r in original_results if r["id"] != original_result["id"]],
                associated_methods=[con for con in blueprint.conclusions if
                                    con["id"] in original_result["associated_conclusions"]],
                blueprint=blueprint
            )"""
            cascaded_redactions = []

            change_log["cascaded_redactions"] = cascaded_redactions

            if redactions is None or len(redactions) == 0:
                logging.warning(f"No redactions found for result. Skipping deformation.")
                continue

            # 7) merge within and across redactions and cascaded redactions
            overrides, detailed_changes, originals = self.merge_paper_changes(redactions=redactions + cascaded_redactions)

            # 8) validate and realize the redactions
            underlying_methods = list({r["step_number"]: r for rs in steps.values() for r in rs}.values())
            associated_conclusion = [con for con in conclusions if con["id"] in original_result["associated_conclusions"]][0]
            bb_graph = {
                "finding": fin,
                "conclusion": associated_conclusion,
                "result": original_result,
                "method": underlying_methods
            }

            modified_content = {
                "modification_type": "Adding a new virtual result and revising the conclusion accordingly",
                "modification_steps": best_deformation["detailed_changes"],
                "modified_result": best_deformation["masked_version"],
                "reasoning": best_deformation[
                    "negated_key_fact"] if "negated_key_fact" in best_deformation else None,
                "explanation": best_deformation[
                    "conclusion_verification"] if "conclusion_verification" in best_deformation else None
            }
            redacted_paper, edits = self.validate_and_realize_redactions(
                blueprint=blueprint,
                overrides=overrides,
                originals=originals,
                detailed_changes=detailed_changes,
                original_bb=original_result,
                revised_bb=modified_content,
                bb_graph=bb_graph,
                constant_bbs={"experimental_design": underlying_methods}, #todo add conclusions and findings
                cascaded_redactions=cascaded_redactions,
                with_filtering=True
            )
            change_log["edits"] = edits
            change_log["considered_edits"] = overrides

            if redacted_paper is not None:
                redacted_paper.changes = change_log

                return redacted_paper

        # 9) if no deformation is accepted, return None
        return None

    def redact(self, original_bb: dict, revised_bb: dict, derived_conclusions: list[dict], blueprint: PaperBlueprint):
        paper = blueprint.paper.without_appendix()

        # 1) get inputs for redaction
        # get all relevant passages from the paper
        locations = []
        for loc in original_bb["evidences"]:
            aligned = PaperSiteEngineer.get_paragraphs_from_evidence_location(loc["location"], paper)

            if aligned is None:
                aligned = PaperSiteEngineer.get_paragraph_from_evidence_span(loc["span"], paper)

            if aligned is None:
                continue

            for a in aligned:
                if a[0].lower().strip() == "figure":
                    fig = PaperSiteEngineer.get_media_by_id(media_type="figure", media_id=a[1], paper=paper, to_cleaned_text=False)
                    locations += [("figure", a[1], fig["text"])]
                elif a[0].lower().strip() == "table":
                    tbl = PaperSiteEngineer.get_media_by_id(media_type="table", media_id=a[1], paper=paper, to_cleaned_text=False)
                    locations += [("table", a[1], tbl)]
                else:
                    locations += [a]

        if len(locations) == 0:
            return None

        # reformat the bb to adhere expected format
        original_bb_formatted = {**original_bb}
        original_bb_formatted["summary"] = original_bb["result_summary"]
        original_bb_formatted["evidences"] = original_bb["evidences"]
        del original_bb_formatted["result_summary"]

        # produce a list of constant bbs (results)
        constant_bbs = []
        for c in derived_conclusions:
            constant_bbs += [{
                "id": c["id"],
                "type": "conclusion",
                "summary": c["conclusion_summary"],
                "location": c["evidences"],
                "relation": c["relevance_to_claim"],
                "mentions": [
                    {
                        "location": f"paragraph{co['location']}",
                        "span": co["span"]
                    } for co in c["coreferences"]
                ]
            }]

        # 2) get redactions for the paper
        # get redactions of the paper
        redactions = self.suggest_paper_changes_for_bb_revision(
            bb_type="result",
            original_bb=original_bb_formatted,
            revised_bb=revised_bb,
            locations=locations,
            constant_bb_types=["conclusions", "finding contributions"],
            blueprint=blueprint,
            constant_bbs=constant_bbs,
            ensure_constant=False  # todo for now: no validation since the LLM is bad at it
        )

        return redactions

    def redact_methods(self):
        pass

    def cascade(self, original_result: dict, revised_result: dict, constant_results: list[dict],
                associated_methods: list[dict],
                blueprint: PaperBlueprint):
        raise NotImplementedError("NOt implemented for now")

        modified_content = {
            "modification_type": "Adapting a result to disalign with the derived conclusion.",
            "modification_steps": revised_result["detailed_changes"],
            "modified_result": revised_result["masked_version"],
            "reasoning": revised_result[
                "negated_key_fact"] if "negated_key_fact" in revised_result else None,
            "explanation": revised_result[
                "conclusion_verification"] if "conclusion_verification" in revised_result else None
        }

        self.load_prompt("cascade")
        revised_methods = self.call_llm_structured_output(
            json_template=self.load_json_template("cascade"),
            validate=lambda x: type(x) == list and all(
                type(y) == dict and "method_step_number" in y and "revised_step" in y for y in x),
            original_result=original_result,
            revised_result=modified_content,
            methodological_steps=associated_methods,
            constant_results=constant_results,
            paper_title=blueprint.paper.get_title(),
            research_goal=blueprint.research_goal["research_goal"]
        )

        # produce redactions for the findings with the findings perturbator
        blueprint_revised = copy.deepcopy(blueprint)
        blueprint_revised.results[original_result["id"]] = {
            "id": original_result["id"],
            "result_summary": revised_result["masked_version"],
            "evidence_location": original_result["evidence_location"],
            "relevance_to_conclusion": original_result["relevance_to_conclusion"],
            "associated_conclusions": original_result["associated_conclusions"],
            "coreferences": [{"location": r["location"], "span": r["revised_evidence"]} for r in
                             revised_result["revisions_to_evidences"]]
        }

        # fixme below needs to be adapted to results and use a cascade to method prompt
        """output_redactions = []
        for rm in revised_methods:
            if rm["revised_claim"] is None:
                continue

            original_fin = next((m for m in associated_methods if m["step_number"] == rm["contribution_claim_id"]),
                                None)
            if original_fin is None:
                continue

            # no changes needed, skip
            if rm["revised_claim"].lower().strip() == original_fin["claim_summary"].lower().strip():
                continue

            modified_fin = {
                "modification_type": "Update finding to align with revised conclusion",
                "modification_steps": rm["modification_steps"],
                "modified_finding": rm["revised_claim"],
                "reasoning": rm[
                    "effect_of_new_conclusion_on_claim"] if "effect_of_new_conclusion_on_claim" in rm else None,
                "explanation": rm["explanation"] if "explanation" in rm else None
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

        return output_redactions"""
