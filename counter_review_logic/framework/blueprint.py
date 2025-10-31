import json
import logging
import os
import random
import re
from pathlib import Path
from typing import Callable

import Levenshtein

from ..data import Paper
from ..llm import ChatLLM, parse_llm_output_as_single_json


class PaperBlueprint:
    def __init__(self,
                 paper: Paper,
                 research_goal=None,
                 all_contributions=None,
                 findings_contributions=None,
                 conclusions=None,
                 results=None,
                 method=None,
                 meta=None):
        self._paper = paper

        self._research_goal = research_goal
        self._all_contributions = all_contributions
        self._findings_contributions = findings_contributions
        self._conclusions = conclusions
        self._results = results
        self._method = method
        self._meta = meta if meta is not None else {}

    def is_complete(self):
        return (self._research_goal is not None and
                self._all_contributions is not None and
                self._findings_contributions is not None and len(self._findings_contributions) > 0 and
                self._conclusions is not None and
                self._results is not None and
                self._method is not None)

    @property
    def paper(self):
        return self._paper

    @property
    def research_goal(self):
        return self._research_goal

    @research_goal.setter
    def research_goal(self, value):
        if isinstance(value, dict) or value is None:
            self._research_goal = value
        else:
            raise TypeError("research_goal must be a dict")

    @property
    def all_contributions(self):
        return self._all_contributions

    @all_contributions.setter
    def all_contributions(self, value):
        self._all_contributions = value

    @property
    def findings_contributions(self):
        return self._findings_contributions

    @findings_contributions.setter
    def findings_contributions(self, value):
        self._findings_contributions = value

    @property
    def conclusions(self):
        return self._conclusions

    @conclusions.setter
    def conclusions(self, value):
        self._conclusions = value

    @property
    def results(self):
        return self._results

    @results.setter
    def results(self, value):
        self._results = value

    @property
    def method(self):
        return self._method

    @method.setter
    def method(self, value):
        self._method = value

    @property
    def meta(self):
        return self._meta

    @meta.setter
    def meta(self, value):
        if isinstance(value, dict) or value is None:
            self._meta = value
        else:
            raise TypeError("meta must be a dict")

    def get_building_block(self, contribution_id: str):
        assert contribution_id in [f["id"] for f in
                                   self._findings_contributions], f"Contribution ID {contribution_id} not found in findings contributions."

        contribution = next((c for c in self._findings_contributions if c["id"] == contribution_id), None)
        conclusions = [c for c in self.conclusions.values() if contribution_id in c["associated_contribution_claims"]]
        results = {
            conclusion["id"]: [r for r in self.results.values() if conclusion["id"] in r["associated_conclusions"]] for
            conclusion in conclusions}
        steps = {r["id"]: [s for s in self.method["steps"] if r["id"] in s["results"]] for rs in results.values() for r
                 in rs}

        return contribution, conclusions, results, steps

    def building_block_chains(self, contribution_id: str, conclusion_id: str | None = None,
                              result_id: str | None = None):
        contribution, conclusions, results, steps = self.get_building_block(contribution_id)

        if conclusion_id is not None:
            conclusions = [c for c in conclusions if c["id"] == conclusion_id]

        if result_id is not None:
            results = {conclusion["id"]: [r for r in results[conclusion["id"]] if r["id"] == result_id] for conclusion
                       in conclusions}

        output = {}
        for conclusion in conclusions:
            for result in results[conclusion["id"]]:
                for step in steps[result["id"]]:
                    output += [(conclusion, result, step)]

        return contribution, output

    def to_dict(self):
        return {
            "paper": self.paper.id,
            "research_goal": self._research_goal,
            "all_contributions": self._all_contributions,
            "findings_contributions": [c["id"] for c in
                                       self._findings_contributions] if self.findings_contributions else None,
            "conclusions": self._conclusions,
            "results": self._results,
            "method": self._method,
            "meta": self._meta,
        }

    @classmethod
    def from_dict(cls, data:dict, paper: Paper):
        contributions = data.get("all_contributions", None)
        findings_contributions = data.get("findings_contributions", None)

        return cls(
            paper=paper,
            research_goal=data.get("research_goal", {}),
            all_contributions=contributions,
            findings_contributions=[next(c for c in contributions if c["id"] == fid) for fid in
                                    findings_contributions] if findings_contributions is not None else None,
            conclusions=data.get("conclusions", {}),
            results=data.get("results", {}),
            method=data.get("method", {}),
            meta=data.get("meta", {})
        )

    def save_json(self, path):
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=4)

    @classmethod
    def load_json(cls, path, paper: Paper):
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        assert data[
                   "paper"] == paper.id, "The paper ID in the JSON file does not match the provided paper ID. Please check the file."

        return cls.from_dict(data, paper)


class PaperArchitect:
    def __init__(self, llm: ChatLLM, prompt_base_path: str | Path = None, cache_dir: str | Path = None):
        self.llm = llm

        if isinstance(prompt_base_path, str):
            prompt_base_path = Path(prompt_base_path)

        self.prompt_base_path = prompt_base_path
        if prompt_base_path is None and "PROMPT_DIR" in os.environ:
            self.prompt_base_path = Path(os.environ["PROMPT_DIR"].replace("\"", "")) / "cfgen" / "blueprint"
        elif prompt_base_path is None:
            self.prompt_base_path = Path(__file__).resolve().parent / "prompts" # default path

        assert self.prompt_base_path.exists(), f"the prompt base path needs to exist. It does not: {self.prompt_base_path}."

        self.max_attempts = 5
        self.cache_dir = cache_dir

        if self.cache_dir is not None:
            self.cache_dir = Path(self.cache_dir)
            assert self.cache_dir.exists(), f"the cache dir needs to exist. It does not: {self.cache_dir}."

        self.cache = {}

    def load_prompt(self, prompt_name: str):
        if prompt_name == "revise_valid_json":
            prompt_path = self.prompt_base_path / "revise_valid_json.txt"
        elif prompt_name == "research_goal":
            prompt_path = self.prompt_base_path / "research_goal" / "base_prompt.txt"
        elif prompt_name == "contribution_claims":
            prompt_path = self.prompt_base_path / "contribution_claims" / "base_prompt.txt"
        elif prompt_name == "contribution_claims:revise":
            prompt_path = self.prompt_base_path / "contribution_claims" / "revise.txt"
        elif prompt_name == "contribution_claims:rank":
            prompt_path = self.prompt_base_path / "contribution_claims" / "rank.txt"
        elif prompt_name == "conclusions":
            prompt_path = self.prompt_base_path / "conclusions" / "base_prompt.txt"
        elif prompt_name == "conclusions_all":
            prompt_path = self.prompt_base_path / "conclusions" / "find_all.txt"
        elif prompt_name == "conclusions:feedback":
            prompt_path = self.prompt_base_path / "conclusions" / "feedback.txt"
        elif prompt_name == "conclusions:revise":
            prompt_path = self.prompt_base_path / "conclusions" / "revise.txt"
        elif prompt_name == "conclusions:merge":
            prompt_path = self.prompt_base_path / "conclusions" / "merge.txt"
        elif prompt_name == "conclusions:group":
            prompt_path = self.prompt_base_path / "conclusions" / "group.txt"
        elif prompt_name == "results":
            prompt_path = self.prompt_base_path / "results" / "base_prompt.txt"
        elif prompt_name == "results_all":
            prompt_path = self.prompt_base_path / "results" / "find_all_text.txt"
        elif prompt_name == "results_all_media":
            prompt_path = self.prompt_base_path / "results" / "find_all_media.txt"
        elif prompt_name == "results_match_media":
            prompt_path = self.prompt_base_path / "results" / "match_media.txt"
        elif prompt_name == "results:group":
            prompt_path = self.prompt_base_path / "results" / "group.txt"
        elif prompt_name == "results:feedback":
            prompt_path = self.prompt_base_path / "results" / "feedback.txt"
        elif prompt_name == "results:revise":
            prompt_path = self.prompt_base_path / "results" / "revise.txt"
        elif prompt_name == "results:merge":
            prompt_path = self.prompt_base_path / "results" / "merge.txt"
        elif prompt_name == "methods":
            prompt_path = self.prompt_base_path / "methods" / "base_prompt.txt"
        elif prompt_name == "methods:feedback":
            prompt_path = self.prompt_base_path / "methods" / "feedback.txt"
        elif prompt_name == "methods:revise":
            prompt_path = self.prompt_base_path / "methods" / "revise.txt"
        elif prompt_name == "coreferences":
            prompt_path = self.prompt_base_path / "coreferences" / "base_prompt.txt"
        else:
            raise ValueError("Invalid prompt name.")

        self.llm.load_prompt(prompt_path)

    def load_json_template(self, template_name: str):
        if template_name == "research_goal":
            json_template_path = self.prompt_base_path / "research_goal" / "template.json"
        elif template_name == "contribution_claims":
            json_template_path = self.prompt_base_path / "contribution_claims" / "template.json"
        elif template_name == "contribution_claims:rank":
            json_template_path = self.prompt_base_path / "contribution_claims" / "template_rank.json"
        elif template_name == "conclusions":
            json_template_path = self.prompt_base_path / "conclusions" / "template.json"
        elif template_name == "conclusions_all":
            json_template_path = self.prompt_base_path / "conclusions" / "template_all.json"
        elif template_name == "conclusions:merge":
            json_template_path = self.prompt_base_path / "conclusions" / "template_merge.json"
        elif template_name == "conclusions:group":
            json_template_path = self.prompt_base_path / "conclusions" / "template_group.json"
        elif template_name == "results":
            json_template_path = self.prompt_base_path / "results" / "template.json"
        elif template_name == "results_all":
            json_template_path = self.prompt_base_path / "results" / "template_all.json"
        elif template_name == "results_match_media":
            json_template_path = self.prompt_base_path / "results" / "template_match_media.json"
        elif template_name == "results:group":
            json_template_path = self.prompt_base_path / "results" / "template_group.json"
        elif template_name == "results:merge":
            json_template_path = self.prompt_base_path / "results" / "template_merge.json"
        elif template_name == "methods":
            json_template_path = self.prompt_base_path / "methods" / "template.json"
        elif template_name == "coreferences":
            json_template_path = self.prompt_base_path / "coreferences" / "template.json"
        else:
            raise ValueError("Invalid template name.")

        with open(json_template_path, 'r', encoding='utf-8') as f:
            json_template = f.read()

        return json_template

    def call_llm_unstructured_output(self, **params):
        return self.llm(params)

    def call_llm_structured_output(self, json_template: str, validate: Callable, **params):
        response = self.llm(params)
        parsed = parse_llm_output_as_single_json(response)[1]

        if parsed is not None and not validate(parsed):
            parsed = None

        response2 = response
        for i in range(self.max_attempts):
            if parsed is not None:
                return parsed

            logging.info(f"Retrying after bad format {i + 1} of {self.max_attempts} times")

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

    def _save_to_cache(self, blueprint: PaperBlueprint):
        self.cache[blueprint.paper.id] = blueprint

        if self.cache_dir:
            blueprint.save_json(self.cache_dir / f"{blueprint.paper.id}.json")

    def _load_from_cache(self, paper: Paper):
        if paper.id in self.cache:
            return self.cache[paper.id]

        # if no disc caching, return None
        if self.cache_dir is None:
            return None

        # check by loading, if existent return
        paper_path = self.cache_dir / f"{paper.id}.json"
        if paper_path.exists():
            return PaperBlueprint.load_json(paper_path, paper)

        return None

    def get_paper_text(self, paper: Paper):
        p = paper.without_appendix()

        title = p.get_title()
        abstract = p.get_abstract().replace("###### Abstract\n", "") if p.get_abstract() is not None else None

        if abstract is None:
            return None, None, None

        full_paper_paragraphs = p.get_text_with_paragraph_index(with_media=False)
        full_paper_paragraphs_with_media = p.get_text_with_paragraph_index(with_media=True)

        introduction = None
        for s in p.get_section_names():
            if "introduction" in s.lower().strip():
                introduction = p.get_paragraphs(s, with_line_numbers=False, numbered=True)
                figures = p.get_figures(s)
                tables = p.get_tables(s)
                algorithms = p.get_algorithms(s)

                introduction = [f"\n##{s}\n"] + introduction + ["\n", "\n".join(
                    [fig["text"] for fig in figures.values() if fig is not None]) + "\n" + "\n".join(
                    [tbl for tbl in tables.values()]) + "\n" + "\n".join(
                    [algo for algo in algorithms.values()])]

                break

        if introduction is None:
            introduction = full_paper_paragraphs[
                           :len(full_paper_paragraphs) // 2]  # just take the first half of the article

        # only intro
        introduction = "\n".join(introduction)
        paper_intro = f"## {title}\n\n###### Abstract\nparagraph 0: {abstract}\n\n{introduction}\n\n[rest of the paper omitted for brevity]"

        return paper_intro, full_paper_paragraphs, full_paper_paragraphs_with_media

    def identify_research_goal(self, paper_front_matter: str):
        self.load_prompt("research_goal")
        research_goal = self.call_llm_structured_output(
            json_template=self.load_json_template("research_goal"),
            validate=lambda x: type(x) == dict and "research_goal" in x and "paper_genre" in x,
            paper=paper_front_matter
        )

        return research_goal

    def extract_contribution_claims(self, paper_front_matter: str):
        self.load_prompt("contribution_claims")
        contribution_claims = self.call_llm_structured_output(
            json_template=self.load_json_template("contribution_claims"),
            validate=lambda x: type(x) == list and all(
                (type(y) == dict and "claim_summary" in y and "type" in y and "evidence" in y and "location" in y) for y
                in x),
            paper=paper_front_matter
        )

        # revise to assure quality
        self.load_prompt("contribution_claims:revise")
        contribution_claims = self.call_llm_structured_output(
            json_template=self.load_json_template("contribution_claims"),
            validate=lambda x: type(x) == list and all(
                (type(y) == dict and "claim_summary" in y and "type" in y and "evidence" in y and "location" in y) for y
                in x),
            paper=paper_front_matter,
            contribution_claims=contribution_claims
        )

        return contribution_claims

    def rank_contribution_claims(self, paper_front_matter: str, contribution_claims: list[dict], research_goal: dict):
        self.load_prompt("contribution_claims:rank")
        ranked_contribution_claims = self.call_llm_structured_output(
            json_template=self.load_json_template("contribution_claims:rank"),
            validate=lambda x: type(x) == list and all((type(
                y) == dict and "claim_summary" in y and "type" in y and "evidence" in y and "location" in y and "score" in y)
                                                       for y in x),
            paper=paper_front_matter,
            contribution_claims=contribution_claims,
            research_goal=research_goal
        )

        return ranked_contribution_claims

    def extract_conclusions(self, all_conclusions: list[dict], paper_paragraphs: str, paper_title:str, paper_abstract:str, contribution_claim: dict):
        # group conclusions
        self.load_prompt("conclusions:group")
        grouped_conclusions = self.call_llm_structured_output(
            json_template=self.load_json_template("conclusions:group"),
            validate=lambda y: type(y) == dict and "choice" in y and all(
                (type(x) == dict and "conclusion_summary" in x and "id" in x and "relevance_to_claim" in x) for x in y["choice"]),
            paper_title=paper_title,
            paper_abstract=paper_abstract,
            contribution_claim=contribution_claim,
            conclusions=all_conclusions
        )

        if grouped_conclusions is None:
            return None

        # selected the grouped conclusions and merge the fields
        conclusions = []
        for conclusion in grouped_conclusions["choice"]:
            con = next((c for c in all_conclusions if c["id"] == conclusion["id"]), None)
            if con:
                conclusions.append({
                    **con,
                    "associated_contribution_claims": [contribution_claim["id"]],
                    "relevance_to_claim": conclusion["relevance_to_claim"]
                })

        return conclusions

    def merge_conclusions(self, paper_paragraphs: str, conclusions: dict, with_llm_merging=False):
        # first merge all conclusions by ids
        all_conclusions = {}
        for cc_id, conclusion_list in conclusions.items():
            for conclusion in conclusion_list:
                if conclusion["id"] not in all_conclusions:
                    all_conclusions[conclusion["id"]] = conclusion
                else:
                    # merge conclusions with same id
                    all_conclusions[conclusion["id"]]["associated_contribution_claims"] += conclusion["associated_contribution_claims"]
                    all_conclusions[conclusion["id"]]["relevance_to_claim"] += " " + conclusion["relevance_to_claim"]

        # rename conclusions to have simple number range even if new conclusions where added
        renamed_conclusions = {}
        j = 1
        for cid, conc in all_conclusions.items():
            conc["id"] = f"conc{j}"
            renamed_conclusions[conc["id"]] = conc
            j += 1

        all_conclusions = renamed_conclusions

        # merge
        if with_llm_merging:
            self.load_prompt("conclusions:merge")
            all_conclusions = self.call_llm_structured_output(
                json_template=self.load_json_template("conclusions:merge"),
                validate=lambda y: type(y) == list and all(
                    (type(x) == dict and "conclusion_summary" in x and "evidences" in x and "context" in x) for
                    x in y),
                paper=paper_paragraphs,
                conclusions_with_ids=all_conclusions
            )
        else:
            all_conclusions = list(all_conclusions.values())

        # rename the conclusions
        result = {}
        for con in all_conclusions:
            con["id"] = f"conc{len(result)}"

            if "original_conclusions" in con:
                del con["original_conclusions"]  # remove original conclusions field

            result[con["id"]] = con

        return result

    def extract_results(self, all_results, paper_with_media: str, paper_title: str, paper_abstract: str, conclusion: dict, with_feedback=True):
        self.load_prompt("results:group")
        grouped_results = self.call_llm_structured_output(
            json_template=self.load_json_template("results:group"),
            validate=lambda y: type(y) == dict and "choice" in y and all(
                (type(x) == dict and "result_summary" in x and "id" in x and "relevance_to_conclusion" in x) for x in y["choice"]),
            paper_title=paper_title,
            paper_abstract=paper_abstract,
            conclusion=conclusion,
            results=all_results
        )

        # selected the grouped conclusions and merge the fields
        results = []
        for result in grouped_results["choice"]:
            res = next((c for c in all_results if c["id"] == result["id"]), None)
            if res:
                results.append({
                    **res,
                    "associated_conclusions": [conclusion["id"]],
                    "relevance_to_conclusion": result["relevance_to_conclusion"]
                })

        return results

    def merge_results(self, paper_paragraphs: str, results: dict, conclusions: dict, llm_based_merging=False):
        all_results = {}
        for con_id, results_list in results.items():
            for result in results_list:
                if result["id"] not in all_results:
                    all_results[result["id"]] = result
                else:
                    # merge conclusions with same id
                    all_results[result["id"]]["associated_conclusions"] += result[
                        "associated_conclusions"]
                    all_results[result["id"]]["relevance_to_conclusion"] += " " + result[
                        "relevance_to_conclusion"]

        renamed_results = {}
        j = 1
        for resid, res in all_results.items():
            res["id"] = f"res{j}"
            renamed_results[res["id"]] = res
            j += 1

        all_results = renamed_results

        # prep conclusions for input
        conclusion_summaries_with_ids = [
            {
                "id": conclusion["id"],
                "summary": conclusion["conclusion_summary"]
            } for conclusion in conclusions.values()
        ]

        if llm_based_merging:
            self.load_prompt("results:merge")
            merged_results = self.call_llm_structured_output(
                json_template=self.load_json_template("results:merge"),
                validate=lambda y: type(y) == list and all((type(
                    x) == dict and "result_summary" in x and "evidences" in x and "associated_conclusions" in x and "original_results" in x)
                                                           for x in y),
                paper=paper_paragraphs,
                results_with_conclusion_ids=list(all_results.values()),
                conclusion_summaries_with_ids=conclusion_summaries_with_ids
            )
        else:
            merged_results = list(all_results.values())

        # remove original conclusions field
        for result in merged_results:
            if "original_results" in result:
                del result["original_results"]

        # add new ids
        output = {}
        for i, result in enumerate(merged_results):
            result["id"] = f"res{i}"
            output[result["id"]] = result

        return output

    def extract_methods(self, paper_paragraphs: str, results: list[dict], with_feedback=True):
        self.load_prompt("methods")
        methods = self.call_llm_structured_output(
            json_template=self.load_json_template("methods"),
            validate=lambda y: type(y) == dict and "steps" in y and type(y["steps"]) == list and \
                               all((type(
                                   x) == dict and "step_number" in x and "description" in x and "evidences" in x and "results" in x)
                                   for x in y["steps"]),

            paper=paper_paragraphs,
            results=results
        )

        # feedback
        if with_feedback:
            self.load_prompt("methods:feedback")
            feedback = self.call_llm_unstructured_output(
                paper=paper_paragraphs,
                results=results,
                output=methods
            )

            self.load_prompt("methods:revise")
            methods = self.call_llm_structured_output(
                json_template=self.load_json_template("methods"),
                validate=lambda y: type(y) == dict and "steps" in y and type(y["steps"]) == list and \
                                   all((type(
                                       x) == dict and "step_number" in x and "description" in x and "evidences" in x and "results" in x)
                                       for x in y["steps"]),
                paper=paper_paragraphs,
                results=results,
                output=methods,
                feedback=feedback,
            )

        return methods

    def extract_coreferences(self, paper_text: str, contributions: list, conclusions: dict, results: dict,
                             method: dict):
        # Prepare the extracted elements
        extracted_contributions = []
        for contribution in contributions:
            extracted_contributions += [{
                "id": contribution["id"],
                "summary": contribution["claim_summary"],
                "location": contribution["location"],
                "span": contribution["evidence"]
            }]

        extracted_conclusions = []
        for conclusion in conclusions.values():
            extracted_conclusions += [{
                "id": conclusion["id"],
                "summary": conclusion["conclusion_summary"],
                "location": conclusion["evidences"],
                "span": None
            }]

        extracted_results = []
        for result in results.values():
            extracted_results += [{
                "id": result["id"],
                "summary": result["result_summary"],
                "location": result["evidences"],
                "span": None
            }]

        extracted_methods = []
        for step in method["steps"]:
            extracted_methods += [{
                "id": "step" + str(step["step_number"]),
                "summary": step["description"],
                "location": step["evidences"],
                "span": None
            }]

        def extract_with_llm(extracted_concepts):
            self.load_prompt("coreferences")
            cor = self.call_llm_structured_output(
                json_template=self.load_json_template("coreferences"),
                validate=lambda y: type(y) == list and all(
                    type(x) == dict and "extracted_element_id" in x and "coreferences" in x and type(
                        x["coreferences"] == list) for x in y),
                paper=paper_text,
                extracted_elements=extracted_concepts
            )

            if cor is None:
                return {}

            output = {}
            for e in cor:
                if "coreferences" in e:
                    output[e["extracted_element_id"]] = e["coreferences"]
                else:
                    logging.warning(f"Faulty coreferences found in output: {e}")

            return output

        augmented_contributions = extract_with_llm(extracted_contributions)
        augmented_conclusions = extract_with_llm(extracted_conclusions)
        augmented_results = extract_with_llm(extracted_results)
        augmented_methods = extract_with_llm(extracted_methods)

        return augmented_contributions, augmented_conclusions, augmented_results, augmented_methods

    def construct_blueprint(self, paper: Paper, with_feedback=True, with_merging=False):
        paper_front_matter, full_paper_paragraphs, full_paper_paragraphs_with_media = self.get_paper_text(paper)
        if paper_front_matter is None:
            logging.error(f"Could not extract text from paper; can't create blueprint {paper.id}.")
            return None

        title = paper.get_title()
        abstract = paper.get_abstract().replace("###### Abstract\n", "")

        # check disc cache for data and load
        blueprint = self._load_from_cache(paper)
        if blueprint is None:
            blueprint = PaperBlueprint(paper, meta={
                "architect": self.llm.model,
                "feedback": with_feedback,
                "merge": with_merging
            })

        # 1) determine research goal
        logging.info(f"Extracting research goal for paper {paper.id}...")
        if blueprint.research_goal is None:
            research_goal = self.identify_research_goal(paper_front_matter)

            if research_goal is None:
                logging.warning(f"Could not identify research goal for paper {paper.id}.")
                research_goal = "Unknown research goal. Please check the paper manually."

            blueprint.research_goal = research_goal
            self._save_to_cache(blueprint)

        # 2) extract contribution claims
        logging.info(f"Extracting contribution claims for paper {paper.id}...")
        if blueprint.all_contributions is None:
            contribution_claims = self.extract_contribution_claims(paper_front_matter)

            if contribution_claims is None:
                logging.warning(f"Could not extract contribution claims for paper {paper.id}.")
                contribution_claims = []

            # rank contribution claims
            random.shuffle(contribution_claims)  # to avoid bias, randomize order
            contribution_claims = self.rank_contribution_claims(paper_front_matter, contribution_claims,
                                                                blueprint.research_goal)

            # assign IDs
            for i, cc in enumerate(contribution_claims):
                cc["id"] = f"cc{i}"

            # save
            blueprint.all_contributions = contribution_claims
            blueprint.findings_contributions = [c for c in contribution_claims if
                                                c["type"] == "finding" and c["speculative"] == "No"]
            self._save_to_cache(blueprint)

        # stop pre-emptively, because the paper cannot be processed if it has no findings contributions
        if not blueprint.findings_contributions:
            logging.warning(f"Paper {paper.id} has no findings contributions. Cannot extract further information.")
            return None

        logging.info(f"{len(blueprint.findings_contributions)} findings contributions for paper {paper.id}.")

        # 3) extract conclusions
        logging.info(f"Extracting conclusions for paper {paper.id}...")
        if blueprint.conclusions is None:
            # extract all conclusions up-front
            self.load_prompt("conclusions_all")
            all_conclusions = self.call_llm_structured_output(
                json_template=self.load_json_template("conclusions_all"),
                validate=lambda y: type(y) == list and all(
                    (type(x) == dict and "conclusion_summary" in x and "evidences" in x and "context" in x) for x in y),
                paper=full_paper_paragraphs
            )

            # assign IDs
            for i, con in enumerate(all_conclusions):
                con["id"] = f"conc{i}"

            # feedback on conclusions
            if with_feedback:
                self.load_prompt("conclusions:feedback")
                feedback = self.call_llm_unstructured_output(
                    paper=full_paper_paragraphs,
                    contribution_claims=blueprint.findings_contributions,
                    output=all_conclusions
                )

                self.load_prompt("conclusions:revise")
                all_conclusions = self.call_llm_structured_output(
                    json_template=self.load_json_template("conclusions"),
                    validate=lambda y: type(y) == list and all(
                        (type(
                            x) == dict and "conclusion_summary" in x and "evidences" in x and "context" in x and "associated_contribution_claims" in x and "id" in x)
                        for x in y),
                    paper=full_paper_paragraphs,
                    contribution_claims=blueprint.findings_contributions,
                    output=all_conclusions,
                    feedback=feedback,
                )

                if all_conclusions is None:
                    return None

                for i, con in enumerate(all_conclusions):
                    con["id"] = f"conc{i}"

            # check conclusions for each contribution
            conclusions = {}
            for contribution in blueprint.findings_contributions:
                conclusions_for_cc = self.extract_conclusions(all_conclusions, full_paper_paragraphs, title, abstract, contribution)

                if conclusions_for_cc is None:
                    return None # failure

                if len(conclusions_for_cc) == 0:
                    continue

                conclusions[contribution["id"]] = conclusions_for_cc

            # merge and deduplicate
            conclusions = self.merge_conclusions(full_paper_paragraphs, conclusions, False)

            # save
            blueprint.conclusions = conclusions
            self._save_to_cache(blueprint)

        logging.info(
            f"{len(blueprint.conclusions)} conclusions for paper {paper.id}.")

        # 4) extract results
        logging.info(f"Extracting results for paper {paper.id}...")
        if blueprint.results is None:
            # extract all conclusions up-front
            self.load_prompt("results_all")
            all_results = self.call_llm_structured_output(
                json_template=self.load_json_template("results_all"),
                validate=lambda y: type(y) == list and all(
                    (type(x) == dict and "result_summary" in x and "evidences" in x) for x in y),
                paper=full_paper_paragraphs_with_media
            )

            # assign IDs
            for i, res in enumerate(all_results):
                res["id"] = f"res{i}"

            # check for figures and tables
            self.load_prompt("results_all_media")
            media_results = self.call_llm_structured_output(
                json_template=self.load_json_template("results_all"),
                validate=lambda y: type(y) == list and all(
                    (type(x) == dict and "result_summary" in x and "evidences" in x) for x in y),
                paper=full_paper_paragraphs_with_media
            )

            # assign IDs
            for i, res in enumerate(media_results):
                res["id"] = f"mres{i}"

            # match figures to textual results
            self.load_prompt("results_match_media")
            matched = self.call_llm_structured_output(
                json_template=self.load_json_template("results_match_media"),
                validate=lambda y: type(y) == dict and "matched" in y and "unmatched" in y and all(type(x) == dict and "id" in x and "matching_results" in x for x in y["matched"]),
                paper=full_paper_paragraphs_with_media,
                results=all_results,
                figures_and_tables=media_results
            )

            if matched is not None:
                # merge media results with textual results
                for m in matched["matched"]:
                    associated_results = m["matching_results"]
                    media = next((k for k in media_results if k["id"] == m["id"]), None)

                    if media is None:
                        continue

                    for rid in associated_results:
                        res = next((r for r in all_results if r["id"] == rid), None)

                        if res:
                            for ev in media["evidences"]:
                                res["evidences"].append({**ev, "media_summary": media["result_summary"]})

                for m in matched["unmatched"]:
                    media = next((k for k in media_results if k["id"] == m), None)
                    if media is not None:
                        all_results.append(media)

                for i, res in enumerate(all_results):
                    res["id"] = f"res{i}"

            results = {}
            for conclusion_id, conclusion in blueprint.conclusions.items():
                results_of_con = self.extract_results(all_results, full_paper_paragraphs_with_media, title, abstract, conclusion)

                results[conclusion["id"]] = results_of_con

            results = self.merge_results(full_paper_paragraphs, results, blueprint.conclusions, False)

            blueprint.results = results
            self._save_to_cache(blueprint)

        logging.info(f"{len(blueprint.results)} results for paper {paper.id}.")

        # 5) extract methods
        logging.info(f"Extracting methods for paper {paper.id}...")
        if not blueprint.method:
            method = self.extract_methods(full_paper_paragraphs, list(blueprint.results.values()), with_feedback)

            blueprint.method = method
            self._save_to_cache(blueprint)

        logging.info(f"Extracted {len(blueprint.method['steps'])} methods for paper {paper.id}.")

        # 6) extract coreferences for each element
        logging.info(f"Extracting coreferences for paper {paper.id}...")

        # check if blueprint already has coreferences
        has_coreferences = False
        for c in blueprint.findings_contributions:
            if "coreferences" in c:
                has_coreferences = True
                break

        if not has_coreferences:
            coreferences_contributions, coreferences_conclusions, coreferences_results, coreferences_methods = self.extract_coreferences(
                full_paper_paragraphs_with_media,
                blueprint.findings_contributions,
                blueprint.conclusions,
                blueprint.results,
                blueprint.method
            )

            for c in blueprint.findings_contributions:
                if c["id"] in coreferences_contributions:
                    c["coreferences"] = coreferences_contributions[c["id"]]
                else:
                    c["coreferences"] = []

            for conclusion in blueprint.conclusions.values():
                if conclusion["id"] in coreferences_conclusions:
                    conclusion["coreferences"] = coreferences_conclusions[conclusion["id"]]
                else:
                    conclusion["coreferences"] = []

            for result in blueprint.results.values():
                if result["id"] in coreferences_results:
                    result["coreferences"] = coreferences_results[result["id"]]
                else:
                    result["coreferences"] = []

            for step in blueprint.method["steps"]:
                stepid = "step" + str(step["step_number"])

                if stepid in coreferences_methods:
                    step["coreferences"] = coreferences_methods[stepid]
                else:
                    step["coreferences"] = []

            self._save_to_cache(blueprint)

        return blueprint


class PaperSiteEngineer:
    def __init__(self):
        pass

    @staticmethod
    def clean_markdown(text):
        # remove links
        while True:
            match = re.search(r"\[(.+?)\]\((#.+?)\)", text)
            if match is None:
                break

            text = text[0:match.start()] + match.group(1) + text[match.end():]

        # remove any styling and line breaks
        text = re.sub("\*\*", "", text)
        text = re.sub("__", "", text)
        text = re.sub("#", "", text)
        text = re.sub("\n", "  ", text)
        text = re.sub("\$", "", text)

        return text

    @staticmethod
    def clean_table(table_text: str):
        if table_text is None:
            return None

        # search for table start tag
        start = table_text.find("<table")
        end = table_text.find("</table>")

        if start != -1 and end != -1:
            # remove the table start tag
            table_text = table_text[0:start] + table_text[end + len("</table>"):]
            # remove the new lines
            table_text = re.sub(r"\n+", " ", table_text)

        return table_text

    @staticmethod
    def get_media_by_id(media_type, media_id, paper, to_cleaned_text=False):
        try:
            media_number = int(media_id)
        except:
            media_number = None

        if media_number is None:  # get by name/id
            if media_type == "figure":
                fig = paper.get_figure(media_id)
                if to_cleaned_text:
                    return fig["text"] if fig is not None else None
                else:
                    return fig
            elif media_type == "table":
                tbl = paper.get_table(media_id)
                if to_cleaned_text:
                    return PaperSiteEngineer.clean_table(tbl) if tbl is not None else None
                else:
                    return tbl
            else:
                raise ValueError(f"Unknown media type: {media_type}.")
        else:  # get by number
            if media_type == "figure":
                for fig_name in paper.get_figures().keys():
                    match = re.search(rf"\.F{media_number}\.", fig_name)
                    if match:
                        fig = paper.get_figure(fig_name)
                        if to_cleaned_text:
                            return fig["text"] if fig is not None else None
                        else:
                            return fig
            elif media_type == "table":
                for tbl_name in paper.get_tables().keys():
                    match = re.search(rf"\.T{media_number}", tbl_name)
                    if match:
                        tbl = paper.get_table(tbl_name)
                        if to_cleaned_text:
                            return PaperSiteEngineer.clean_table(tbl) if tbl is not None else None
                        else:
                            return tbl

        return None

    @staticmethod
    def get_paragraphs_from_evidence_location(evidence: str|int, paper: Paper):
        # assume its referring to a paragraph
        if type(evidence) == int:
            p  = paper.get_paragraph_by_number(evidence)
            if p is None:
                return None

            return [("paragraph", evidence, p[1])]

        evidence_cleaned = evidence.strip().lower()

        # check for a paragraph number
        matched = re.search(r"paragraphs?\s*(\d+)", evidence_cleaned)
        if matched:
            para_num = int(matched.group(1))
            para = paper.get_paragraph_by_number(para_num)
            if para is not None:
                return [("paragraph", para_num, para[1])]

        # check for paragraph with no.
        matched = re.search(r"paragraphs?\sno\.?\s*(\d+)", evidence_cleaned)
        if matched:
            para_num = int(matched.group(1))
            para = paper.get_paragraph_by_number(para_num)
            if para is not None:
                return [("paragraph", para_num, para[1])]

        # check for paragraph range
        matched = re.search(r"paragraphs?\s*(\d+)-(\d+)", evidence_cleaned)
        if not matched:
            matched = re.search(r"paragraphs?\s*(\d+)\s*to\s*(\d+)", evidence_cleaned)
        if not matched:
            matched = re.search(r"paragraphs?\s*(\d+)\s*and\s*(\d+)", evidence_cleaned)
        if not matched:
            matched = re.search(r"paragraphs?\s*(\d+)\s*[,;]\s*(\d+)", evidence_cleaned)

        if matched:
            start, end = int(matched.group(1)), int(matched.group(2))

            paras = []
            for i in range(start, end + 1):
                para = paper.get_paragraph_by_number(i)
                if para is not None:
                    paras += [(i, para[1])]

            if len(paras) > 0:
                return [("paragraph", pn, p) for pn, p in paras]

        # check for abstract
        if "Abstract" in evidence:
            abstract_paras = paper.get_abstract()
            return [("paragraph", -1, abstract_paras)]

        # check for figure / table
        matched = re.search(r"(figure|table) (.+)", evidence_cleaned)
        if matched:
            media_number = matched.group(2)
            media_type = matched.group(1)
            media = PaperSiteEngineer.get_media_by_id(media_type, media_number, paper, to_cleaned_text=True)

            if media is None:
                return None

            return [(media_type.capitalize(), media_number, media)]

        ## worst case check for plain number and assume it's a paragraph
        # check for plain number
        matched = re.search(r"(\d+)", evidence_cleaned)
        if matched:
            para_num = int(matched.group(1))
            para = paper.get_paragraph_by_number(para_num)
            if para is not None:
                return [("paragraph", para_num, para[1])]

        ## only if none of the above fired, we move to longer sections

        # check for a named section
        candidates = []
        for section in paper.get_section_names():
            sn = section.lower().strip()
            if sn in evidence_cleaned:
                candidates += [section]

        if len(candidates) > 0:
            best = list(sorted(candidates, key=len, reverse=True))[0]
            return [("section", best, paper.get_section(best))]

        # check for numbered section
        matched = re.search(r"section *(\d+)", evidence_cleaned)
        if matched:
            section_num = matched.group(1)
            candidates = [section for section in paper.get_sections().keys() if section.startswith(section_num)]

            if len(candidates) > 0:
                return [("section", candidates[0], paper.get_section(candidates[0]))]

        return None

    @staticmethod
    def fuzzy_substring_match(substring: str, large_string: str, max_edit_ratio: float = 0.3) -> bool:
        sub = substring.split(" ")
        main = large_string.split(" ")

        max_edits = int(len(sub) * max_edit_ratio)

        sub_len = len(sub)
        for i in range(len(main) - sub_len + 1):
            chunk = main[i:i + sub_len]
            if Levenshtein.distance(sub, chunk) <= max_edits:
                return True

        return False

    @staticmethod
    def get_paragraph_from_evidence_span(evidence: str, paper: Paper):
        # excluding abstract at the moment
        for paragraph in paper.get_paragraphs(numbered=True):
            paragraph_number = int(re.match("^paragraph (\d+):", paragraph).group(1))
            cleaned_paragraph = re.sub(r"^paragraph \d+:", "", paragraph)

            e = evidence[:-1].lower().strip()
            c = PaperSiteEngineer.clean_markdown(cleaned_paragraph.lower().strip())
            if e in c:
                return [("paragraph", paragraph_number, cleaned_paragraph)]

        # try again, but with fuzzy matching
        for paragraph in paper.get_paragraphs(numbered=True):
            paragraph_number = int(re.match("^paragraph (\d+):", paragraph).group(1))
            cleaned_paragraph = re.sub(r"^paragraph \d+:", "", paragraph)

            e = evidence[:-1].lower().strip()
            c = PaperSiteEngineer.clean_markdown(paragraph.lower().strip())
            if PaperSiteEngineer.fuzzy_substring_match(e, c, max_edit_ratio=0.3):
                return [("paragraph", paragraph_number, cleaned_paragraph)]

        return None

    def __call__(self, blueprint: PaperBlueprint): #fixme: adapt to new blueprint with new evide
        paper = blueprint.paper
        output = {
            "findings_contributions": {},
            "conclusions": {},
            "results": {},
            "method": {}
        }
        errors = []

        def add_coreference(co):
            loc = co["location"]

            # cover only paragraphs for now
            if type(loc) == int or loc.isdigit():
                p = paper.get_paragraph_by_number(int(loc))[1]
                return ("paragraph", int(loc), p, co["span"])

            return None

        # 1) finding contributions
        if blueprint.findings_contributions is None or len(blueprint.findings_contributions) == 0:
            logging.warning(f"No findings contributions found in blueprint for paper {paper.id}.")
            return None, errors

        for fc in blueprint.findings_contributions:
            location = fc["location"]
            para = self.get_paragraph_from_evidence_span(location, paper)

            if para is None:
                # retry by location
                para = self.get_paragraphs_from_evidence_location(location, paper)

            if para is None:
                errors += [("findings_contributions", fc["id"], "failed to find span evidence")]
                continue

            if "coreferences" in fc:
                coreferences = [add_coreference(loc) for loc in fc["coreferences"]]
                coreferences = [c for c in coreferences if c is not None]
            else:
                coreferences = []

            output["findings_contributions"][fc["id"]] = {"primary": [para], "coreferences": coreferences}

        # 2) conclusions
        for conclusion_id, conclusion in blueprint.conclusions.items():
            locations = conclusion["evidences"]

            evs = []
            for loc in locations:
                para = self.get_paragraphs_from_evidence_location(loc["location"], paper)

                if para is None:
                    para = self.get_paragraph_from_evidence_span(loc["span"], paper)

                if para is not None:
                    evs += para

            if len(evs) == 0:
                errors += [("conclusions", conclusion["id"], "failed to find location evidence: " + str(locations))]
                continue

            if "coreferences" in conclusion:
                coreferences = [add_coreference(loc) for loc in conclusion["coreferences"]]
                coreferences = [c for c in coreferences if c is not None]
            else:
                coreferences = []

            output["conclusions"][conclusion["id"]] = {"primary": evs, "coreferences": coreferences}

        # 4) results
        for result_id, result in blueprint.results.items():
            locations = result["evidences"]

            evs = []
            for loc in locations:
                para = self.get_paragraphs_from_evidence_location(loc["location"], paper)

                if para is None:
                    para = self.get_paragraph_from_evidence_span(loc["span"], paper)

                if para is not None:
                    evs += para

            if len(evs) == 0:
                errors += [("results", result["id"], "failed to find location evidence: " + str(locations))]
                continue

            if "coreferences" in result:
                coreferences = [add_coreference(loc) for loc in result["coreferences"]]
                coreferences = [c for c in coreferences if c is not None]
            else:
                coreferences = []

            output["results"][result["id"]] = {"primary": evs, "coreferences": coreferences}

        # 5) methods
        output["method"] = {}
        for step in blueprint.method["steps"]:
            locations = step["evidences"]

            evs = []
            for loc in locations:
                para = self.get_paragraphs_from_evidence_location(loc["location"], paper)

                if para is None and "span" in loc:
                    para = self.get_paragraph_from_evidence_span(loc["span"], paper)

                if para is not None:
                    evs += para

            if len(evs) == 0:
                errors += [("methods", "method" + ":" + str(step["step_number"]),
                            "failed to find location evidence: " + str(locations))]
                continue

            if "coreferences" in step:
                coreferences = [add_coreference(loc) for loc in step["coreferences"]]
                coreferences = [c for c in coreferences if c is not None]
            else:
                coreferences = []

            output["method"][step["step_number"]] = {"primary": evs, "coreferences": coreferences}

        return output, errors
