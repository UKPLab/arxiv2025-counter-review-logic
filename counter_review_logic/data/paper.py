import copy
import io
import json
import random
import re
from pathlib import Path


class Paper:
    """
    A class representing a paper with its metadata, markdown content, and structured representation.

    The paper is expected to have a markdown format with sections, figures, tables, algorithms, and paragraphs.
    The metadata should contain a 'media' key pointing to the path of the media files.
    The structured representation is created from the markdown content and includes sections, subsections,
    figures, tables, algorithms, and paragraphs with their respective line indices.
    """
    def __init__(self,
                 id:str,
                 meta:dict,
                 md: str,
                 structured_md: dict[str, tuple[int, int]] = None):
        self._meta = meta
        self._id = id
        self._md = md

        assert "media" in meta, "meta must contain a 'media' key pointing to the path of the media files"

        self._media = {}
        self._md_lines = md.split("\n")
        self._structured_md = {}

        if structured_md:
            self._structured_md = structured_md
        else:
            self._to_structured_md()

    def _indexes_to_text(self, start_idx, end_idx, with_line_numbers=False):
        """
        Converts a range of lines in the markdown content to a single string.

        :param start_idx: The starting index of the range.
        :param end_idx: The ending index of the range.
        :param with_line_numbers: If True, each line will be prefixed with its line number.
        """
        if with_line_numbers:
            return "\n".join([f"line {i}: {l}" for i, l in
                              zip(range(start_idx, end_idx + 1), self._md_lines[start_idx:end_idx + 1])])
        else:
            return "\n".join(self._md_lines[start_idx:end_idx + 1])

    def _load_media(self, fig_name):
        """
        Loads the media file associated with a figure name from the paper's media directory.

        :param fig_name: The name of the figure to load.
        """
        if fig_name not in self._media:
            # load and cache
            fig_txt = self.get("figures", fig_name)
            path = re.findall(r"!\[Figure .+\]\((.+)\)", fig_txt)
            if len(path) == 0:
                return None
            else:
                path = Path(path[0])

                with open(f"{self._meta['media']}/{path.name}", "rb") as f:
                    self._media[fig_name] = io.BytesIO(f.read())

        return self._media[fig_name]

    def without_appendix(self):
        """
        Creates a new Paper instance without the appendix sections.
        """
        new_paper = copy.deepcopy(self)

        appendix_sections = [(k, v) for k, v in new_paper._structured_md["sections"].items() if "appendix" in k.lower()]

        # delete lines
        offset = 0
        for sec, sec_lines in sorted(appendix_sections, key=lambda x: x[1][0]):
            new_paper._md_lines = new_paper._md_lines[:sec_lines[0] - offset] + new_paper._md_lines[
                                                                                sec_lines[-1] + 1 - offset:]
            offset += sec_lines[-1] - sec_lines[0] + 1

        new_paper._md = "\n".join(new_paper._md_lines)
        new_paper._to_structured_md()

        return new_paper

    def override_multi(self, changes):
        """
        This allows only reasonable edits that do not lead to an ill-constructed document. E.g. you cannot erase
        one section and an alter it afterwards. This will lead to an error

        :param changes: a list of changes to be made to the paper. Each change is a tuple of (content_type, new_text, [key]).
        :return:
        """
        new_paper = copy.deepcopy(self)
        for x in changes:
            assert len(x) == 2 or len(x) == 3, "expected either 2 or 3 components to each change. type, text, [key]."
            ct = x[0]
            nt = x[1]
            ck = x[2] if len(x) == 3 else None

            if ct.lower().strip() == "table":
                ct = "tables"
            elif ct.lower().strip() == "figure":
                ct = "figures"

            if ct == "tables" and ck not in self.get_tables().keys():
                ck = self.get_table_name(ck)
            elif ct == "figures" and ck not in self.get_figures().keys():
                ck = self.get_figure_name(ck)

            if ct != "abstract" and ck is None:
                raise ValueError(f"Provided {ct} with number {ck} does not exist.")

            # cannot have escapes inside #todo do so only if we are not overriding whole sections! then it should be allowed!!!
            if ct == "sections" or ct == "abstract":
                original_start_idx, original_end_idx = new_paper._find_content(ct, ck)
                new_paper = self._override(original_start_idx, original_end_idx, nt, new_paper)
            else:
                nt = nt.replace("[/FIGURE]", "")
                nt = nt.replace("[/TABLE]", "")
                nt = nt.replace("[/ALGORITHM]", "")

                # get the original lines to be changed from the original paper
                original_start_idx, original_end_idx = self._find_content(ct, ck)
                original_lines = self._md_lines[original_start_idx:original_end_idx + 1]

                # check for the correct position in the new paper
                start_idx, end_idx = -1, -1
                for ix, line in enumerate(new_paper._md_lines):
                    if start_idx > 0 and ix - start_idx > (original_end_idx - original_start_idx):
                        start_idx = -1

                    if line.strip() == original_lines[0].strip():
                        start_idx = ix

                    if start_idx > 0 and line.strip() == original_lines[-1].strip():
                        end_idx = ix
                        break #todo we could have trivial matches on the first and last line which is only meaningful if we edit very unique text in those lines, which we cannot count on

                assert start_idx >= 0 and end_idx >= 0, "failed to find original lines in new paper. "

                # override with new lines
                new_paper = self._override(start_idx, end_idx, nt, new_paper)

        return new_paper

    def _find_content(self, content_type, content_key=None):
        """
        Finds the start and end indices of the content in the markdown lines based on the content type and key.
        """
        if content_type == "sections":
            assert content_key is not None, f"requires name of section to override"

            start_idx = self._structured_md["sections"][content_key][0]
            end_idx = self._structured_md["sections"][content_key][-1]
        elif content_type in ["figures", "tables", "algorithms", "subsections"]:
            assert content_key is not None, f"requires name of {content_type} to override"

            sec = next((k for k, v in self._structured_md[content_type].items() if content_key in v.keys()), None)
            assert sec is not None, f"cannot find {content_type} {content_key} in structured md: {self._structured_md[content_type]}"

            start_idx = self._structured_md[content_type][sec][content_key][0]
            end_idx = self._structured_md[content_type][sec][content_key][-1]-1
        elif content_type == "paragraph":
            assert content_key is not None, "requires paragraph number to override"

            try:
                content_key = int(content_key)
            except ValueError:
                raise ValueError(f"paragraph number {content_key} is not an integer")

            if content_key == 0: # abstract
                return self._structured_md["abstract"][0]-1, self._structured_md["abstract"][-1] #offset start by 1, since the ####Abstract is also part of the lines

            range = self.get_paragraph_lines_by_number(content_key)
            if range is None:
                raise ValueError(f"paragraph {content_key} not found in paper {self.id}")

            start_idx = range[0]
            end_idx = range[1]
        elif content_type == "abstract":
            start_idx = self._structured_md["abstract"][0]
            end_idx = self._structured_md["abstract"][-1]
        else:
            assert content_key is None, f"cannot process content key for {content_type}"

            start_idx = self._structured_md[content_type][0]
            end_idx = self._structured_md[content_type][-1]

        return start_idx, end_idx

    def _override_by_content_type(self, content_type, new_text, content_key=None):
        """
        Overrides the content of the paper based on the content type and key.
        """
        assert content_type in list(self._structured_md.keys()) + ["paragraph"], \
            f"Content type '{content_type}' not allowed. Choose either of {list(self._structured_md.keys())}"

        new_paper = copy.deepcopy(self)
        start_idx, end_idx = self._find_content(content_type, content_key)

        return self._override(start_idx, end_idx, new_text, new_paper)

    def _override(self, start_idx, end_idx, new_text, new_paper):
        """
        Overrides the content of the paper between start_idx and end_idx with new_text.

        :param start_idx: The starting index of the content to be overridden.
        :param end_idx: The ending index of the content to be overridden.
        :param new_text: The new text to replace the existing content.
        :param new_paper: The new paper instance to be modified.
        :return: The modified paper instance with the overridden content.
        """
        new_lines = new_text.split("\n")

        # delete existing lines and insert new ones
        new_paper._md_lines = new_paper._md_lines[:start_idx] + new_lines + (
            new_paper._md_lines[end_idx+1:] if end_idx < len(new_paper._md_lines) - 1 else [])
        new_paper._md = "\n".join(new_paper._md_lines)

        # recompute structured md
        new_paper._to_structured_md()

        return new_paper

    def override(self, content_type, new_text, content_key=None):
        """
        Overrides the content of the paper based on the content type and key with new text.
        """
        return self._override(content_type, new_text, content_key)

    def get_title(self):
        """
        Returns the title of the paper.
        """
        return self.get("title")

    def get(self, content_type, key=None, with_line_numbers=False):
        """
        Retrieves the content of the paper based on the content type and key.
        :param content_type: The type of content to retrieve (e.g., "abstract", "sections", "figures", etc.).
        :param key: The key for the specific content to retrieve (e.g., section name, figure name, etc.).
        :param with_line_numbers: If True, each line will be prefixed with its line number.
        """
        assert content_type in self._structured_md, f"Content type '{content_type}' not allowed. Choose either of {list(self._structured_md.keys())}"

        if isinstance(self._structured_md[content_type], dict):
            if content_type == "subsections":
                assert key is not None, f"Key must be provided for content type '{content_type}'"

                for section in self._structured_md["subsections"]:
                    if key in self._structured_md["subsections"][section]:
                        return self._indexes_to_text(self._structured_md["subsections"][section][key][0],
                                                     self._structured_md["subsections"][section][key][-1],
                                                     with_line_numbers)
                return None
            elif content_type in ["figures", "tables", "algorithms"]:
                assert key is not None, f"Key must be provided for content type '{content_type}'"

                for section in self._structured_md[content_type]:
                    if key in self._structured_md[content_type][section]:
                        return self._indexes_to_text(self._structured_md[content_type][section][key][0],
                                                     self._structured_md[content_type][section][key][-1],
                                                     with_line_numbers)
            elif content_type == "paragraphs":
                if key:
                    return [self._indexes_to_text(p[0],
                                                  p[-1],
                                                  with_line_numbers) for p in self._structured_md["paragraphs"][key]]
                else:
                    return [(section, self._indexes_to_text(p[0], p[-1], with_line_numbers)) for section in
                            self._structured_md["paragraphs"] for p in self._structured_md["paragraphs"][section]]
            else:
                if self._structured_md[content_type] is None:
                    return None
                else:
                    return self._indexes_to_text(self._structured_md[content_type][key][0],
                                                 self._structured_md[content_type][key][-1], with_line_numbers)
        else:
            if self._structured_md[content_type] is None:
                return None
            else:
                return self._indexes_to_text(self._structured_md[content_type][0],
                                             self._structured_md[content_type][-1], with_line_numbers)

    def get_abstract(self, with_line_numbers=False):
        """
        Returns the abstract of the paper.
        """
        return self.get("abstract", with_line_numbers=with_line_numbers)

    def get_section(self, name, with_line_numbers=False):
        """
        Returns the content of a specific section by its name.
        """
        return self.get("sections", name, with_line_numbers=with_line_numbers)

    def get_section_names(self):
        """
        Returns a list of section names in the paper.
        """
        return list(self._structured_md["sections"].keys())

    def get_sections(self, with_line_numbers=False):
        """
        Returns a dictionary of sections with their names as keys and their content as values.
        """
        return {k: self.get_section(k, with_line_numbers) for k in self._structured_md["sections"]}

    def get_paragraphs(self, section=None, with_line_numbers=False, numbered=False):
        """
        Returns a list of paragraphs in the paper, optionally filtered by section.
        """
        res = self.get("paragraphs", key=None, with_line_numbers=with_line_numbers)
        numbers = [i for i in range(1, len(res)+1)]

        if section is not None:
            res_n, numbers_n = [],[]
            for r, n in zip(res, numbers):
                if r[0] == section:
                    res_n.append(r[1])
                    numbers_n.append(n)
            res = res_n
            numbers = numbers_n
        else:
            res = [r[1] for r in res]

        if numbered:
            return [f"paragraph {i}: {p}" for i, p in zip(numbers, res)]
        else:
            return res

    def get_paragraph_by_number(self, paragraph_number:int, with_line_numbers=False):
        """
        Returns the content of a specific paragraph by its number.
        """
        res = self.get("paragraphs", key=None, with_line_numbers=with_line_numbers)

        if paragraph_number == 0:
            return self.get_abstract(with_line_numbers=with_line_numbers)

        if not(0 <= paragraph_number <= len(res)):
            return None

        return res[paragraph_number - 1]

    def get_paragraph_lines_by_number(self, paragraph_number:int):
        """
        Returns the line numbers of a specific paragraph by its number.
        """
        pt = self.get_paragraph_by_number(paragraph_number, with_line_numbers=True)
        if pt is None:
            return None

        pt = pt[1]
        lines = pt.split("\n")

        first_line = None
        last_line = None
        for line in lines:
            line_match = re.match(r"^line (\d+):", line)
            if line_match is None:
                continue

            if first_line is None:
                first_line = int(line_match.group(1))

            last_line = int(line_match.group(1))

        if first_line is None or last_line is None:
            return None

        return first_line, last_line

    def get_subsection(self, name, with_line_numbers=False):
        return self.get("subsections", name, with_line_numbers=with_line_numbers)

    def get_subsections(self, with_line_numbers=False):
        return {kk: self.get_subsection(kk, with_line_numbers=with_line_numbers) for k, v in
                self._structured_md["subsections"].items() for kk in v}

    def get_figure(self, name, multimodal=False):
        res = self.get("figures", name)

        if res is not None:
            return {
                "text": res,
                "media": None if not multimodal else self._load_media(name)
            }
        else:
            return None

    def get_figure_name(self, figure_number:int):
        names = list(self.get_figures().keys())
        for name in names:
            if re.search(fr"\[FIGURE .*F{figure_number}.*]", name) is not None:
                return name

        return None

    def get_table_name(self, table_number: int):
        names = list(self.get_tables().keys())
        for name in names:
            if re.search(fr"\[TABLE .*T{table_number}.*]", name) is not None:
                return name

        return None

    def get_figures(self, section=None, multimodal=False):
        return {kk: self.get_figure(kk, multimodal) for k, v in self._structured_md["figures"].items() for kk in v if section is None or k == section}

    def get_table(self, name):
        return self.get("tables", name)

    def get_tables(self, section=None):
        return {kk: self.get_table(kk) for k, v in self._structured_md["tables"].items() for kk in v if section is None or k == section}

    def get_algorithm(self, name):
        return self.get("algorithms", name)

    def get_algorithms(self, section=None):
        return {kk: self.get_algorithm(kk) for k, v in self._structured_md["algorithms"].items() for kk in v if section is None or k == section}

    def get_text_with_index(self):
        return [f"line {i}: {line}" for i, line in enumerate(self._md_lines)]

    def get_text_with_paragraph_index(self, with_media, multimodal=False):
        output = [self.get_title() + "\n", "paragraph 0: " + self.get_abstract() + "\n"]

        for sec in self.get_section_names():
            paras = self.get_paragraphs(section = sec, numbered=True)
            output += [f"## {sec}\n"] + paras + ["\n"]

            if with_media:
                tbs = self.get_tables(section=sec)
                if len(tbs) > 0:
                    output += ["\n"] + [t for tn, t in tbs.items()] + ["\n"]

                algos = self.get_algorithms(section=sec)
                if len(algos) > 0:
                    output += ["\n"] + [t for tn, t in algos.items()] + ["\n"]

                figs = self.get_figures(section=sec, multimodal=multimodal)
                if len(figs) > 0:
                    output += ["\n"] + [t["text"] for tn, t in figs.items()] + ["\n"]

        return "\n".join(output)

    def _get_passages(self, doc: list[str], start_regex, end_regex):
        passages = {}
        current_passage = None
        for lix, line in enumerate(doc):
            sm = re.match(start_regex, line)
            se = re.match(end_regex, line)

            if se is not None:
                current_passage = None

            if sm is not None:
                current_passage = re.sub("^#+ ", "", sm.group(0)).strip()
                passages[current_passage] = [lix]
            elif current_passage is not None:
                passages[current_passage] += [lix]

        return {k: (v[0], v[-1]) for k, v in passages.items()}

    def _get_subpassages(self, docs: dict[str, list[int]], start_regex, end_regex, exclude_end=True):
        subpassages = {}
        for passage in docs:
            current_subpassage = None
            subpassages[passage] = {}

            for line_index in range(docs[passage][0], docs[passage][-1] + 1):
                sm = re.match(start_regex, self._md_lines[line_index])
                se = re.match(end_regex, self._md_lines[line_index])

                if se is not None:
                    current_subpassage = None

                if sm is not None:
                    current_subpassage = re.sub("^#+ ", "", sm.group(0)).strip()
                    subpassages[passage][current_subpassage] = [line_index]
                elif current_subpassage is not None:
                    subpassages[passage][current_subpassage] += [line_index]

            subpassages[passage] = {k: (v[0], v[-1] + (0 if exclude_end else 1)) for k, v in
                                    subpassages[passage].items()}

        return subpassages

    def _get_paragraphs(self, sections, figures, tables, algos):
        res = {}
        for section in sections:
            cur_para = None
            paras = []

            # get lines to skip
            figures_in_sec = figures[section]
            tables_in_sec = tables[section]
            algos_in_sec = algos[section]

            lines_to_skip = set(
                r for fig, fig_lines in figures_in_sec.items() for r in range(fig_lines[0], fig_lines[-1] + 1))
            lines_to_skip = lines_to_skip.union(
                set(r for fig, fig_lines in tables_in_sec.items() for r in range(fig_lines[0], fig_lines[-1] + 1)))
            lines_to_skip = lines_to_skip.union(
                set(r for fig, fig_lines in algos_in_sec.items() for r in range(fig_lines[0], fig_lines[-1] + 1)))

            for line_index in range(sections[section][0] + 1, sections[section][-1] + 1):
                text = self._md_lines[line_index]

                is_para_break = text == ""
                is_start_of_fig_tbl_algo = re.match(r"^\[(FIGURE|TABLE|ALGORITHM) .*\]$", text) is not None
                is_start_of_subsec = re.match(r"^#+ .+$", text) is not None

                to_skip = line_index in lines_to_skip

                stop_cur_para = is_para_break or is_start_of_fig_tbl_algo or is_start_of_subsec

                # skip lines, close before if not already happened
                if to_skip:
                    # end current paragraph
                    if cur_para is not None and len(cur_para) > 0:
                        paras += [cur_para]
                    cur_para = None

                if stop_cur_para:
                    if cur_para is None or len(cur_para) == 0:
                        cur_para = []
                    else:
                        paras += [cur_para]
                        cur_para = []

                if cur_para is not None and not is_start_of_subsec and not is_start_of_fig_tbl_algo and not is_para_break and not to_skip:
                    cur_para += [line_index]

            res[section] = [(p[0], p[-1]) for p in paras]

        return res

    def _to_structured_md(self):
        try:
            title = next(filter(lambda x: re.match(r"^# (.+)", x[1]) is not None, enumerate(self._md_lines)))[0]
        except StopIteration:
            title = None

        # get abstract
        abstract = self._get_passages(doc=self._md_lines, start_regex="^###### Abstract$", end_regex="^## (.+)$")
        if len(abstract) == 0:
            abstract = None
        else:
            abstract = abstract["Abstract"]

        # get sections and sub sections
        sections = self._get_passages(doc=self._md_lines,
                                      start_regex="^## (.+)$",
                                      end_regex="^## (.+)$")

        # unusually short sections, try with one hash more
        if len(sections) < 3:
            sections2 = self._get_passages(doc=self._md_lines,
                                           start_regex="^### (.+)$",
                                           end_regex="^### (.+)$")

            if len(sections2) >= 3:
                sections = sections2

        subsections = self._get_subpassages(docs=sections,
                                            start_regex="^###+ (.+)$",
                                            end_regex="^##+ (.+)$")
        figures = self._get_subpassages(docs=sections,
                                        start_regex="^\[FIGURE .*\]$",
                                        end_regex="\[/FIGURE\]",
                                        exclude_end=False)
        tables = self._get_subpassages(docs=sections,
                                       start_regex="^\[TABLE .*\]$",
                                       end_regex="\[/TABLE\]",
                                       exclude_end=False)
        algorithms = self._get_subpassages(docs=sections,
                                           start_regex="^\[ALGORITHM .*\]$",
                                           end_regex="\[/ALGORITHM\]",
                                           exclude_end=False)
        paragraphs = self._get_paragraphs(sections, figures, tables, algorithms)

        self._structured_md = {
            "title": (title, title) if title is not None else None,
            "abstract": abstract,
            "sections": sections,
            "subsections": subsections,
            "paragraphs": paragraphs,
            "figures": figures,
            "tables": tables,
            "algorithms": algorithms
        }

    @property
    def meta(self):
        return self._meta

    @meta.setter
    def meta(self, value):
        self._meta = value

    @property
    def id(self):
        return self._id

    @id.setter
    def id(self, value):
        self._id = value

    @property
    def md(self):
        return self._md

    @md.setter
    def md(self, value):
        self._md = value

        self._to_structured_md()

    @property
    def structured_md(self):
        return self._structured_md

    @structured_md.setter
    def structured_md(self, value):
        self._structured_md = value

    def to_json_obj(self):
        return {
            "id": self._id,
            "meta": self._meta,
            "md": self._md,
            "structured_md": self._structured_md
        }

    @staticmethod
    def from_json_obj(obj):
        return Paper(
            id=obj["id"],
            meta=obj["meta"],
            md=obj["md"],
            structured_md=obj["structured_md"] if "structured_md" in obj else None
        )



def load_paper_dataset(path: Path | str, by_split: bool, load_structure: bool = True, pids:list[str] = None, venue_meta: dict | None = None) -> list[Paper]:
    """
    Loads the raw paper dataset from the specified path into a list of papers.

    :param path: the Path to the dataset of a specific venue
    :return: a list of papers
    """
    if isinstance(path, str):
        path = Path(path)

    papers = []
    for pdir in path.iterdir():
        if pdir.is_file():
            continue

        paper_id = pdir.name
        venue_id, paper_sub_id = paper_id.split("%")

        if pids is not None and paper_id not in pids:
            continue

        # load the paper
        with (pdir / "paper.md").open("r") as f:
            paper_md = f.read()

        # load metadata
        with (pdir / "meta.json").open("r") as f:
            paper_meta = json.load(f)

        # set venue config if provided
        if venue_meta:
            paper_meta["venue_config"] = venue_meta

        paper_meta["venue"] = venue_id
        paper_meta["oid"] = paper_id
        paper_meta["media"] = str((pdir / "media").absolute())

        # load structure if applicable
        if load_structure and (pdir / "index.json").exists():
            with (pdir / "index.json").open("r") as f:
                structure = json.load(f)
        else:
            structure = None

        papers += [Paper(id=paper_id, meta={"venue": venue_id, **paper_meta}, md=paper_md, structured_md=structure)]

    if by_split:
        split_path = path / "split.json"
        if split_path.exists():
            with split_path.open("r") as f:
                split = json.load(f)

            papers = {
                "train": [p for p in papers if p.meta["oid"] in split["train"]],
                "dev": [p for p in papers if p.meta["oid"] in split["dev"]],
                "test": [p for p in papers if p.meta["oid"] in split["test"]]
            }
        else:
            raise ValueError("Cannot load dataset by split since it does not exist.")

    return papers


def load_paper_datasets(path: Path | str, venues: list[str] = None, by_split: bool = True, load_structure: bool = True, pids: list[str] = None):
    """
    Loads the raw paper datasets from the specified path into a dictionary of lists of papers.

    :param path: the Path to the dataset of a specific venue
    :return: a dictionary of lists of papers
    """
    if isinstance(path, str):
        path = Path(path)

    # load venue meta data
    venues_meta = {}
    assert (path / "meta").exists() and (path / "meta").is_dir(), "meta venue directory is missing"

    for vm in (path / "meta").iterdir():
        if vm.suffix == ".json":
            with vm.open("r") as f:
                venues_meta[vm.stem] = json.load(f)

    datasets = {}
    for vdir in path.iterdir():
        if vdir.is_file():
            continue

        if vdir.name == "meta":
            continue

        venue_id = vdir.name

        if venues is not None and venue_id not in venues:
            continue

        papers = load_paper_dataset(vdir, by_split, load_structure, pids, venues_meta[venue_id])
        datasets[venue_id] = papers

    if len(datasets) == 0:
        raise ValueError("No datasets found in the specified path.")

    return datasets


def split_paper_dataset(papers: list[Paper], split_ratio: tuple[float, float, float]):
    """
    Splits the dataset of papers into a training and a validation set.

    :param papers: the list of papers
    :param split_ratio: the ratio of the training set
    :return: a tuple of training and validation sets
    """
    assert len(
        split_ratio) == 3, "split ratio has to be a tuple of 3 values; one for training, validation, and test sets"
    assert round(sum(split_ratio), 2) == 1.0, "sum of split ratio has to be 1"
    assert all(0.0 < r < 1.0 for r in split_ratio), "each split ratio has to be between 0 and 1"

    split_sizes = [int(len(papers) * r) for r in split_ratio]
    split_idxs = [sum(split_sizes[:i]) for i in range(len(split_sizes))] + [len(papers)]

    result = []
    for i in range(3):
        start, end = split_idxs[i], split_idxs[i + 1]
        result += [papers[start:end]]

    return result


def store_random_split_dataset(path: Path | str, split_ratio: tuple[float, float, float], seed=None):
    """
    Stores the random split of the dataset into a training and a validation set.

    :param path: the Path to the dataset of a specific venue
    :param split_ratio: the ratio of the training set
    :return: a tuple of training and validation sets
    """
    if isinstance(path, str):
        path = Path(path)

    if seed is not None:
        random.seed(seed)

    # load and shuffle papers
    papers = load_paper_dataset(path, by_split=False)
    random.shuffle(papers)

    # split the papers at random
    train, val, test = split_paper_dataset(papers, split_ratio)

    # store the split
    with (path / "split.json").open("w+") as f:
        json.dump({
            "train": [p.id for p in train],
            "dev": [p.id for p in val],
            "test": [p.id for p in test],
            "seed": seed
        }, f, indent=4)


def store_random_split_datasets(path: Path | str, split_ratio: tuple[float, float, float], seed=None):
    """
    Stores the random split of the datasets into training and validation sets.

    :param path: the Path to the dataset of a specific venue
    :param split_ratio: the ratio of the training set
    :return: a tuple of training and validation sets
    """
    if isinstance(path, str):
        path = Path(path)

    for vdir in path.iterdir():
        store_random_split_dataset(vdir, split_ratio, seed)


def subsample_by_splits(dataset_path, split_samples:dict, seed = None):
    """
    Subsamples the dataset by the specified split samples and stores the result in a JSON file.
    """
    random.seed = seed

    if isinstance(dataset_path, str):
        dataset_path = Path(dataset_path)

    papers = load_paper_datasets(dataset_path, venues=None, by_split=True)

    res = {}
    for venue in papers:
        for split in papers[venue]:
            assert split in split_samples, f"Split {split} not found in split_samples"

            random.shuffle(papers[venue][split])
            subset = papers[venue][split][:split_samples[split]]

            res[split] = res.get(split, []) + [p.id for p in subset]

    res["seed"] = seed

    with open(dataset_path / "split.json", "w+") as file:
        json.dump(res, file, indent=4)
