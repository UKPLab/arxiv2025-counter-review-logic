import copy
import io
import re

from pathlib import Path




class Review:
    def __init__(self,
                 sections: dict[str, str] | str,
                 scores: dict[str, int | float] | None = None,
                 id: str = None,
                 meta: dict[str, str] = None,
                 main_section="main",
                 overall_score="overall"):
        if type(sections) == str:
            sections = {main_section: sections}

        assert id is not None or main_section in sections, f"failed to provide a main section called '{main_section}' in {list(sections.keys())}"
        assert id is not None or scores is None or overall_score in scores, f"failed to provide an overall score called '{overall_score}' in {list(scores.keys())}"

        self._sections = sections
        if scores is not None:
            self._scores = scores
        else:
            self._scores = {overall_score: None}

        self.main_section_title = main_section
        self.overall_score_title = overall_score

        self._meta = meta

        self._id = id

    @property
    def sections(self):
        return self._sections

    @sections.setter
    def sections(self, value):
        self._sections = value

    @property
    def scores(self):
        return self._scores

    @scores.setter
    def scores(self, value):
        self._scores = value

    @property
    def id(self):
        return self._id

    @id.setter
    def id(self, value):
        self._id = value

    @property
    def meta(self):
        return self._meta

    @meta.setter
    def meta(self, value):
        self._meta = value

    def get_main_section(self):
        return self._sections[self.main_section_title]

    def get_overall_score(self):
        return self._scores[self.overall_score_title] if self.overall_score_title in self._scores else None

    def get_text(self):
        return "\n".join(f"### {k}\n{v}\n" for k, v in self._sections.items())

    def __str__(self):
        return f"REVIEW {self.id} [Text]: " + str(self.sections) + "; [Scores]: " + str(self.scores)

    def to_json(self, id_only=False):
        """
        Converts the Review instance to a JSON serializable dictionary.
        """
        if id_only:
            return {
                "id": self._id
            }
        else:
            return {
                "id": self._id,
                "sections": self._sections,
                "scores": self._scores,
                "meta": self._meta
            }

    @staticmethod
    def from_json(obj):
        """
        Creates a Review instance from a JSON object.
        """
        return Review(
            sections=obj["sections"] if "sections" in obj else None,
            scores=obj["scores"] if "scores" in obj else None,
            id=obj["id"] if "id" in obj else None,
            meta=obj["meta"] if "meta" in obj else None
        )
