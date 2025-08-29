from collections import Counter

import spacy

from ...data import Review
from ...framework.rcd import ReviewChangeDetector, ReviewDelta
from .utils import get_review_text_data

import Levenshtein

import evaluate


class SurfaceRCD(ReviewChangeDetector):
    """
    This class determines changes between two reviews using surface features.
    """
    def __init__(self):
        super().__init__("surface_rcd")

    def run(self, review1: Review, review2: Review, **config) -> ReviewDelta:
        result = {"by_section": {}}

        data = get_review_text_data(review1, review2)
        r1_full, r2_full = data["r1"], data["r2"]

        r1_by_section, r2_by_section = data["r1_sections"], data["r2_sections"]
        added_sections, removed_sections, common_sections = data["added_sections"], data["removed_sections"], data["common_sections"]

        result["section_changes"] = {
            "added": list(added_sections),
            "removed": list(removed_sections)
        }

        # 1) compute Levensthein edits
        result["levensthein_edits"] = Levenshtein.editops(r1_full, r2_full)
        result["by_section"]["levensthein_edits"] = {
            section: Levenshtein.editops(r1_by_section[section], r2_by_section[section]) for section in common_sections
        }

        # 2) compute vocab changes
        nlp = spacy.load("en_core_sci_sm")
        r1_parsed = nlp(r1_full)
        r2_parsed = nlp(r2_full)
        r1_parsed_sections = {section: nlp(r1_by_section[section]) for section in common_sections}
        r2_parsed_sections = {section: nlp(r2_by_section[section]) for section in common_sections}

        voc1 = {token.lemma_ for token in r1_parsed}
        voc2 = {token.lemma_ for token in r2_parsed}

        stopwords = [token for token in nlp.Defaults.stop_words] + ["*", "#", "\n", "\n\n", "-", ",", "."]

        result["vocab_edits"] = {
            "most_common1": [w for w in Counter([token.lemma_ for token in r1_parsed]).most_common() if w not in stopwords],
            "most_common2": [w for w in Counter([token.lemma_ for token in r2_parsed]).most_common() if w not in stopwords],
            "added": list(voc2 - voc1),
            "removed": list(voc1 - voc2),
            "common": list(voc1 & voc2)
        }

        # 4) compute n-gram overlap
        r1_tokens = [token.text.lower() for token in r1_parsed if token.text.lower() not in stopwords]
        r2_tokens = [token.text.lower() for token in r2_parsed if token.text.lower() not in stopwords]

        n = 3
        r1_ngrams = []
        r2_ngrams = []
        for i in range(len(r1_tokens) - n + 1):
            r1_ngrams.append(" ".join(r1_tokens[i:i+n]))
        for i in range(len(r2_tokens) - n + 1):
            r2_ngrams.append(" ".join(r2_tokens[i:i+n]))

        r1_ngrams_unique = set(r1_ngrams)
        r2_ngrams_unique = set(r2_ngrams)

        result["3gram_overlap"] = {
            "most_common_3grams1": list(Counter(r1_ngrams).most_common(50)),
            "most_common_3grams2": list(Counter(r1_ngrams).most_common(50)),
            "added": list(r2_ngrams_unique - r1_ngrams_unique),
            "removed": list(r1_ngrams_unique - r2_ngrams_unique),
            "common": list(r1_ngrams_unique & r2_ngrams_unique)
        }

        # 5) compute length changes
        result["token_count_changes"] = {
            "diff": len(r2_tokens) - len(r1_tokens)
        }

        result["by_section"]["token_count_changes"] = {
            section: len(r1_parsed_sections[section]) - len(r2_parsed_sections[section]) for section in common_sections
        }

        # 6) compute rouge-l
        rouge = evaluate.load("rouge")
        rouges = rouge.compute(predictions=[r1_full], references=[r2_full])

        result["rouge-1"] = rouges["rouge1"]
        result["rouge-2"] = rouges["rouge2"]
        result["rouge-l"] = rouges["rougeL"]

        result["by_section"]["rouge-l"] = {
            section: rouge.compute(predictions=[r1_by_section[section]], references=[r2_by_section[section]])["rougeL"] for section in common_sections
        }

        return ReviewDelta(review1, review2, result)