import logging
import os
from collections import Counter, defaultdict
from pathlib import Path

import spacy
import torch
from transformers import AutoTokenizer

from cerg.data import Review
from cerg.framework.rcd import ReviewChangeDetector, ReviewDelta
from cerg.models.rcds.utils import get_review_text_data

ASPECT_CATEGORIES = ['-', 'AI', 'Ablation', 'Accuracy', 'Adaptation', 'Adversarial', 'Agent', 'Algorithm', 'Analysis',
                     'Annotation', 'Application', 'Approach', 'Architecture', 'Assumption', 'Attention', 'Baseline',
                     'Benchmark', 'Clarity', 'Comparison', 'Complexity', 'Confusion', 'Contribution', 'Data',
                     'Definition', 'Description', 'Detail', 'Discussion', 'Effectiveness', 'Efficiency',
                     'Embeddings', 'Evaluation', 'Evidence', 'Experiment', 'Explanation', 'Figure', 'Findings',
                     'Fine-tuning', 'Framework', 'Generalization', 'Grammar', 'Hypothesis', 'Impact',
                     'Implementation', 'Importance', 'Improvement', 'Interpretation', 'Intuition', 'Justification',
                     'Method', 'Metric', 'Model', 'Motivation', 'Notation', 'Novelty', 'Parameter', 'Performance',
                     'Presentation', 'Prompt', 'Related Work', 'Result', 'Robustness', 'Significance',
                     'Statistical Significance', 'Table', 'Task', 'Technique', 'Terminology', 'Theory',
                     'Training', 'Transformer', 'Typo', 'Validation']


class AspectRCD(ReviewChangeDetector):
    """
    This class determines changes between two reviews consider the covered aspects
    in the review using the model by Shen, Kuznetsov, and Gurevych (2025)

    """

    def __init__(self, model_path: str = None):
        super().__init__("aspect_rcd")

        if model_path is None:
            assert "ASPECT_RCD_MODEL_PATH" in os.environ, "Please provide the path to the model"
            self.model_path = os.environ["ASPECT_RCD_MODEL_PATH"]
        else:
            self.model_path = model_path

        if type(self.model_path) == str:
            self.model_path = Path(self.model_path)

        self._setup()

    def _setup(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model_id = "20250102_135556" # fine-grained

        self.model = torch.load(self.model_path / f'{model_id}.pth', weights_only=False).to(device)
        self.tokenizer = AutoTokenizer.from_pretrained('FacebookAI/roberta-base')
        self.model.eval()

    def _predict_aspects(self, sentences):
        result = []
        for sentence in sentences:
            # convert to input format
            inp = self.tokenizer(sentence,
                                 return_tensors='pt',
                                 truncation=True,
                                 max_length=512)
            inp = {k: v.to(self.model.device) for k, v in inp.items()}

            # predict
            with torch.no_grad():
                output = self.model(**inp)

            logits = output.logits
            prediction = torch.sigmoid(logits) > 0.5
            prediction = prediction.cpu().numpy().squeeze()

            # add labels
            labels = []
            for i in range(len(prediction)):
                if prediction[i] == 1:
                    labels.append(ASPECT_CATEGORIES[i])

            result += [labels]

        return result

    def run(self, review1: Review, review2: Review, **config) -> ReviewDelta:
        result = {"by_section": {}}

        data = get_review_text_data(review1, review2)
        r1_by_section, r2_by_section = data["r1_sections"], data["r2_sections"]
        common_sections = data["common_sections"]

        # 1. split into sentences, remember the section
        nlp = spacy.load("en_core_sci_sm")

        r1_sentences = []
        r1_sentences_by_section = {}
        for section in r1_by_section:
            sentences = [sent.text for sent in nlp(str(r1_by_section[section])).sents]
            s, e = len(r1_sentences), len(r1_sentences) + len(sentences)
            r1_sentences_by_section[section] = range(s, e)
            r1_sentences += sentences

        r2_sentences = []
        r2_sentences_by_section = {}
        for section in r2_by_section:
            sentences = [sent.text for sent in nlp(str(r2_by_section[section])).sents]
            s, e = len(r2_sentences), len(r2_sentences) + len(sentences)
            r2_sentences_by_section[section] = range(s, e)
            r2_sentences += sentences

        # 2. predict aspects for each sentence
        r1_aspects = self._predict_aspects(r1_sentences)
        r2_aspects = self._predict_aspects(r2_sentences)

        r1_aspects_by_section = {
            section: [r1_aspects[i] for i in r1_sentences_by_section[section]]
            for section in r1_sentences_by_section
        }
        r2_aspects_by_section = {
            section: [r2_aspects[i] for i in r2_sentences_by_section[section]]
            for section in r2_sentences_by_section
        }

        # 3. compare the aspects on the whole review and per common section
        result["aspects"] = {
            "review1": r1_aspects,
            "review2": r2_aspects
        }
        result["by_section"]["aspects"] = {
            "review1": r1_aspects_by_section,
            "review2": r2_aspects_by_section
        }

        aspect_set_r1 = set(a for aspects in r1_aspects for a in aspects)
        aspect_set_r2 = set(a for aspects in r2_aspects for a in aspects)

        result["aspect_changes"] = {
            "common_aspects": list(aspect_set_r1 & aspect_set_r2),
            "added_aspects": list(aspect_set_r2 - aspect_set_r1),
            "removed_aspects": list(aspect_set_r1 - aspect_set_r2)
        }

        result["by_section"]["aspect_changes"] = {}
        for section in common_sections:
            aspect_set_section_r1 = set(a for aspects in r1_aspects_by_section[section] for a in aspects)
            aspect_set_section_r2 = set(a for aspects in r2_aspects_by_section[section] for a in aspects)

            result["by_section"]["aspect_changes"][section] = {
                "common_aspects": list(aspect_set_section_r1 & aspect_set_section_r2),
                "added_aspects": list(aspect_set_section_r2 - aspect_set_section_r1),
                "removed_aspects": list(aspect_set_section_r1 - aspect_set_section_r2)
            }

        result["sentences"] = {
            "review1": r1_sentences,
            "review2": r2_sentences,
        }

        return ReviewDelta(review1, review2, result)
