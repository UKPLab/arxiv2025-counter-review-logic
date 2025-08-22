from ..data import Paper
from . import PaperCounterfactual, PaperCounterfactualGenerator


class Perturbator():
    def __init__(self, name):
        self._name = name

    def __call__(self, paper: Paper, feedback: str=None, previous_solution: PaperCounterfactual=None) -> PaperCounterfactual:
        """
        Perturbs a given paper and returns a counterfactual.

        :param paper: the original paper
        :param feedback: optional feedback from an inspector
        :return: the counterfactual paper
        """
        raise NotImplementedError()

    @property
    def name(self):
        return self._name


class Inspector():
    def __init__(self, name):
        self._name = name

    def __call__(self, paper: Paper, paper_cf: PaperCounterfactual) -> tuple[float, str]:
        """
        Inspects a given paper and its counterfactual and returns a score and feedback.

        :param paper: the original paper
        :param paper_cf: the counterfactual paper
        :return: a tuple of score and feedback
        """
        raise NotImplementedError()

    @property
    def name(self):
        return self._name


class PerturbatorInspectorCF(PaperCounterfactualGenerator):
    def __init__(self, name, perturbator, inspector=None, config=None):
        super().__init__(name=name + "_picf")

        self.perturbator = perturbator
        self.inspector = inspector
        self.config = config

    def _setup(self):
        if self.config is None:
            self.config = {}

    def single_run(self, paper: Paper, **config) -> PaperCounterfactual:
        it_to_end = config.get("max_iterations", 3)
        accept_threshold = config.get("accept_threshold", 0.5)

        feedback = None
        paper_cf = None
        while True:
            # terminate by number of repetitions
            it_to_end -= 1
            if it_to_end < 0:
                break

            # generate cf
            paper_cf = self.perturbator(paper, feedback, paper_cf)

            if paper_cf is None: # some failure occured, repeat
                continue

            # skip validation if no inspector is given
            if self.inspector is None:
                return paper_cf

            # validate cf
            score, feedback = self.inspector(paper, paper_cf)

            # terminate by validation
            if score > accept_threshold:
                return paper_cf

        if paper_cf is None:
            raise ValueError("Could not generate a counterfactual paper")

        return paper_cf