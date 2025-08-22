from cerg.data import Paper
from cerg.framework.cfg import PaperCounterfactualGenerator, PaperCounterfactual


class NoChangeCF(PaperCounterfactualGenerator):
    """
    This counterfactual generator is meant only for debugging purposes. It produces a counterfactual that is
    identical to the original o_paper.
    """
    def __init__(self, name=None):
        super().__init__(name="debug_cf_without_changes" if name is None else name)

    def single_run(self, paper:Paper, **config) -> PaperCounterfactual:

        return PaperCounterfactual(paper, self.name, paper, []) # apply no changes whatsoever