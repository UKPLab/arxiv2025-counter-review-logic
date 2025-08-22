from cerg.experiments.utils import get_train_data
from cerg.models.argtors.DebugARGtor import DebugARGtor
from cerg.models.cfgens.NoChangeCF import NoChangeCF
from cerg.models.rcds.DebugRCD import DebugReviewChangeDetector
from cerg.models.rdes.DebugRDE import DebugReviewDeltaEvaluator
from cerg.pipeline import full_pipeline

output_path = "/home/dycke/Projects/CERGS/results"
dataset_path = "/home/dycke/Projects/CERGS/data/papers/data"
venues = ["2023.acl"]

papers = get_train_data(dataset_path)
papers = [p for k, v in papers.items() for p in v if k in venues]

argtors = [DebugARGtor("debug_argtor"), DebugARGtor("debug_argtor2!")]
cf_gens = [NoChangeCF(), NoChangeCF("another_cfgen")]
rcd = DebugReviewChangeDetector()
rde = DebugReviewDeltaEvaluator()

# run pipeline with only counterfactuals
res = full_pipeline(paper_dataset=papers,
                    paper_dataset_name=f"debug.{'_'.join(venues)}",
                    experiment_name="debug",
                    cf_generators=cf_gens,
                    out_path=output_path,
                    argtors=argtors,
                    rcd=rcd,
                    rde=rde,
                    argtor_config=None,
                    rcd_config=None,
                    cached=True)
