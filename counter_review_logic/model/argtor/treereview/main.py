# main.py
import argparse
import json
import os
import traceback

from .models.paper import Paper
from .utility.paper_loader import PaperLoader
from .agents.question_generator import QuestionGenerator
from .utility.context_ranker import ContextRanker
from .agents.answer_synthesizer import AnswerSynthesizer
from .utility.LLMClient import LLMClient
from .core import ReviewPipeline, PipelineConfig


def parse_arguments():
    """Parse command-line arguments for the review pipeline."""
    parser = argparse.ArgumentParser(
        description="Run TreeReview pipeline for generating peer review feedback on a single paper.")
    parser.add_argument("--paper-id", type=str, required=True, help="Unique identifier for the paper.")
    parser.add_argument("--mmd-path", type=str, required=True, help="Path to the paper content in MMD format.")
    parser.add_argument("--output-path", type=str, default="review_output.json",
                        help="Path to save the generated review result. Default is 'review_output.json'.")
    parser.add_argument("--state-file", type=str, default=None,
                        help="Path to the checkpoint state file. If not provided, defaults to 'checkpoint_<paper_id>.json'.")
    parser.add_argument("--max-depth", type=int, default=4,
                        help="Maximum depth of the question tree. Default is 4.")
    parser.add_argument("--retrieval-top-k", type=int, default=3,
                        help="Number of top chunks to retrieve for context. Default is 3.")
    parser.add_argument("--chunk-size", type=int, default=1024,
                        help="Size of text chunks for processing. Default is 1024.")
    return parser.parse_args()


def load_config(args) -> PipelineConfig:
    """Load pipeline configuration based on command-line arguments."""
    return PipelineConfig(
        max_depth=args.max_depth,
        retrieval_top_k=args.retrieval_top_k
    )


def initialize_agents():
    """Initialize the core agents for question generation, context ranking, and answer synthesis."""
    llm = LLMClient()
    question_gen = QuestionGenerator(llm=llm)
    context_ranker = ContextRanker()
    answer_syn = AnswerSynthesizer(llm=llm)
    return question_gen, context_ranker, answer_syn


def load_paper(paper_id: str, mmd_path: str) -> Paper:
    """Load paper data using the provided paper ID and MMD file path."""
    try:
        paper_loader = PaperLoader(paper_id=paper_id, mmd_path=mmd_path)
        return paper_loader.get_paper()
    except Exception as e:
        print(f"Error loading paper: {e}")
        traceback.print_exc()
        raise


def main():
    """Main function to run the TreeReview pipeline for a single paper."""
    args = parse_arguments()

    # Ensure output directory exists
    output_dir = os.path.dirname(args.output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    try:
        # Load paper data
        print(f"Loading paper with ID: {args.paper_id}")
        paper = load_paper(args.paper_id, args.mmd_path)

        # Load configuration and initialize agents
        config = load_config(args)
        question_gen, context_ranker, answer_syn = initialize_agents()

        # Build pipeline for the paper
        print("Initializing review pipeline...")
        pipeline = ReviewPipeline(
            paper=paper,
            question_generator=question_gen,
            context_ranker=context_ranker,
            answer_synthesizer=answer_syn,
            config=config,
            state_file=args.state_file
        )

        # Run pipeline to generate review
        print("Running TreeReview pipeline...")
        result = pipeline.run()

        # Save results to output file
        print(f"Saving results to {args.output_path}")
        with open(args.output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print("Review generation completed successfully.")

    except Exception as e:
        print(f"Error during pipeline execution: {e}")
        traceback.print_exc()
        exit(1)


if __name__ == "__main__":
    main()
