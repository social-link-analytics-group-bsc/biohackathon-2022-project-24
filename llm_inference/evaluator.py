import sys
import argparse
import json
import os
import datetime
import logging

from langchain.evaluation import JsonEditDistanceEvaluator
from datasets import DatasetDict
from utils.model_loader import ModelLoader
from utils.post_process_answer import format_answer
from utils.preprocess_dataset import process_dataset, print_simple_info
from utils.utils import (
    dynamic_import,
    load_config,
)

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))


def setup_logger() -> logging.Logger:
    """Setup the logger configuration for consistency."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[logging.StreamHandler()],
    )
    return logging.getLogger(__name__)


logger = setup_logger()


def clean_keys(answer: dict, reference: dict) -> dict:
    """
    Recursively ensure the keys in 'answer' match those in 'reference'.
    Unmatched keys in 'answer' are pruned.
    """
    if not isinstance(reference, dict) or not isinstance(answer, dict):
        return answer

    pruned = {}
    for key in set(reference.keys()) | set(answer.keys()):
        ref_value = reference.get(key)
        ans_value = answer.get(key)
        if ref_value == ans_value:
            if isinstance(ref_value, dict) and isinstance(ans_value, dict):
                pruned[key] = clean_keys(ans_value, ref_value)
            else:
                pruned[key] = ref_value
    return pruned


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments for the script."""
    parser = argparse.ArgumentParser(description="Model Evaluation Script")

    parser.add_argument(
        "--model", required=True, type=str, help="Path or name of the model to load."
    )
    parser.add_argument(
        "--quantization", type=str, help="Quantization level for the model."
    )
    parser.add_argument(
        "--instruct",
        action="store_true",
        help="Flag indicating if the model is an instruct model.",
    )
    parser.add_argument(
        "--adapter",
        action="store_true",
        help="Flag to indicate if an adapter should be loaded.",
    )
    parser.add_argument(
        "--adapter_quantization",
        default=False,
        type=str,
        help="Specify if the adapter is quantized.",
    )
    parser.add_argument(
        "--prompt", required=True, type=str, help="Path to the prompt file."
    )
    parser.add_argument(
        "--full_eval",
        action="store_true",
        help="Run evaluation on the full dataset without splitting.",
    )
    parser.add_argument(
        "--training_set", required=True, type=str, help="Path to the training dataset."
    )

    return parser.parse_args()


def return_eval_score(
    evaluator: JsonEditDistanceEvaluator, ref_json: dict, pred_json: dict
) -> float:
    """Calculate evaluation score using an evaluator object."""
    score = evaluator.evaluate_strings(prediction=pred_json, reference=ref_json)
    return score["score"]


def split_dataset(ds_training_set: DatasetDict, full_eval: bool) -> DatasetDict:
    """Split dataset into train, validation, and test sets."""
    if full_eval:
        return ds_training_set

    # First split for training and test sets
    ds_training_set = ds_training_set.train_test_split(
        test_size=0.2, seed=42, stratify_by_column="answer"
    )

    # Split test set further into validation and test sets
    ds_devtest = ds_training_set["test"].train_test_split(
        test_size=0.5, seed=42, stratify_by_column="answer"
    )

    return DatasetDict(
        {
            "training": ds_training_set["train"],
            "validation": ds_devtest["train"],
            "test": ds_devtest["test"],
        }
    )


def evaluate_model_on_data(
    llm_model: object, data_loaded: DatasetDict, prompt_instruction: str
) -> dict:
    """Run evaluation on the dataset and return scores by answer type."""
    evaluator = JsonEditDistanceEvaluator()
    scores_by_answer_type = {}

    for idx, data in enumerate(data_loaded):
        answer_training = data["answer_training"]
        answer_type = data["answer"]
        pmcid = data["pmcid"]
        method_text = data["text"]

        predicted_answer = llm_model.passing_article_to_llm(
            prompt_instruction=prompt_instruction, text=method_text
        )

        clean_ref_json = format_answer(answer_training)
        clean_pred_json = format_answer(predicted_answer)

        if clean_ref_json is None or clean_pred_json is None:
            logger.warning(f"Skipping data point {pmcid} due to invalid format.")
            continue
        clean_pred_json = clean_keys(clean_pred_json, answer_training)

        if answer_type == 0:
            logger.debug(f"Cleaned pred json: {clean_pred_json}")
            logger.debug(f"Cleaned ref json: {clean_ref_json}")

        score = return_eval_score(evaluator, clean_ref_json, clean_pred_json)
        scores_by_answer_type.setdefault(answer_type, []).append(score)

        logger.info(f"{pmcid}: {answer_type}: {score}")

        if idx >= 2:  # Adjust or remove this condition as needed
            break

    return scores_by_answer_type


def calculate_average_scores(scores_by_answer_type: dict) -> tuple[float, dict]:
    """Calculate overall and per answer-type average scores."""
    total_scores = [
        score for scores in scores_by_answer_type.values() for score in scores
    ]
    if not total_scores:
        return 0.0, {}

    overall_average_score = sum(total_scores) / len(total_scores)

    avg_score_by_answer_type = {
        answer_type: sum(scores) / len(scores)
        for answer_type, scores in scores_by_answer_type.items()
    }

    return overall_average_score, avg_score_by_answer_type


def save_results(eval_result_path: str, results: dict) -> None:
    """Save evaluation results to a JSON file."""
    with open(eval_result_path, "a") as result_file:
        json.dump(results, result_file, indent=2)
        result_file.write(",\n")


def main():
    args = parse_arguments()
    logger.info(f"Arguments: {args}")

    # Load config
    config_path = os.path.join(os.path.dirname(__file__), "../config", "config.yaml")
    config_all = load_config(config_path)
    logger.info(f"Using model: {args.model}")

    # Initialize ModelLoader
    model_loader = ModelLoader(
        model_path=args.model,
        quantization=args.quantization,
        instruct=args.instruct,
        adapter=args.adapter,
        adapter_quantization=args.adapter_quantization,
        generation_params=config_all["llm_params"]["generation_params"],
        bits_and_bytes_config=config_all["llm_params"]["bits_and_bytes"],
    )

    # Load the model
    llm_model = model_loader.get_model()

    # Load prompt instruction
    prompt_instruction = dynamic_import(f"utils.{args.prompt}", "prompt_instruction")

    # Load dataset
    logger.info("Loading dataset for evaluation")
    ds_training_set = process_dataset(args.training_set, prompt_instruction)
    logger.info(f"Dataset loaded:\n{ds_training_set}")

    data_loaded = split_dataset(ds_training_set, args.full_eval)
    logger.info(f"Data loaded:\n{data_loaded}")

    # Evaluate the model
    scores_by_answer_type = evaluate_model_on_data(
        llm_model, data_loaded["test"], prompt_instruction
    )

    # Calculate overall and by-answer-type scores
    overall_score, score_by_answer_type = calculate_average_scores(
        scores_by_answer_type
    )
    logger.info(f"Overall average score: {overall_score}")
    logger.info(f"Average score by answer type: {score_by_answer_type}")

    # Save results
    results = {
        "date": datetime.datetime.now().strftime("%d/%m/%Y %H:%M"),
        "model_name": os.path.basename(args.model),
        "prompt": args.prompt,
        "dataset": args.training_set,
        "full_eval": args.full_eval,
        "model_quantization": model_loader.model_quantization,
        "adapter": args.adapter,
        "adapter_quantization": args.adapter_quantization,
        "score_agg": overall_score,
        "scores_by_type": score_by_answer_type,
        "eval_size": len(
            [score for scores in scores_by_answer_type.values() for score in scores]
        ),
    }
    save_results(config_all["llm_params"]["eval_result_path"], results)
    logger.info("Evaluation results saved.")


if __name__ == "__main__":
    main()
