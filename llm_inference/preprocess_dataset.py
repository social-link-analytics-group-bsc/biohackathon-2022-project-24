# To transform the json into the meaninful json answer to train the model.
# Based on Blanca Calvo script and adapted
import os
import sys
import yaml
import collections
import argparse
from datasets import load_dataset, Dataset, ClassLabel

from prompt_instructions import prompt_instruction_3


sys.path.append(os.path.join(os.path.dirname(__file__), ".."))


def parsing_arguments(parser):
    parser.add_argument(
        "--data",
        type=str,
        default="nothing",
        help="Prodigy data to transform into conll",
    )
    return parser


def _extract_labels(example):
    token_to_label = {}
    try:
        for span in example["spans"]:
            if span["token_start"] == span["token_end"]:
                label = span["label"]
                #FIXME As soon as it is fixed in prodigy, can remove the if
                if label == "n_fem":
                    label = "n_female"
                # if label not in token_to_label.keys():
                #     print(label)
                #     raise
                value = example["tokens"][span["token_start"]]["id"]
                token_to_label.setdefault(label, []).append(value)
            # FIXME no idea what this part does it is Blanca code
            else:
                for r in range(
                    span["token_start"], span["token_end"] + 1
                ):  # TODO: finish this
                    label = span["label"]
                    #FIXME As soon as it is fixed in prodigy, can remove the if
                    if label == "n_fem":
                        label = "n_female"
                    value = example["tokens"][span["token_start"]]["id"]
                    token_to_label.setdefault(label, []).append(value)
    except KeyError:  # if there are no spans
        token_to_label = {}
    example["labels"] = token_to_label
    return example


def _create_description(example):
    """
    Parse the example and transform the type of answer to a text
    The different answers are:
        - 'accept': Accept: All information is clear and about human subject
        - 'ignore': Ignore: This is not about human subjects
        - 'reject': Reject: There are label(s) but at least one is confusing.
    """
    if example["answer"] == "accept":
        example["reason"] = "Accept: All information is clear and about human subject"
    elif example["answer"] == "reject":
        example[
            "reason"
        ] = """Reject: There are label(s) but at least one is confusing.   
                             There is at least 1 label that you are doubting if or how to annotate based on the provided rules."""
    elif example["answer"] == "ignore":
        example[
            "reason"
        ] = """There are no labels of interest. 
There are no labels of interest. For instance, the paper is about animals (not human), cell lines, strains, etc. In other words, there are no numbers or percentages of human subjects, males or females."""

    return example


def _create_full_json_answer(example):
    """
    Create the final answer for the model training.
    No more information is used for the training
    """
    example["answer_training"] = str(
        {"answer": example["answer"], "labels": example["labels"]}
    )
    return example


def _create_prompt(example, prompt_instruction):

    example["prompt_instruction"] = f"""{prompt_instruction}"""
    return example


def _create_prompt_text(example, prompt_instruction):
    example[
        "prompt_text"
    ] = f"""{example['prompt_instruction']}
    ## Article text
    {example['text']}
    """
    return example


def _create_chat_data(example):
    example["message"] = [
        {
            "role": "user",
            "content": f"""{example['prompt_instruction']}\n##Article text\n{example['text']}""",
        },
        {"role": "assistant", "content": example['answer_training']},
    ]
    return example


def _transform_key_meta(example):
    """
    Rename the embedded key meta.pmcid to pmcid
    """
    example["pmcid"] = example["meta"]["pmcid"]
    return example


def _drop_unused_data(dataset: Dataset) -> Dataset:
    """
    Use the built-in function for dropping columns.
    Need the dataset and not an example here
    Dropping all the columns except:
    - answer: answer for the model
    - pmcid: pmcid to be able to match the article
    - prompt: needed to generate the prompt
    - _annotator_id: The annotator in case random on that is important
    """
    col_to_drop = [
        "text",
        "reason",
        "spans",
        "tokens",
        "meta",
        "label",
        "labels",
        "_input_hash",
        "_task_hash",
        "_view_id",
        "_timestamp",
        "_session_id",
    ]
    dataset = dataset.remove_columns(col_to_drop)
    return dataset


def cast_label(dataset: Dataset) -> Dataset:
    """
    Cast the column answer as ClassLabel type to
    stratify in the test/train split
    """
    new_features = dataset.features.copy()
    new_features["answer"] = ClassLabel(names=["accept", "reject", "ignore"])
    dataset = dataset.cast(new_features)
    return dataset


def print_simple_info(dataset):
    print(dataset)
    print(collections.Counter(dataset["answer"]))
    print(collections.Counter(dataset["_annotator_id"]))


def main():
    parser = argparse.ArgumentParser()
    parser = parsing_arguments(parser)
    args = parser.parse_args()

    config_path = os.path.join(os.path.dirname(__file__), "../config", "config.yaml")
    config_all = yaml.safe_load(open(config_path))
    training_set_path = config_all["llm_params"]["training_set_path"]

    dataset = load_dataset("json", data_files=args.data, split="train")
    dataset = dataset.map(_extract_labels)
    dataset = dataset.map(_create_description)
    dataset = dataset.map(_transform_key_meta)
    dataset = dataset.map(_create_full_json_answer)
    dataset = dataset.map(
        _create_prompt, fn_kwargs={"prompt_instruction": prompt_instruction_3}
    )
    dataset = dataset.map(_create_chat_data)
    dataset = cast_label(dataset)
    dataset = _drop_unused_data(dataset)
    print_simple_info(dataset)

    # Save the dataset as an arrow file
    dataset.save_to_disk(training_set_path)


if __name__ == "__main__":
    main()