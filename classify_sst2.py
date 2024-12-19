import os
import pandas as pd
import polars as pl
import torch
from torch.utils.data import DataLoader
import numpy as np
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from data_handling import create_few_shot_sst2_messages, SST2Dataset
from tqdm import tqdm
import argparse
import json
import warnings


def load_sst2_data():
    splits = {
        "train": "data/train-00000-of-00001.parquet",
        "validation": "data/validation-00000-of-00001.parquet",
        "test": "data/test-00000-of-00001.parquet",
    }
    sst2 = pl.read_parquet("hf://datasets/stanfordnlp/sst2/" + splits["train"])
    sst2_train, sst2_test = train_test_split(sst2, test_size=0.5, random_state=42, shuffle=True, stratify=sst2["label"])
    # sst2_test = pl.read_parquet("hf://datasets/stanfordnlp/sst2/" + splits["test"])
    return sst2_train, sst2_test


def load_model(model_id):
    tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side="left", add_prefix_space=False)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    if "gemma" in model_id:
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto",
            quantization_config=quantization_config,
        )
        # update chat template to allow for assistant role to go first
        tokenizer.chat_template = "{{ bos_token }}{% for message in messages %}\
            {% if (message['role'] == 'assistant') %}{% set role = 'model' %}\
            {% else %}{% set role = message['role'] %}{% endif %}\
            {{ '<start_of_turn>' + role + '\n' + message['content'] | trim + '<end_of_turn>\n' }}{% endfor %}\
            {% if add_generation_prompt %}{{'<start_of_turn>model\n'}}{% endif %}"
    elif "llama" in model_id:
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_id, device_map="auto", quantization_config=quantization_config, torch_dtype="auto"
        )
    elif "Falcon" in model_id:
        quantization_config = BitsAndBytesConfig(load_in_4bit=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_id, device_map="auto", quantization_config=quantization_config, torch_dtype="auto"
        )
    elif "Mistral" in model_id:
        quantization_config = BitsAndBytesConfig(load_in_4bit=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_id, device_map="auto", quantization_config=quantization_config, torch_dtype="auto"
        )

    else:
        warnings.warn(f"Model {model_id} not supported")
        model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", torch_dtype="auto")
    return tokenizer, model


def update_generation_config(model, tokenizer):
    generation_config_updates = {
        "pad_token_id": tokenizer.eos_token_id,
        "return_dict_in_generate": True,
        "output_scores": True,
        "do_sample": False,
        "temperature": None,
        "top_p": None,
    }
    model.generation_config.__dict__.update(generation_config_updates)
    print(model.generation_config)


def check_environment():
    assert "HF_HOME" in os.environ, "HF_HOME is not set"
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        raise ValueError("No CUDA or MPS device available")


def prepare_data(sst2_train, sst2_test, config):
    context_samples = sst2_train.sample(config["n_context"], seed=config["seed"])
    # print the proportion of positive and negative samples
    # print(context_samples["label"].value_counts(normalize=True))
    target_samples = sst2_test.sample(config["n_target"], seed=config["seed"])
    target_messages = create_few_shot_sst2_messages(context_samples, target_samples)
    return context_samples, target_samples, target_messages


def build_dataloader(target_messages, batch_size):
    dataset = SST2Dataset(target_messages)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=lambda x: x)


def generate_predictions(loader, tokenizer, model, device):
    test_preds = []
    test_probs = []
    for batch_messages in tqdm(loader):
        batch_messages_str = tokenizer.apply_chat_template(
            batch_messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        batch_tokens = tokenizer.batch_encode_plus(batch_messages_str, return_tensors="pt", padding=True).to(device)
        outputs = model.generate(  # max_new_tokens was 10 initially
            **batch_tokens, max_new_tokens=1, output_scores=True, output_logits=True, renormalize_logits=True
        )
        # print(batch_messages_str)
        # print(outputs)
        # -3: includes the end-of-turn and end-of-sequence tokens
        preds = tokenizer.batch_decode(outputs["sequences"][:, -1:], skip_special_tokens=True)
        # if len(preds[0]) == 0:
        #     print(tokenizer.batch_decode(outputs["sequences"], skip_special_tokens=False))
        print(preds)
        # filter out predictions that are not 0 or 1, if it can be converted to an int
        # get indices of preds that can be converted to an int
        # valid_indices = [i for i, pred in enumerate(preds) if pred in ["0", "1"]]
        # convert preds to int, otherwise fill with nan
        preds = [int(pred) if pred in ["0", "1"] else np.nan for pred in preds]
        test_preds.extend(preds)
        probs = torch.softmax(outputs["logits"][0], dim=-1).max(dim=-1).values.cpu().numpy().tolist()
        test_probs.extend(probs)
    return np.array(test_preds), np.array(test_probs)


def save_results(test_preds, test_probs, target_samples, context_samples, config):
    test_labels = target_samples["label"].to_numpy()
    test_df = pd.DataFrame(
        {
            "target_idx": target_samples["idx"].to_numpy(),
            "sentence": target_samples["sentence"].to_numpy(),
            "pred": test_preds,
            "prob": test_probs,
            "true_label": test_labels,
            "context_idx": [context_samples["idx"].to_list()] * config["n_target"],
        }
    )
    test_df.attrs["config"] = config
    test_df.to_csv(
        f"results/sst2_context_{config['n_context']}_{config['model_id'].replace('/', '_')}.csv", index=False
    )
    test_df.to_pickle(f"results/sst2_context_{config['n_context']}_{config['model_id'].replace('/', '_')}.pkl")


def parse_args():
    parser = argparse.ArgumentParser(description="Run SST-2 classification.")
    parser.add_argument("--config_file", type=str, help="Path to the JSON config file.")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for data loading.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--n_context", type=int, default=100, help="Number of context samples.")
    parser.add_argument("--n_target", type=int, default=500, help="Number of target samples.")
    parser.add_argument("--model_id", type=str, default="meta-llama/Llama-3.3-1B-Instruct", help="Model ID.")
    return parser.parse_args()


def load_config(args):
    if args.config_file:
        with open(args.config_file, "r") as f:
            config = json.load(f)
    else:
        config = {
            "batch_size": args.batch_size,
            "seed": args.seed,
            "n_context": args.n_context,
            "n_target": args.n_target,
            "model_id": args.model_id,
        }
    return config


def main(config):
    print(f"Running with config: {config}")
    sst2_train, sst2_test = load_sst2_data()
    tokenizer, model = load_model(config["model_id"])
    update_generation_config(model, tokenizer)
    device = check_environment()
    # every test sample gets the same context
    context_samples, target_samples, target_messages = prepare_data(sst2_train, sst2_test, config)
    loader = build_dataloader(target_messages, config["batch_size"])
    test_preds, test_probs = generate_predictions(loader, tokenizer, model, device)
    save_results(test_preds, test_probs, target_samples, context_samples, config)


if __name__ == "__main__":
    args = parse_args()
    config = load_config(args)
    main(config)
