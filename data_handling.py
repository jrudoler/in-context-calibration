import polars as pl
from copy import deepcopy
from torch.utils.data import Dataset


def create_few_shot_sst2_messages(df_context: pl.DataFrame, df_target: pl.DataFrame) -> list[list[dict]]:
    """
    Create few-shot prompt messages for the SST-2 dataset, with shared context but different target sentences.
    """
    messages = []
    # add system message -- gemma doesn't support system messages so send as an assistant message
    messages.append({"role": "assistant", "content": "I will classify the given text with label 0 or 1."})
    for row in df_context.iter_rows(named=True):
        messages.append({"role": "user", "content": f"{row['sentence']}"})
        messages.append({"role": "assistant", "content": f"{row['label']}"})
    # create a copy of the messages for each target sentence
    target_messages = [deepcopy(messages) for _ in range(len(df_target))]
    for i, row in enumerate(df_target.iter_rows(named=True)):
        target_messages[i].append({"role": "user", "content": f"{row['sentence']}"})
    return target_messages


class SST2Dataset(Dataset):
    def __init__(self, messages: list[list[dict]]):
        self.messages = messages

    def __len__(self):
        return len(self.messages)

    def __getitem__(self, idx):
        return self.messages[idx]
