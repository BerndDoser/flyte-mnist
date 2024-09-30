import pandas as pd
from datasets import load_dataset
from flytekit import Deck, ImageSpec, current_context, task, workflow
from flytekit.deck.renderer import MarkdownRenderer
from flytekitplugins.deck.renderer import ImageRenderer

image = ImageSpec(
    packages=[
        "datasets",
        "flytekitplugins-deck-standard",
        "markdown",
        "pandas",
        "pillow",
    ],
    python_version="3.12",
    registry="registry.h-its.org/doserbd/flyte",
)


@task(
    cache=True,
    cache_version="6",
    enable_deck=True,
    container_image=image,
)
def get_huggingface_dataset(name: str) -> pd.DataFrame:
    """
    Get a dataset from the Huggingface datasets library.

    Args:
        name (str): The name of the dataset to load.

    Returns:
        pd.DataFrame: The dataset as a pandas DataFrame.
    """

    ctx = current_context()

    dataset = load_dataset(name, split="train")
    ctx.default_deck.append(MarkdownRenderer().to_html("hey1"))
    deck = Deck("Images", ImageRenderer().to_html(dataset[0]["image"]))
    deck.append(MarkdownRenderer().to_html("hey2"))
    ctx.decks.insert(0, deck)

    return dataset.to_pandas()


@task(container_image=image)
def train_model(dataset: pd.DataFrame) -> float:
    """
    Get a dataset from the Huggingface datasets library.

    Args:
        dataset (pd.DataFrame): The dataset to train the model on

    Returns:
        loss (float): The loss of the model.
    """

    loss = 0.1
    return loss


@workflow
def wf(name: str) -> float:
    """
    ML classification workflow.

    Args:
        name (str):  The name of the dataset

    Returns:
        loss (float): The loss of the model.
    """
    dataset = get_huggingface_dataset(name=name)
    loss = train_model(dataset)
    return loss


if __name__ == "__main__":
    print(f"Running workflow() {wf()}")
