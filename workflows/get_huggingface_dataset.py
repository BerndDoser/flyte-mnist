import pandas as pd
from datasets import load_dataset
from flytekit import Deck, ImageSpec, current_context, task, workflow
from flytekitplugins.deck.renderer import ImageRenderer

image = ImageSpec(
    packages=[
        "datasets==3.0.0",
        "flytekitplugins-deck-standard==1.13.5",
        "pillow==10.4.0",
    ],
    python_version="3.12",
    registry="registry.h-its.org/doserbd/flyte",
)


@task(
    cache=True,
    cache_version="3",
    enable_deck=True,
    container_image=image,
)
def get_huggingface_dataset(name: str) -> pd.DataFrame:

    ctx = current_context()

    dataset = load_dataset(name, split="train")

    deck = Deck("Images", ImageRenderer().to_html(image_src=dataset[0]["image"]))
    ctx.decks.insert(0, deck)

    return dataset.to_pandas()


@workflow
def wf(name: str) -> pd.DataFrame:

    return get_huggingface_dataset(name=name)


if __name__ == "__main__":
    print(f"Running workflow() {wf()}")
