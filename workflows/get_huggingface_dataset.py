import pandas as pd
from datasets import load_dataset
from flytekit import Deck, ImageSpec, task
from flytekitplugins.deck.renderer import ImageRenderer


@task(
    cache=True,
    cache_version="1",
    enable_deck=True,
    container_image=ImageSpec(
        packages=[
            "datasets==3.0.0",
            "flytekitplugins-deck-standard==1.13.5",
            "pillow==10.4.0",
        ],
        python_version="3.12",
        registry="registry.h-its.org/doserbd/flyte",
    ),
)
def get_huggingface_dataset(name: str) -> pd.DataFrame:

    dataset = load_dataset(name, split="train")

    Deck("Images", ImageRenderer().to_html(image_src=dataset[0]["image"]))

    return dataset.to_pandas()
