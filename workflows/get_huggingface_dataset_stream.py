import pandas as pd
from flytekit import ImageSpec, task, Deck
from flytekitplugins.deck.renderer import ImageRenderer
from torch.utils.data import DataLoader
from datasets import load_dataset

@task(
    cache=True,
    cache_version="1",
    enable_deck=True,
    container_image=ImageSpec(
        packages=[
            "datasets==3.0.0"
        ],
        python_version="3.12",
        registry="registry.h-its.org/doserbd/flyte",
    ),
)
def get_huggingface_dataset(name: str) -> DataLoader:

    dataset = load_dataset(name, streaming=True, split="train")
    
    # Deck("Images", ImageRenderer().to_html(image_src=dataset[0]['image']))
    
    return DataLoader(dataset, batch_size=32)
