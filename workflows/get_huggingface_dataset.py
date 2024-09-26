import pandas as pd
from flytekit import ImageSpec, task, Deck
from flytekitplugins.deck.renderer import ImageRenderer

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
def get_huggingface_dataset(name: str) -> pd.DataFrame:

    from datasets import load_dataset
    ds = load_dataset(name)
    
    Deck("Images", ImageRenderer().to_html(image_src=ds['train'][0]['image']))
    
    return pd.DataFrame(ds['train'])
