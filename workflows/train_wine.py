import pandas as pd
from flytekit import task, workflow
from flytekit.types.pickle import FlytePickle
from sklearn.datasets import load_wine
from sklearn.linear_model import LogisticRegression


@task
def get_data() -> pd.DataFrame:
    """Get the wine dataset."""
    return load_wine(as_frame=True).frame


@task
def process_data(data: pd.DataFrame) -> pd.DataFrame:
    """Simplify the task from a 3-class to a binary classification problem."""
    return data.assign(target=lambda x: x["target"].where(x["target"] == 0, 1))


@task
def train_model(data: pd.DataFrame) -> FlytePickle:
    """Train a model on the wine dataset."""
    features = data.drop("target", axis="columns")
    target = data["target"]
    return LogisticRegression(max_iter=1000).fit(features, target)


@workflow
def training_workflow() -> FlytePickle:
    """Put all of the steps together into a single workflow."""
    data = get_data()
    processed_data = process_data(data=data)
    return train_model(data=processed_data)


if __name__ == "__main__":
    print(f"Running training_workflow() {training_workflow()}")
