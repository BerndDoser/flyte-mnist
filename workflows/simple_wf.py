from flytekit import task, workflow


@task
def hello_world(name: str) -> str:
    return f"Hello {name}"


@workflow
def main(name: str) -> str:
    return hello_world(name=name)
