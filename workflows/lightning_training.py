import logging
import os

import lightning as L
from flytekit import ImageSpec, PodTemplate, Resources, task, workflow
from flytekit.extras.accelerators import T4
from flytekit.types.directory import FlyteDirectory
from flytekitplugins.kfpytorch.task import Elastic
from kubernetes.client.models import (
    V1Container,
    V1EmptyDirVolumeSource,
    V1PodSpec,
    V1Volume,
    V1VolumeMount,
)
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor

custom_image = ImageSpec(
    packages=[
        "torch",
        "torchvision",
        "flytekitplugins-kfpytorch",
        "kubernetes",
        "lightning",
        "wandb",
    ],
    python_version="3.10",
    registry="registry.h-its.org/doserbd/flyte",
)


container = V1Container(
    name=custom_image.name,
    volume_mounts=[V1VolumeMount(mount_path="/dev/shm", name="dshm")],
)
volume = V1Volume(name="dshm", empty_dir=V1EmptyDirVolumeSource(medium="Memory"))
custom_pod_template = PodTemplate(
    primary_container_name=custom_image.name,
    pod_spec=V1PodSpec(
        runtime_class_name="nvidia", containers=[container], volumes=[volume]
    ),
)


class MNISTAutoEncoder(L.LightningModule):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def training_step(self, batch, batch_idx):
        x, y = batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = nn.functional.mse_loss(x_hat, x)
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


class MNISTDataModule(L.LightningDataModule):
    def __init__(self, root_dir, batch_size=64, dataloader_num_workers=0):
        super().__init__()
        self.root_dir = root_dir
        self.batch_size = batch_size
        self.dataloader_num_workers = dataloader_num_workers

    def prepare_data(self):
        MNIST(self.root_dir, train=True, download=True)

    def setup(self, stage=None):
        self.dataset = MNIST(
            self.root_dir,
            train=True,
            download=False,
            transform=ToTensor(),
        )

    def train_dataloader(self):
        persistent_workers = self.dataloader_num_workers > 0
        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            num_workers=self.dataloader_num_workers,
            persistent_workers=persistent_workers,
            pin_memory=True,
            shuffle=True,
        )


NUM_NODES = 1  # 2
NUM_DEVICES = 1  # 8


@task(
    container_image=custom_image,
    # task_config=Elastic(
    #     nnodes=NUM_NODES,
    #     nproc_per_node=NUM_DEVICES,
    #     rdzv_configs={"timeout": 36000, "join_timeout": 36000},
    #     max_restarts=3,
    # ),
    # accelerator=T4,
    requests=Resources(mem="32Gi", cpu="48", gpu="1", ephemeral_storage="100Gi"),
    pod_template=custom_pod_template,
)
def train_model(dataloader_num_workers: int) -> FlyteDirectory:
    """Train an autoencoder model on the MNIST."""

    encoder = nn.Sequential(nn.Linear(28 * 28, 64), nn.ReLU(), nn.Linear(64, 3))
    decoder = nn.Sequential(nn.Linear(3, 64), nn.ReLU(), nn.Linear(64, 28 * 28))
    autoencoder = MNISTAutoEncoder(encoder, decoder)

    root_dir = os.getcwd()
    data = MNISTDataModule(
        root_dir,
        batch_size=4,
        dataloader_num_workers=dataloader_num_workers,
    )

    logging.info(
        f"Training the model with {dataloader_num_workers} dataloader workers."
    )

    # while True:
    #     time.sleep(10)

    model_dir = os.path.join(root_dir, "model")
    trainer = L.Trainer(
        default_root_dir=model_dir,
        max_epochs=1,
        num_nodes=NUM_NODES,
        devices=NUM_DEVICES,
        accelerator="gpu",
        strategy="ddp",
        precision="16-mixed",
    )
    trainer.fit(model=autoencoder, datamodule=data)
    return FlyteDirectory(path=str(model_dir))


@workflow
def wf(dataloader_num_workers: int = 1) -> FlyteDirectory:
    return train_model(dataloader_num_workers=dataloader_num_workers)


if __name__ == "__main__":
    print(f"Running workflow() {wf()}")
