"""
Image Data loading and preprocessing

things to look out for:
 - control data package cacheing
 - provide specific target according to task
"""
import timm
import datasets
from pathlib import Path
import functools
import torch
from torch.utils.data import DataLoader

from image_basics import model
from image_basics.model import _DEFAULT_MODEL

_DEFAULT_DATA = "pets"
_DEFAULT_TARGET = "label_cat_dog"


def _get_data_name(name: str = _DEFAULT_DATA):
    """Retrieve official data name based on simplified tag."""
    if name == "pets":
        return "timm/oxford-iiit-pet"
    else:
        raise ValueError(f"{name} is not a valid data str")


def _unnorm_images(images, model_name: str = _DEFAULT_MODEL):
    image_list = images if isinstance(images, list) else [images]

    model_config = model._get_model_config(model_name)
    # these are tuples
    mean, std = model_config["mean"], model_config["std"]
    mean, std = torch.as_tensor(mean), torch.as_tensor(std)
    mean, std = mean.view(-1, 1, 1), std.view(-1, 1, 1)
    unnormed = [img.mul(std).add(mean) for img in image_list]
    return unnormed if isinstance(images, list) else unnormed[0]


def _get_model_transforms(
    name: str = _DEFAULT_MODEL, use_test_size: bool = False, **kwargs
):
    f"""Generate torchvision transforms based on model config.

    Parameters
    ----------
    name : str, optional
        model tag name, by default {_DEFAULT_MODEL}
    use_test_size : bool, optional
        determines input size, passed to `resolve_data_config`, by default False
    kwargs are passed to `create_transform` and can override model config.

    Returns
    -------
    torchvision transforms
    """
    model_config = model._get_model_config(name)
    data_config = timm.data.resolve_data_config(
        pretrained_cfg=model_config, use_test_size=use_test_size
    )
    data_config.update(kwargs)
    transforms = timm.data.create_transform(**data_config)
    return transforms


def _apply_transforms(examples, transforms):
    # helper apply transforms to batch of images
    # there are some RGBA lurking, using PIL convert
    examples["image"] = [
        transforms((image.convert("RGB") if image.mode != "RGB" else image))
        for image in examples["image"]
    ]
    return examples


def create_dataset(
    data_name: str = _DEFAULT_DATA,
    target: str = _DEFAULT_TARGET,
    model_name: str = _DEFAULT_MODEL,
    use_test_size: bool = False,
    cache_dir: str | Path | None = None,
    num_proc: int = 4,
):
    """Create DataSet for given task (and split).

    returns train_ds, test_ds
    standardizes data feature names to "image" and "target"
    """
    # default save to $HF_HOME set in utils
    hfds = datasets.load_dataset(
        _get_data_name(data_name), cache_dir=cache_dir, num_proc=num_proc
    )
    # separate train and test transforms
    train_transforms = _get_model_transforms(
        model_name, use_test_size=False, is_training=True
    )
    test_transforms = _get_model_transforms(
        model_name, use_test_size=use_test_size, is_training=False
    )

    train_processed = (
        hfds["train"]
        .select_columns(["image", target])
        .rename_columns({target: "target"})
        .with_format("torch")
        .with_transform(
            functools.partial(_apply_transforms, transforms=train_transforms)
        )
    )
    test_processed = (
        hfds["test"]
        .select_columns(["image", target])
        .rename_columns({target: "target"})
        .with_format("torch")
        .with_transform(
            functools.partial(_apply_transforms, transforms=test_transforms)
        )
    )
    return train_processed, test_processed


def create_dataloader(
    dataset: datasets.Dataset,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 4,
):
    """Create pytorch Dataloader from a Dataset.

    default drop last batch if smaller
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=True,
    )


def get_feature_data():
    pass
