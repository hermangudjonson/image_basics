"""Initial GPU evaluation tasks.
"""
import json
import time
from pathlib import Path
from pprint import pprint

import fire
import torch
from torch.utils.benchmark import Compare, Timer

from image_basics import train, utils


def _save_json(data_dict: dict, json_file: Path):
    """Save dictionary to json (helper)."""
    json_file.parent.mkdir(parents=True, exist_ok=True)
    with json_file.open("w") as f:
        json.dump(data_dict, f, indent=4)


def easy_pets_recipe():
    hparams = {
        "model_params": {"name": "regnety"},
        "data_params": {
            "data_name": "pets",
            "target": "label_cat_dog",
            "model_name": "regnety",
            "train_subset": 512,
            "val_subset": 512,
            "train_batch_size": 32,
            "val_batch_size": 32,
        },
        "optimizer_params": {
            "lr": 1e-3,
        },
        "num_epochs": 1,
        "device": None,
    }
    return hparams


def batch_time(batch_sizes=None, subset_size=None, output_dir=None, device=None):
    """Compare matmul, forward pass and train() timing across batch sizes.

    Parameters
    ----------
    batch_sizes : list(int), optional
        list of batch sizes to compare, by default [2, 4]
    output_dir : str | Path, optional
        directory to save batch timing results, by default <working_dir>/results/gpu_init/
    device : torch device, optional
        device to use, by default gpu/cuda if available
    """
    batch_sizes = batch_sizes if batch_sizes is not None else [2, 4]
    output_dir = (
        Path(output_dir)
        if output_dir is not None
        else utils.WORKING_DIR / "results/gpu_init"
    )

    results = {}
    timer_list = []
    hparams = easy_pets_recipe()
    for b in batch_sizes:
        hparams["data_params"]["train_batch_size"] = b
        hparams["data_params"]["val_batch_size"] = b
        if subset_size is not None:
            hparams["data_params"]["train_subset"] = subset_size
            hparams["data_params"]["val_subset"] = subset_size
        hparams["device"] = device

        trainer = train.create_trainer(**hparams)

        device = utils.get_device(device)
        print(f"using device {device}")

        # matmul
        A = torch.randn(16, 3 * 244 * 244, device=device)  # ~3M params
        b = torch.randn(3 * 244 * 244, b, device=device)
        timer = Timer(
            "A.matmul(b)",
            globals={"A": A, "b": b},
            description=f"batch {b}",
            sub_label="matmul",
        )
        timer_result = timer.blocked_autorange(min_run_time=1)
        matmul_time = timer_result.median
        timer_list.append(timer_result)

        # forward pass
        X_rand = torch.randn(b, 3, 244, 244, device=device)
        trainer.model.to(device)
        timer = Timer(
            "trainer.model(X_rand)",
            globals={"trainer": trainer, "X_rand": X_rand},
            description=f"batch {b}",
            sub_label="forward pass",
        )
        timer_result = timer.blocked_autorange(min_run_time=1)
        forward_time = timer_result.median
        timer_list.append(timer_result)

        # trainer.train()
        if device.type == "gpu":
            torch.cuda.synchronize()
        train_start = time.time()
        trainer.train()
        if device.type == "gpu":
            torch.cuda.synchronize()
        train_time = time.time() - train_start
        print(f"trainer used device {trainer.device}")

        results[b] = dict(
            **{
                "matmul_time": matmul_time,
                "matmul_time_per_image": matmul_time / b,
                "matmul_images_per_s": b / matmul_time,
                "forward_time": forward_time,
                "forward_time_per_image": forward_time / b,
                "forward_images_per_s": b / forward_time,
                "train_time": train_time,
                "train_step_time": train_time
                / (trainer.num_epochs * len(trainer.train_dl)),
                "train_images_per_s": (trainer.num_epochs * len(trainer.train_dl) * b)
                / train_time,
            },
            **trainer.best_metrics,
        )

    # display and save results
    timer_compare = Compare(timer_list)
    timer_compare.print()
    pprint(results)
    _save_json(results, output_dir / "batch_timing.json")


if __name__ == "__main__":
    fire.Fire()
