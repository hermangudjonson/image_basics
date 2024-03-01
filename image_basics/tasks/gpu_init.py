"""Initial GPU evaluation tasks."""

import json
import time
from pathlib import Path
from pprint import pprint

import fire
import torch
from torch.utils.benchmark import Compare, Timer

from image_basics import train, utils


def _get_output_dir(output_dir=None):
    """Resolve output dir for experiments in this module."""
    output_dir = (
        Path(output_dir)
        if output_dir is not None
        else utils.WORKING_DIR / "results/gpu_init"
    )
    return output_dir


def _save_json(data_dict: dict, json_file: Path):
    """Save dictionary to json (helper)."""
    json_file.parent.mkdir(parents=True, exist_ok=True)
    with json_file.open("w") as f:
        json.dump(data_dict, f, indent=4)


def easy_pets_recipe(subset_size=512, num_epochs=1, device=None):
    hparams = {
        "model_params": {"name": "regnety"},
        "data_params": {
            "data_name": "pets",
            "target": "label_cat_dog",
            "model_name": "regnety",
            "train_subset": subset_size,
            "val_subset": subset_size,
            "train_batch_size": 32,
            "val_batch_size": 32,
        },
        "optimizer_params": {
            "lr": 1e-3,
        },
        "num_epochs": num_epochs,
        "device": device,
    }
    return hparams


def batch_time(
    batch_sizes=None, subset_size=512, num_epochs=1, output_dir=None, device=None
):
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
    output_dir = _get_output_dir(output_dir)

    results = {}
    timer_list = []
    hparams = easy_pets_recipe(
        subset_size=subset_size, num_epochs=num_epochs, device=device
    )
    for b in batch_sizes:
        hparams["data_params"]["train_batch_size"] = b
        hparams["data_params"]["val_batch_size"] = b

        trainer = train.create_trainer(**hparams)

        device = utils.get_device(device)
        print(f"using device {device}")

        # matmul
        A = torch.randn(16, 3 * 244 * 244, device=device)  # ~3M params
        v = torch.randn(3 * 244 * 244, b, device=device)
        timer = Timer(
            "A.matmul(v)",
            globals={"A": A, "v": v},
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


def matmul_time(model_sizes=None, batch_sizes=None, output_dir=None, device=None):
    """Sweep idealized 'model' sizes across different batch sizes.

    the idealized model proxy is a simple matrix multiplication, standard image input dimensions (flattened).
    our objective is to see where optimal GPU efficiency saturates in ideal conditions
    """
    model_sizes = (
        model_sizes if model_sizes is not None else [3e6, 6e6, 12e6, 24e6, 48e6]
    )  # 200MF to ~10GF regime
    batch_sizes = (
        batch_sizes
        if batch_sizes is not None
        else [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
    )
    output_dir = _get_output_dir(output_dir)

    device = utils.get_device(device)
    print(f"using device {device}")

    results = []
    timer_list = []

    for nparams in model_sizes:
        for b in batch_sizes:
            # matmul
            image_size = 3 * 244 * 244
            A = torch.randn(int(nparams / image_size), image_size, device=device)
            v = torch.randn(image_size, b, device=device)
            timer = Timer(
                "A.matmul(v)",
                globals={"A": A, "v": v},
                description=f"batch {b}",
                sub_label=f"model size {nparams}",
            )
            timer_result = timer.blocked_autorange(min_run_time=1)
            matmul_time = timer_result.median
            timer_list.append(timer_result)

            results.append(
                {
                    "model_size": nparams,
                    "batch_size": b,
                    "matmul_time": matmul_time,
                    "matmul_time_per_image": matmul_time / b,
                    "matmul_images_per_s": b / matmul_time,
                }
            )

    # display and save results
    timer_compare = Compare(timer_list)
    timer_compare.print()
    pprint(results)
    _save_json(results, output_dir / "matmul_timing.json")


def dl_time(
    batch_sizes=None, subset_size=1024, num_epochs=2, output_dir=None, device=None
):
    # sweep dataloader num threads (and pin_memory) against batch size to see if affects GPU efficiency
    def iter_dataloader(dl, device):
        """Iterate through dataloader, loading to devie."""
        for i, batch in enumerate(dl):
            X, y = batch["image"], batch["target"]
            X, y = X.to(device), y.to(device)

    batch_sizes = (
        batch_sizes if batch_sizes is not None else [2, 4, 8, 16, 32, 64, 128, 256, 512]
    )
    output_dir = _get_output_dir(output_dir)

    cpu_nthreads = torch.get_num_threads()
    print(f"detected {cpu_nthreads} threads")

    device = utils.get_device(device)
    print(f"using device {device}")

    results = []
    timer_list = []
    hparams = easy_pets_recipe(
        subset_size=subset_size, num_epochs=num_epochs, device=device
    )
    for pin_memory in [False, True]:
        for num_proc in range(0, cpu_nthreads + 1):
            for b in batch_sizes:
                hparams["data_params"]["train_batch_size"] = b
                hparams["data_params"]["val_batch_size"] = b
                hparams["data_params"]["num_proc"] = num_proc
                hparams["data_params"]["pin_memory"] = pin_memory

                trainer = train.create_trainer(**hparams)

                # forward pass
                timer = Timer(
                    "iter_dataloader(trainer.train_dl, trainer.device)",
                    globals={"iter_dataloader": iter_dataloader, "trainer": trainer},
                    description=f"batch {b}",
                    label=f"pin memory {pin_memory}",
                    sub_label=f"num threads {num_proc}",
                    num_threads=max(num_proc, 1),
                )
                timer_result = timer.blocked_autorange(min_run_time=1)
                iterdl_time = timer_result.median
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

                results.append(
                    dict(
                        **{
                            "pin_memory": pin_memory,
                            "num_proc": num_proc,
                            "batch_size": b,
                            "iterdl_time": iterdl_time,
                            "iterdl_time_per_batch": iterdl_time
                            / len(trainer.train_dl),
                            "iterdl_time_per_image": iterdl_time
                            / (b * len(trainer.train_dl)),
                            "iterdl_images_per_s": (b * len(trainer.train_dl))
                            / iterdl_time,
                            "train_time": train_time,
                            "train_step_time": train_time
                            / (trainer.num_epochs * len(trainer.train_dl)),
                            "train_images_per_s": (
                                trainer.num_epochs * len(trainer.train_dl) * b
                            )
                            / train_time,
                        },
                        **trainer.best_metrics,
                    )
                )

    # display and save results
    timer_compare = Compare(timer_list)
    timer_compare.print()
    pprint(results)
    _save_json(results, output_dir / "dataloader_timing.json")


if __name__ == "__main__":
    fire.Fire()
