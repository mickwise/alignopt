from typing import List, Tuple, Dict, Iterable, Literal
from math import ceil
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler, AdamW, Optimizer
import numpy as np

from alignopt.data.loaders import load_and_split_data, batch_and_collect_data, DataBatch
from alignopt.data.data_config import RANDOM_SEED
from alignopt.backends.backends_config import BATCH_SIZE
from alignopt.backends.torch.dpo.logprobs_torch import completion_logprobs_torch
from alignopt.backends.torch.dpo.dpo_loss_torch import dpo_loss_torch
from alignopt.backends.torch.dpo.dpo_validation_torch import validate_training_knobs
from alignopt.backends.torch.dpo.torch_dpo_types import PreferenceDataset


def train_dpo_torch(
    data_path_str: str,
    reference_model: AutoModelForCausalLM,
    *,
    initial_policy_model: AutoModelForCausalLM | None = None,
    provided_tokenizer: AutoTokenizer | None = None,
    provided_device: torch.device | str | None = None,
    provided_optimizer: Optimizer | None = None,
    provided_scheduler: lr_scheduler.LRScheduler | None = None,
    scheduler_step: Literal["batch", "epoch"] = "batch",
    shuffle_train: bool = True,
    drop_last: bool = False,
    num_workers: int = 0,
    pin_memory: bool | None = None,
    persistent_workers: bool | None = None,
    beta: float = 0.1,
    training_fraction: float = 0.8,
    random_seed: int = RANDOM_SEED,
    batch_size: int = BATCH_SIZE,
    max_length: int | None = None,
    lr: float = 1e-3,
    betas: Tuple[float, float] = (0.9, 0.999),
    eps: float = 1e-8,
    weight_decay: float = 1e-2,
    num_epochs: int = 1,
    steps_to_metrics: int | None = None,
) -> None:
    validate_training_knobs(
        batch_size,
        max_length,
        lr,
        eps,
        weight_decay,
        num_epochs,
        steps_to_metrics,
    )

    # Enforce your scheduler contract *now* (don’t defer this).
    # If you let users pick scheduler_step="epoch" while you still construct a per-batch scheduler,
    # you are guaranteeing incorrect behavior.
    if provided_scheduler is None and scheduler_step != "batch":
        raise ValueError("Default scheduler is stepped per-batch; provide a scheduler if scheduler_step='epoch'.")

    policy_model, tokenizer, current_device = _initialize_model_tokenizer_and_device(
        reference_model,
        initial_policy_model,
        provided_tokenizer,
        provided_device,
    )

    # Resolve DataLoader defaults into actual booleans (professional contract)
    effective_pin_memory: bool = pin_memory if pin_memory is not None else (current_device.type == "cuda")
    if persistent_workers is None:
        effective_persistent_workers: bool = num_workers > 0
    else:
        effective_persistent_workers = persistent_workers
    if effective_persistent_workers and num_workers == 0:
        raise ValueError("persistent_workers=True requires num_workers > 0.")

    training_loader, evaluation_loader = _get_data(
        data_path_str=data_path_str,
        training_fraction=training_fraction,
        random_seed=random_seed,
        batch_size=batch_size,
        shuffle_train=shuffle_train,
        drop_last=drop_last,
        num_workers=num_workers,
        pin_memory=effective_pin_memory,
        persistent_workers=effective_persistent_workers,
    )

    optimizer, scheduler = _initialize_optimizer_and_scheduler(
        provided_optimizer=provided_optimizer,
        provided_scheduler=provided_scheduler,
        params=policy_model.parameters(),
        steps_per_epoch=len(training_loader),
        num_epochs=num_epochs,
        scheduler_step=scheduler_step,
        lr=lr,
        betas=betas,
        eps=eps,
        weight_decay=weight_decay,
    )

    current_metrics: Dict[str, float] = {}
    for _ in range(num_epochs):
        reference_model.eval()
        policy_model.train()
        _zero_metric_dict(current_metrics)
        _training_loop(
            reference_model=reference_model,
            policy_model=policy_model,
            tokenizer=tokenizer,
            optimizer=optimizer,
            training_loader=training_loader,
            current_metrics=current_metrics,
            scheduler=scheduler,
            scheduler_step=scheduler_step,
            beta=beta,
            max_length=max_length,
            steps_to_metrics=steps_to_metrics,
        )
        if scheduler_step == "epoch":
            scheduler.step()
        # TODO: display
        # TODO: evaluation


def _get_data(
    data_path_str: str,
    training_fraction: float,
    random_seed: int,
    batch_size: int,
    shuffle_train: bool,
    drop_last: bool,
    num_workers: int,
    pin_memory: bool,
    persistent_workers: bool,
) -> Tuple[DataLoader, DataLoader]:
    rng: torch.Generator = torch.Generator()
    rng.manual_seed(random_seed)

    train_list, eval_list = load_and_split_data(data_path_str, training_fraction, random_seed)
    training_data = PreferenceDataset(train_list)
    evaluation_data = PreferenceDataset(eval_list)

    training_loader: DataLoader = DataLoader(
        training_data,
        batch_size=batch_size,
        shuffle=shuffle_train,
        generator=rng if shuffle_train else None,
        drop_last=drop_last,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        collate_fn=batch_and_collect_data,
    )

    evaluation_loader: DataLoader = DataLoader(
        evaluation_data,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,  # evaluation typically should not drop data
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        collate_fn=batch_and_collect_data,
    )
    return training_loader, evaluation_loader


def _initialize_model_tokenizer_and_device(
    reference_model: AutoModelForCausalLM,
    initial_policy_model: AutoModelForCausalLM | None,
    provided_tokenizer: AutoTokenizer | None,
    provided_device: torch.device | str | None,
) -> Tuple[AutoModelForCausalLM, AutoTokenizer, torch.device]:
    policy_model: AutoModelForCausalLM = _initialize_policy_model(reference_model, initial_policy_model)

    current_device: torch.device = (
        torch.device(provided_device)
        if provided_device is not None
        else next(reference_model.parameters()).device
    )

    reference_model.to(current_device)
    policy_model.to(current_device)

    tokenizer: AutoTokenizer = (
        provided_tokenizer
        if provided_tokenizer is not None
        else AutoTokenizer.from_pretrained(reference_model.name_or_path, use_fast=True)
    )
    return policy_model, tokenizer, current_device


def _initialize_policy_model(
    reference_model: AutoModelForCausalLM,
    initial_policy_model: AutoModelForCausalLM | None = None,
) -> AutoModelForCausalLM:
    policy_model: AutoModelForCausalLM = (
        initial_policy_model
        if initial_policy_model is not None
        else AutoModelForCausalLM.from_pretrained(reference_model.name_or_path)
    )
    for param in reference_model.parameters():
        param.requires_grad_(False)
    for param in policy_model.parameters():
        param.requires_grad_(True)
    return policy_model


def _initialize_optimizer_and_scheduler(
    provided_optimizer: Optimizer | None,
    provided_scheduler: lr_scheduler.LRScheduler | None,
    params: Iterable[torch.nn.Parameter],
    steps_per_epoch: int,
    num_epochs: int,
    scheduler_step: Literal["batch", "epoch"],
    lr: float,
    betas: Tuple[float, float],
    eps: float,
    weight_decay: float,
) -> Tuple[Optimizer, lr_scheduler.LRScheduler]:
    optimizer: Optimizer = provided_optimizer
    if optimizer is None:
        optimizer = AdamW(
            params=params,
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
        )

    scheduler: lr_scheduler.LRScheduler = provided_scheduler
    if scheduler is None:
        if scheduler_step == "batch":
            t_max: int = steps_per_epoch * num_epochs
        else:
            t_max = num_epochs
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, t_max)

    return optimizer, scheduler


def _zero_metric_dict(current_metrics: Dict[str, float]) -> None:
    current_metrics["count"] = 0
    current_metrics["log_diff_mean"] = 0.0
    current_metrics["log_diff_std"] = 0.0
    current_metrics["log_diff_max"] = -np.inf
    current_metrics["log_diff_min"] = np.inf
    current_metrics["log_diff_frac_pos"] = 0.0
    current_metrics["pi_margin_mean"] = 0.0
    current_metrics["reference_margin_mean"] = 0.0


def _training_loop(
    reference_model: AutoModelForCausalLM,
    policy_model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    optimizer: Optimizer,
    training_loader: DataLoader,
    current_metrics: Dict[str, float],
    scheduler: lr_scheduler.LRScheduler,
    scheduler_step: Literal["batch", "epoch"],
    beta: float,
    max_length: int | None,
    steps_to_metrics: int | None,
) -> Dict[str, float]:
    for step_idx, batch in enumerate(training_loader, start=1):
        optimizer.zero_grad()

        current_mean_loss, new_metrics = _calculate_loss(
            current_batch=batch,
            reference_model=reference_model,
            policy_model=policy_model,
            tokenizer=tokenizer,
            max_length=max_length,
            beta=beta,
        )

        batch_n: int = len(batch.preference_ids)
        _update_current_metrics(current_metrics, new_metrics, batch_n)

        current_mean_loss.backward()
        optimizer.step()

        if scheduler_step == "batch":
            scheduler.step()

        if steps_to_metrics is not None and step_idx % steps_to_metrics == 0:
            # TODO: display metrics snapshot for current_metrics
            pass

    return current_metrics


def _calculate_loss(
    current_batch: DataBatch,
    reference_model: AutoModelForCausalLM,
    policy_model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    max_length: int | None,
    beta: float,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    pi_chosen: torch.Tensor = completion_logprobs_torch(
        policy_model, tokenizer, current_batch.chosen, max_length
    )
    pi_rejected: torch.Tensor = completion_logprobs_torch(
        policy_model, tokenizer, current_batch.rejected, max_length
    )
    with torch.no_grad():
        reference_chosen: torch.Tensor = completion_logprobs_torch(
            reference_model, tokenizer, current_batch.chosen, max_length
        )
        reference_rejected: torch.Tensor = completion_logprobs_torch(
            reference_model, tokenizer, current_batch.rejected, max_length
        )

    current_mean_loss, new_metrics = dpo_loss_torch(
        pi_chosen,
        pi_rejected,
        reference_chosen,
        reference_rejected,
        beta,
    )
    return current_mean_loss, new_metrics


def _update_current_metrics(
    current_metrics: Dict[str, float],
    new_metrics: Dict[str, float],
    batch_size: int,
) -> None:
    old_count: int = current_metrics["count"]
    old_mean: float = current_metrics["log_diff_mean"]
    new_mean: float = new_metrics["log_diff_mean"]

    old_ss: float = (current_metrics["log_diff_std"] ** 2) * (old_count - 1) if old_count > 1 else 0.0
    new_ss: float = (new_metrics["log_diff_std"] ** 2) * (batch_size - 1) if batch_size > 1 else 0.0

    old_frac_pos: float = current_metrics["log_diff_frac_pos"]
    new_frac_pos: float = new_metrics["log_diff_frac_pos"]

    old_pi_mean: float = current_metrics["pi_margin_mean"]
    new_pi_mean: float = new_metrics["pi_margin_mean"]

    old_reference_mean: float = current_metrics["reference_margin_mean"]
    new_reference_mean: float = new_metrics["reference_margin_mean"]

    current_metrics["count"] += batch_size
    current_metrics["log_diff_mean"] = _update_mean(old_mean, old_count, new_mean, batch_size)
    current_metrics["log_diff_std"] = np.sqrt(
        _update_variance(old_mean, old_ss, old_count, new_mean, new_ss, batch_size)
    )
    current_metrics["log_diff_max"] = max(current_metrics["log_diff_max"], new_metrics["log_diff_max"])
    current_metrics["log_diff_min"] = min(current_metrics["log_diff_min"], new_metrics["log_diff_min"])
    current_metrics["log_diff_frac_pos"] = _update_mean(old_frac_pos, old_count, new_frac_pos, batch_size)
    current_metrics["pi_margin_mean"] = _update_mean(old_pi_mean, old_count, new_pi_mean, batch_size)
    current_metrics["reference_margin_mean"] = _update_mean(old_reference_mean, old_count, new_reference_mean, batch_size)


def _update_variance(
    old_mean: float,
    old_ss: float,
    old_count: int,
    new_mean: float,
    new_ss: float,
    batch_size: int,
) -> float:
    updated_count: int = old_count + batch_size
    if updated_count <= 1:
        return 0.0
    mean_diff: float = new_mean - old_mean
    updated_ss: float = old_ss + new_ss + (mean_diff ** 2) * old_count * batch_size / updated_count
    return updated_ss / (updated_count - 1)


def _update_mean(old_mean: float, old_count: int, new_mean: float, batch_size: int) -> float:
    return (old_count * old_mean + batch_size * new_mean) / (old_count + batch_size)
