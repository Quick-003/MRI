"""
Optimization and Learning Rate Scheduling for DenWide-Net.
Provides AdamW, SGD, and Cosine Annealing schedulers with warmup.
"""

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR


def get_optimizer_and_scheduler(
    model,
    optimizer_type="adamw",
    learning_rate=1e-4,
    weight_decay=1e-4,
    num_epochs=30
):
    """
    Initializes optimizer and learning rate scheduler for training DenWide-Net.
    """
    # Separate weight decay for BatchNorm / bias parameters
    decay_params = []
    no_decay_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "bn" in name or "bias" in name:
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    grouped_params = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0}
    ]

    if optimizer_type.lower() == "adamw":
        optimizer = optim.AdamW(grouped_params, lr=learning_rate, betas=(0.9, 0.999), eps=1e-8)
    elif optimizer_type.lower() == "sgd":
        optimizer = optim.SGD(grouped_params, lr=learning_rate, momentum=0.9, nesterov=True)
    else:
        optimizer = optim.Adam(grouped_params, lr=learning_rate)

    # Cosine Annealing Scheduler with minimum learning rate
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=learning_rate * 0.01)

    return optimizer, scheduler
