# Copyright 2026 Addis Pulse Studio. Licensed under the Apache License, Version 2.0.
"""A ``hyperflow`` entry in ComfyUI's scheduler registry.

Nodes that take a scheduler *name* instead of a SIGMAS socket -- KSampler, BasicScheduler, and packs that sample
internally through them, such as Pulse Studio's Pulse Render -- can then select the HyperFlow grid by name. The grid is
the built-in 8-step one shifted by the model's own video shift, the same values ``HyperFlow Sigmas`` produces for the
v1.0 weights; the diffusion-model wrapper still checks every run against the grid in the loaded file.
"""

from __future__ import annotations

import logging

from .schedule import DEFAULT_SIGMAS_8STEP, shift_sigmas

logger = logging.getLogger("HyperFlow")

SCHEDULER_NAME = "hyperflow"
STEPS = len(DEFAULT_SIGMAS_8STEP) - 1


def hyperflow_scheduler(model_sampling, steps):
    import torch

    if int(steps) != STEPS:
        raise ValueError(
            f"The 'hyperflow' scheduler is HyperFlow's fixed {STEPS}-step grid; set steps to {STEPS} "
            f"(got {int(steps)})."
        )
    return torch.tensor(shift_sigmas(DEFAULT_SIGMAS_8STEP, float(model_sampling.shift)), dtype=torch.float32)


def register() -> bool:
    """Add ``hyperflow`` to ``comfy.samplers`` once. Returns whether this call added it."""
    import comfy.samplers as samplers

    if SCHEDULER_NAME in samplers.SCHEDULER_HANDLERS:
        return False
    samplers.SCHEDULER_HANDLERS[SCHEDULER_NAME] = samplers.SchedulerHandler(hyperflow_scheduler, use_ms=True)
    # KSampler.SCHEDULERS is the same list object, so every scheduler combo picks the name up.
    if SCHEDULER_NAME not in samplers.SCHEDULER_NAMES:
        samplers.SCHEDULER_NAMES.append(SCHEDULER_NAME)
    return True
