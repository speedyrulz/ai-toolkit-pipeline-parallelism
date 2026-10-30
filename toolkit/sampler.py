import copy
import math

import numpy as np
from diffusers.utils import logging as diffusers_logging
from diffusers import (
    DDPMScheduler,
    EulerAncestralDiscreteScheduler,
    DPMSolverMultistepScheduler,
    DPMSolverSinglestepScheduler,
    LMSDiscreteScheduler,
    PNDMScheduler,
    DDIMScheduler,
    EulerDiscreteScheduler,
    HeunDiscreteScheduler,
    KDPM2DiscreteScheduler,
    KDPM2AncestralDiscreteScheduler,
    LCMScheduler,
    FlowMatchEulerDiscreteScheduler,
)
from toolkit.samplers.mean_flow_scheduler import MeanFlowScheduler

from toolkit.samplers.custom_flowmatch_sampler import CustomFlowMatchEulerDiscreteScheduler

from toolkit.samplers.custom_lcm_scheduler import CustomLCMScheduler

# scheduler:
SCHEDULER_LINEAR_START = 0.00085
SCHEDULER_LINEAR_END = 0.0120
SCHEDULER_TIMESTEPS = 1000
SCHEDLER_SCHEDULE = "scaled_linear"

sd_config = {
    "_class_name": "EulerAncestralDiscreteScheduler",
    "_diffusers_version": "0.24.0.dev0",
    "beta_end": 0.012,
    "beta_schedule": "scaled_linear",
    "beta_start": 0.00085,
    "clip_sample": False,
    "interpolation_type": "linear",
    "num_train_timesteps": 1000,
    "prediction_type": "epsilon",
    "sample_max_value": 1.0,
    "set_alpha_to_one": False,
    # "skip_prk_steps": False,  # for training
    "skip_prk_steps": True,
    # "steps_offset": 1,
    "steps_offset": 0,
    # "timestep_spacing": "trailing", # for training
    "timestep_spacing": "leading",
    "trained_betas": None
}

pixart_config = {
  "_class_name": "DPMSolverMultistepScheduler",
  "_diffusers_version": "0.22.0.dev0",
  "algorithm_type": "dpmsolver++",
  "beta_end": 0.02,
  "beta_schedule": "linear",
  "beta_start": 0.0001,
  "dynamic_thresholding_ratio": 0.995,
  "euler_at_final": False,
  # "lambda_min_clipped": -Infinity,
  "lambda_min_clipped": -math.inf,
  "lower_order_final": True,
  "num_train_timesteps": 1000,
  "prediction_type": "epsilon",
  "sample_max_value": 1.0,
  "solver_order": 2,
  "solver_type": "midpoint",
  "steps_offset": 0,
  "thresholding": False,
  "timestep_spacing": "linspace",
  "trained_betas": None,
  "use_karras_sigmas": False,
  "use_lu_lambdas": False,
  "variance_type": None
}

flux_config = {
  "_class_name": "FlowMatchEulerDiscreteScheduler",
  "_diffusers_version": "0.30.0.dev0",
  "base_image_seq_len": 256,
  "base_shift": 0.5,
  "max_image_seq_len": 4096,
  "max_shift": 1.15,
  "num_train_timesteps": 1000,
  "shift": 3.0,
  "use_dynamic_shifting": True
}

sd_flow_config = {
  "_class_name": "FlowMatchEulerDiscreteScheduler",
  "_diffusers_version": "0.30.0.dev0",
  "base_image_seq_len": 256,
  "base_shift": 0.5,
  "max_image_seq_len": 4096,
  "max_shift": 1.15,
  "num_train_timesteps": 1000,
  "shift": 3.0,
  "use_dynamic_shifting": False
}

lumina2_config = {
  "_class_name": "FlowMatchEulerDiscreteScheduler",
  "_diffusers_version": "0.33.0.dev0",
  "base_image_seq_len": 256,
  "base_shift": 0.5,
  "invert_sigmas": False,
  "max_image_seq_len": 4096,
  "max_shift": 1.15,
  "num_train_timesteps": 1000,
  "shift": 6.0,
  "shift_terminal": None,
  "use_beta_sigmas": False,
  "use_dynamic_shifting": False,
  "use_exponential_sigmas": False,
  "use_karras_sigmas": False
}


def get_sampler(
        sampler: str,
        kwargs: dict = None,
        arch: str = "sd"
):
    sched_init_args = {}
    if kwargs is not None:
        sched_init_args.update(kwargs)

    config_to_use = copy.deepcopy(sd_config) if arch == "sd" else copy.deepcopy(pixart_config)

    if sampler.startswith("k_"):
        sched_init_args["use_karras_sigmas"] = True

    if sampler == "ddim":
        scheduler_cls = DDIMScheduler
    elif sampler == "ddpm":  # ddpm is not supported ?
        scheduler_cls = DDPMScheduler
    elif sampler == "pndm":
        scheduler_cls = PNDMScheduler
    elif sampler == "lms" or sampler == "k_lms":
        scheduler_cls = LMSDiscreteScheduler
    elif sampler == "euler" or sampler == "k_euler":
        scheduler_cls = EulerDiscreteScheduler
    elif sampler == "euler_a":
        scheduler_cls = EulerAncestralDiscreteScheduler
    elif sampler == "dpmsolver" or sampler == "dpmsolver++" or sampler == "k_dpmsolver" or sampler == "k_dpmsolver++":
        scheduler_cls = DPMSolverMultistepScheduler
        sched_init_args["algorithm_type"] = sampler.replace("k_", "")
    elif sampler == "dpmsingle":
        scheduler_cls = DPMSolverSinglestepScheduler
    elif sampler == "heun":
        scheduler_cls = HeunDiscreteScheduler
    elif sampler == "dpm_2":
        scheduler_cls = KDPM2DiscreteScheduler
    elif sampler == "dpm_2_a":
        scheduler_cls = KDPM2AncestralDiscreteScheduler
    elif sampler == "lcm":
        scheduler_cls = LCMScheduler
    elif sampler == "custom_lcm":
        scheduler_cls = CustomLCMScheduler
    elif sampler == "mean_flow":
        scheduler_cls = MeanFlowScheduler
    elif sampler == "flowmatch":
        scheduler_cls = CustomFlowMatchEulerDiscreteScheduler
        config_to_use = copy.deepcopy(flux_config)
        if arch == "sd":
            config_to_use = copy.deepcopy(sd_flow_config)
        elif arch == "flux":
            config_to_use = copy.deepcopy(flux_config)
        elif arch == "lumina2":
            config_to_use = copy.deepcopy(lumina2_config)
        else:
            print(f"Unknown architecture {arch}, using default flux config")
            # use flux by default
            config_to_use = copy.deepcopy(flux_config)
    else:
        raise ValueError(f"Sampler {sampler} not supported")


    config = copy.deepcopy(config_to_use)
    config.update(sched_init_args)

    scheduler = scheduler_cls.from_config(config)

    return scheduler


SIGMA_SCHEDULES = {
    "karras": "use_karras_sigmas",
    "exponential": "use_exponential_sigmas",
}


def lcm_sigma_schedule_timesteps(scheduler, num_inference_steps, schedule):
    """Timesteps for an LCMScheduler that follow ComfyUI's karras or
    exponential sigmas. ComfyUI spaces sigmas between the model's sigma_min
    and sigma_max, appends a 0 and has its LCM sampler denoise at each sigma
    then re-noise to the next one, which is what LCMScheduler.step does
    between timesteps. Each sigma maps to the training timestep nearest in
    log sigma, as ComfyUI's discrete model sampling does."""
    alphas_cumprod = scheduler.alphas_cumprod.double().cpu().numpy()
    log_sigmas = np.log(((1 - alphas_cumprod) / alphas_cumprod) ** 0.5)
    sigma_min, sigma_max = math.exp(log_sigmas[0]), math.exp(log_sigmas[-1])
    if schedule == "exponential":
        sigmas = np.exp(np.linspace(math.log(sigma_max), math.log(sigma_min), num_inference_steps))
    else:
        rho = 7.0
        ramp = np.linspace(0, 1, num_inference_steps)
        min_inv_rho, max_inv_rho = sigma_min ** (1 / rho), sigma_max ** (1 / rho)
        sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho
    timesteps = np.abs(np.log(sigmas)[:, None] - log_sigmas[None, :]).argmin(axis=1)
    # LCMScheduler needs strictly descending timesteps; only long schedules
    # can land two sigmas on the same timestep near sigma_min
    timesteps = np.unique(timesteps)[::-1]
    return [int(t) for t in timesteps]


def _use_lcm_sigma_schedule(scheduler, schedule):
    """Make this LCMScheduler instance build karras or exponential timesteps
    whenever a pipeline asks it for num_inference_steps."""
    original_set_timesteps = scheduler.set_timesteps

    def set_timesteps(num_inference_steps=None, device=None, original_inference_steps=None,
                      timesteps=None, strength=1.0):
        if timesteps is not None or num_inference_steps is None:
            return original_set_timesteps(num_inference_steps, device, original_inference_steps, timesteps, strength)
        timesteps = lcm_sigma_schedule_timesteps(scheduler, num_inference_steps, schedule)
        # diffusers warns that custom timesteps are off the distillation
        # schedule, which is the point here
        lcm_logger = diffusers_logging.get_logger("diffusers.schedulers.scheduling_lcm")
        level = lcm_logger.level
        lcm_logger.setLevel(diffusers_logging.ERROR)
        try:
            original_set_timesteps(None, device, original_inference_steps, timesteps, strength)
        finally:
            lcm_logger.setLevel(level)

    scheduler.set_timesteps = set_timesteps


def swap_sigma_schedule(pipeline, schedule):
    """Give the pipeline a copy of its scheduler that spaces sigmas on the
    karras or exponential schedule. Returns a function that puts the original
    scheduler back. No-op for 'default' or schedulers without the option."""
    schedule = (schedule or "default").lower()
    scheduler = getattr(pipeline, "scheduler", None)
    if schedule == "default" or scheduler is None:
        return lambda: None
    if schedule not in SIGMA_SCHEDULES:
        raise ValueError(f"Sample scheduler {schedule} not supported, use default, karras or exponential")
    flag = SIGMA_SCHEDULES[schedule]
    if isinstance(scheduler, LCMScheduler):
        sampling_scheduler = copy.deepcopy(scheduler)
        _use_lcm_sigma_schedule(sampling_scheduler, schedule)
        pipeline.scheduler = sampling_scheduler

        def restore():
            pipeline.scheduler = scheduler

        return restore
    if flag not in scheduler.config:
        print(f"Sample scheduler {schedule} is not supported by {type(scheduler).__name__}, using its default sigmas")
        return lambda: None
    overrides = {name: False for name in ("use_karras_sigmas", "use_exponential_sigmas", "use_beta_sigmas")
                 if name in scheduler.config}
    overrides[flag] = True
    # a copy so the training noise scheduler is never touched. from_config is
    # avoided as CustomFlowMatchEulerDiscreteScheduler marks every key as a
    # default there and would lose its shift settings
    sampling_scheduler = copy.deepcopy(scheduler)
    sampling_scheduler.register_to_config(**overrides)
    pipeline.scheduler = sampling_scheduler

    def restore():
        pipeline.scheduler = scheduler

    return restore
