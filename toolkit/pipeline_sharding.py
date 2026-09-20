import torch
from typing import List, Optional

from toolkit.print import print_acc


def _module_param_bytes(module: torch.nn.Module) -> int:
    total = 0
    for p in module.parameters():
        total += p.numel() * p.element_size()
    for b in module.buffers():
        total += b.numel() * b.element_size()
    return total


def _module_device(module: torch.nn.Module) -> Optional[torch.device]:
    p = next(module.parameters(), None)
    if p is not None:
        return p.device
    b = next(module.buffers(), None)
    if b is not None:
        return b.device
    return None


def _move_to_device(obj, device: torch.device):
    # recursively move tensors inside args/kwargs structures; non-tensors pass
    # through untouched. Handles the tuple blockvec and list kv caches that some
    # model forwards thread between blocks.
    if isinstance(obj, torch.Tensor):
        if obj.device != device:
            return obj.to(device)
        return obj
    if isinstance(obj, tuple):
        return tuple(_move_to_device(o, device) for o in obj)
    if isinstance(obj, list):
        return [_move_to_device(o, device) for o in obj]
    if isinstance(obj, dict):
        return {k: _move_to_device(v, device) for k, v in obj.items()}
    return obj


class PipelineShardManager:
    """Splits a transformer's repeated blocks across multiple gpus so models
    larger than one gpu's vram can train without cpu offloading.

    The repeated blocks (from the model's get_transformer_block_names) are
    assigned contiguously to each device by weight size; every other submodule
    stays on the primary device. forward_pre_hooks move each block's tensor
    inputs to that block's device, and autograd routes gradients back across
    the same boundaries, so no trainer changes are needed. The module's .to()
    is overridden (same trick as MemoryManager) so whole-model device moves
    elsewhere in the codebase become dtype-only no-ops instead of collapsing
    the shards back onto one gpu.
    """

    def __init__(self, module: torch.nn.Module, devices: List[torch.device]):
        self.module = module
        self.devices = devices
        self.hook_handles = []
        self.block_assignments = {}  # id(block) -> device

    # ------------------------------------------------------------------
    # .to override
    # ------------------------------------------------------------------
    def sharded_to(self, *args, **kwargs):
        # device moves would collapse the shards onto one gpu; only honor dtype
        dtype = kwargs.get("dtype", None)
        if dtype is None:
            for arg in args:
                if isinstance(arg, torch.dtype):
                    dtype = arg
                    break
        if dtype is not None:
            # Module.to(dtype=...) casts in place without touching devices
            self.module._ps_orig_to(dtype=dtype)
        return self.module

    # ------------------------------------------------------------------
    # attach
    # ------------------------------------------------------------------
    @classmethod
    def attach(
        cls,
        module: torch.nn.Module,
        devices: List[torch.device],
        block_names: List[str],
        dtype: Optional[torch.dtype] = None,
        balance: Optional[List[float]] = None,
    ):
        if hasattr(module, "_shard_manager"):
            return module._shard_manager
        devices = [torch.device(d) for d in devices]
        if len(devices) < 2:
            raise ValueError(
                "pipeline sharding needs at least 2 devices; got "
                f"{[str(d) for d in devices]}"
            )
        available = torch.cuda.device_count()
        for d in devices:
            if d.type == "cuda" and (d.index or 0) >= available:
                raise ValueError(
                    f"pipeline device {d} is not visible to this process "
                    f"(torch sees {available} cuda device(s)). If launched from "
                    "the UI, CUDA_VISIBLE_DEVICES may be pinning the job to one "
                    "gpu; run.py lifts this automatically when the config sets "
                    "pipeline_sharding: true — make sure that code is present."
                )
        if balance is not None and len(balance) != len(devices):
            raise ValueError(
                f"pipeline_balance has {len(balance)} entries for {len(devices)} devices"
            )

        manager = cls(module, devices)
        module._shard_manager = manager

        # gather the repeated blocks; names may be dotted paths
        blocks: List[torch.nn.Module] = []
        for name in block_names:
            block_list = module
            for part in name.split("."):
                block_list = getattr(block_list, part, None)
                if block_list is None:
                    break
            if block_list is not None:
                blocks += list(block_list)
        if len(blocks) == 0:
            raise ValueError(f"no blocks found for names {block_names}")

        block_ids = set(id(b) for b in blocks)

        # contiguous assignment weighted by bytes (and optional balance ratios)
        sizes = [_module_param_bytes(b) for b in blocks]
        total = sum(sizes)
        weights = balance if balance is not None else [1.0] * len(devices)
        wsum = sum(weights)
        # cumulative byte cutoffs per device
        cutoffs = []
        acc = 0.0
        for w in weights:
            acc += w / wsum
            cutoffs.append(acc * total)

        assignment: List[torch.device] = []
        running = 0
        dev_idx = 0
        for s in sizes:
            # advance to the next device once this one's byte budget is used up,
            # but never past the last device
            while dev_idx < len(devices) - 1 and running + s / 2 > cutoffs[dev_idx]:
                dev_idx += 1
            assignment.append(devices[dev_idx])
            running += s

        # move blocks; when dtype is None (quantized models) only the device moves
        per_device_count = {str(d): 0 for d in devices}
        for block, device in zip(blocks, assignment):
            if dtype is not None:
                block.to(device, dtype=dtype)
            else:
                block.to(device)
            manager.block_assignments[id(block)] = device
            per_device_count[str(device)] += 1

        # everything that is not a repeated block lives on the primary device.
        # walk direct children; a child that CONTAINS the block lists (dotted
        # path case) is skipped at this level and its own non-block parts are
        # handled by the recursion below.
        primary = devices[0]

        def move_non_blocks(mod: torch.nn.Module):
            for child in mod.children():
                if id(child) in block_ids:
                    continue
                if any(id(sub) in block_ids for sub in child.modules()):
                    move_non_blocks(child)
                    continue
                if dtype is not None:
                    child.to(primary, dtype=dtype)
                else:
                    child.to(primary)
            # direct parameters/buffers on this module itself
            for p in list(mod._parameters.values()):
                if p is not None and p.device != primary:
                    p.data = p.data.to(primary)
                    if p.grad is not None:
                        p.grad = p.grad.to(primary)
            for name, b in list(mod._buffers.items()):
                if b is not None and b.device != primary:
                    mod._buffers[name] = b.to(primary)

        move_non_blocks(module)
        if dtype is not None:
            # cast any float params that sat directly on container modules
            for p in module.parameters():
                if p.device == primary and p.is_floating_point() and p.dtype != dtype:
                    p.data = p.data.to(dtype)

        # input-routing hooks: each block pulls its tensor inputs onto its own
        # device; tail modules (e.g. the final layer) pull results back to the
        # primary device because that is where their weights live.
        def make_hook(device: torch.device):
            def hook(mod, args, kwargs):
                return _move_to_device(args, device), _move_to_device(kwargs, device)
            return hook

        for block, device in zip(blocks, assignment):
            manager.hook_handles.append(
                block.register_forward_pre_hook(make_hook(device), with_kwargs=True)
            )
        # hooks on non-block direct children with weights (covers modules that
        # run AFTER the block loop, like a LastLayer); harmless no-ops for the
        # ones that run before it.
        for child in module.children():
            if id(child) in block_ids:
                continue
            if any(id(sub) in block_ids for sub in child.modules()):
                continue
            if _module_device(child) is None:
                continue
            manager.hook_handles.append(
                child.register_forward_pre_hook(make_hook(primary), with_kwargs=True)
            )

        # neutralize whole-model device moves
        module._ps_orig_to = module.to
        module.to = manager.sharded_to

        summary = ", ".join(
            f"{dev}: {cnt} blocks" for dev, cnt in per_device_count.items()
        )
        print_acc(f"Pipeline sharding attached ({summary}; extras on {primary})")
        return manager

    # ------------------------------------------------------------------
    # aux-module alignment (LoRA / LoKr networks)
    # ------------------------------------------------------------------
    @staticmethod
    def align_network(network):
        """Move each adapter module (LoKr/LoRA) onto the device of the layer it
        wraps. Networks are created after sharding and force_to()'d onto one
        device; adapter math (e.g. LoKr's factorized einsums) runs against
        activations living on the wrapped layer's device, so mismatched
        placement raises a cross-device error on the first step."""
        moved = 0
        for m in network.get_all_modules():
            org = getattr(m, "org_module", None)
            if not org:
                continue
            dev = _module_device(org[0])
            if dev is None:
                continue
            if _module_device(m) != dev:
                m.to(dev)
                moved += 1
        if moved:
            print_acc(f"Pipeline sharding: moved {moved} adapter modules to their wrapped layers' devices")
