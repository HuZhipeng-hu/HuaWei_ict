"""Internal optional EMG checkpoint warm-start helpers for the release path."""

from __future__ import annotations

from pathlib import Path

from shared.models.blocks import _check_mindspore

try:
    from mindspore import load_checkpoint
except Exception:
    load_checkpoint = None  # type: ignore


def load_emg_encoder_from_checkpoint(event_model, checkpoint_path: str | Path) -> dict[str, int]:
    """Copy matching EMG branch weights into the event model from a checkpoint."""
    _check_mindspore()
    param_dict = load_checkpoint(str(checkpoint_path))
    current = {param.name: param for param in event_model.get_parameters()}
    loaded = 0
    skipped = 0
    mapping_prefixes = {
        "block1.": "emg_block1.",
        "block2.": "emg_block2.",
    }
    for source_name, tensor in param_dict.items():
        target_name = None
        for src_prefix, dst_prefix in mapping_prefixes.items():
            if source_name.startswith(src_prefix):
                target_name = dst_prefix + source_name[len(src_prefix) :]
                break
        if target_name is None or target_name not in current:
            skipped += 1
            continue
        target_param = current[target_name]
        if tuple(int(x) for x in target_param.shape) != tuple(int(x) for x in tensor.shape):
            skipped += 1
            continue
        target_param.set_data(tensor)
        loaded += 1
    return {"loaded": loaded, "skipped": skipped}
