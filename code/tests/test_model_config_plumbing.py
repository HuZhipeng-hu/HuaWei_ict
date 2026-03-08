from argparse import Namespace

import training.evaluate as training_evaluate
from shared.config import AugmentationConfig, ModelConfig, TrainingConfig
from training.train import _apply_cli_overrides


class _DummyModel:
    def set_train(self, _value):
        return None


def test_train_cli_overrides_update_model_and_augmentation_settings():
    args = Namespace(
        model_type="lite",
        base_channels=24,
        use_se=False,
        loss_type="focal",
        hard_mining_ratio=0.5,
        augment_factor=3,
        use_mixup=True,
        augmentation_enabled=True,
        split_seed=99,
    )
    model_cfg = ModelConfig()
    train_cfg = TrainingConfig()
    augmentation_cfg = AugmentationConfig()

    model_cfg, train_cfg, augmentation_cfg = _apply_cli_overrides(args, model_cfg, train_cfg, augmentation_cfg)

    assert model_cfg.model_type == "lite"
    assert model_cfg.base_channels == 24
    assert model_cfg.use_se is False
    assert train_cfg.loss.type == "focal"
    assert train_cfg.sampler.hard_mining_ratio == 0.5
    assert train_cfg.split_seed == 99
    assert augmentation_cfg.augment_factor == 3
    assert augmentation_cfg.use_mixup is True
    assert augmentation_cfg.enabled is True


def test_evaluate_load_model_uses_structural_model_config(monkeypatch):
    seen = {}

    def _fake_build_model(model_config, dropout_rate=None):
        seen["model_type"] = model_config.model_type
        seen["base_channels"] = model_config.base_channels
        seen["use_se"] = model_config.use_se
        seen["dropout_rate"] = dropout_rate
        return _DummyModel()

    monkeypatch.setattr(training_evaluate, "ms", object())
    monkeypatch.setattr(training_evaluate, "load_checkpoint", lambda _path: {"ok": 1})
    monkeypatch.setattr(training_evaluate, "load_param_into_net", lambda _model, _params: None)
    monkeypatch.setattr(training_evaluate, "build_model_from_config", _fake_build_model)

    model_cfg = ModelConfig(model_type="lite", base_channels=32, use_se=False, dropout_rate=0.25)
    model = training_evaluate.load_model_from_checkpoint("dummy.ckpt", model_cfg, dropout_rate=0.0)

    assert isinstance(model, _DummyModel)
    assert seen == {
        "model_type": "lite",
        "base_channels": 32,
        "use_se": False,
        "dropout_rate": 0.0,
    }
