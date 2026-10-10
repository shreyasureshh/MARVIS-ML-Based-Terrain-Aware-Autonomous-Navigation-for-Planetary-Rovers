import glob

import pytest

from marvis.config import load_classes, load_dataset_config, load_yaml
from marvis.train import validate_config


@pytest.mark.parametrize("path", sorted(glob.glob("configs/experiments/*.yaml")))
def test_experiment_configs_valid(path):
    validate_config(load_yaml(path))


@pytest.mark.parametrize("bad", [{"image_size": 250}, {"loss": "focal"}, {"aug": "extreme"}])
def test_invalid_config_rejected(bad):
    cfg = {**load_yaml("configs/experiments/E1_unet_r34_wce_light.yaml"), **bad}
    with pytest.raises(ValueError):
        validate_config(cfg)


def test_class_ids_consistent_across_configs():
    info = load_classes()
    ds = load_dataset_config()
    trav = load_yaml("configs/traversability.yaml")
    assert sorted(set(ds["label_map"].values())) == list(range(info.num_classes))
    assert set(trav["class_costs"]) == set(info.names)
    assert info.ignore_index not in ds["label_map"].values()
    assert set(trav["impassable_classes"]) <= set(info.names) and set(trav["high_risk_classes"]) <= set(info.names)


def test_raw_root_override(tmp_path, monkeypatch):
    monkeypatch.setenv("MARVIS_RAW_ROOT", str(tmp_path))
    assert load_dataset_config()["raw_root"] == str(tmp_path)
