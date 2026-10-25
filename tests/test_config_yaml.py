"""Regression tests for YAML config loading (E050 follow-up).

`load_config` used `yaml.safe_load` without importing yaml, so every YAML config path raised
`NameError: name 'yaml' is not defined`. JSON worked, which is why nothing caught it: the only
callers in the suite went through `--config` with JSON or never hit that branch.
"""
import json

import pytest

from tgtrack.config import Config, load_config


def test_yaml_config_path_is_loadable(tmp_path):
    p = tmp_path / "cfg.yaml"
    p.write_text(
        "detector:\n  imgsz: 960\n  conf: 0.4\ncost:\n  distractor_margin: 0.11\n",
        encoding="utf-8",
    )
    cfg = load_config(p)
    assert isinstance(cfg, Config)
    assert cfg.detector.imgsz == 960
    assert cfg.detector.conf == 0.4
    assert cfg.cost.distractor_margin == 0.11


def test_yml_suffix_is_also_yaml(tmp_path):
    p = tmp_path / "cfg.yml"
    p.write_text("embedder:\n  imgsz: 448\n", encoding="utf-8")
    assert load_config(p).embedder.imgsz == 448


def test_json_config_still_works(tmp_path):
    p = tmp_path / "cfg.json"
    p.write_text(json.dumps({"memory": {"ltm_capacity": 12}}), encoding="utf-8")
    assert load_config(p).memory.ltm_capacity == 12


def test_yaml_then_dotted_overrides(tmp_path):
    p = tmp_path / "cfg.yaml"
    p.write_text("detector:\n  imgsz: 960\n", encoding="utf-8")
    cfg = load_config(p, **{"detector.imgsz": 1280, "cost.search_margin": 0.3})
    assert cfg.detector.imgsz == 1280
    assert cfg.cost.search_margin == 0.3


def test_unknown_key_still_raises(tmp_path):
    p = tmp_path / "cfg.yaml"
    p.write_text("detector:\n  not_a_field: 1\n", encoding="utf-8")
    with pytest.raises(KeyError):
        load_config(p)
