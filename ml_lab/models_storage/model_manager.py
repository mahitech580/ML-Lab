"""
model_manager.py

Persists trained pipelines to disk with Joblib, alongside a JSON
metadata sidecar file (model name, dataset, target, features,
parameters, metrics, timestamp).

Security note: joblib.load can execute arbitrary code if given a
malicious file. Only load model files you trust (e.g. ones this
application saved).
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import joblib

import config

logger = logging.getLogger("ml_lab.models_storage")


@dataclass
class ModelMetadata:
    model_name: str
    dataset: str
    target: str
    features: list
    parameters: dict
    metrics: dict
    created_at: str
    problem_type: str


class ModelManager:
    def __init__(self, directory: Path | None = None):
        self.directory = directory or config.SAVED_MODELS_DIR
        self.directory.mkdir(parents=True, exist_ok=True)

    def _slugify(self, name: str) -> str:
        return "".join(c if c.isalnum() else "_" for c in name.lower())

    def save_model(
        self,
        pipeline,
        model_name: str,
        dataset: str,
        target: str,
        features: list,
        parameters: dict,
        metrics: dict,
        problem_type: str,
    ) -> tuple[Path, Path]:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = f"{self._slugify(model_name)}_{timestamp}"
        model_path = self.directory / f"{base_name}.joblib"
        meta_path = self.directory / f"{base_name}.json"

        joblib.dump(pipeline, model_path)

        metadata = ModelMetadata(
            model_name=model_name,
            dataset=dataset,
            target=target,
            features=features,
            parameters=_json_safe(parameters),
            metrics=_json_safe(metrics),
            created_at=datetime.now().isoformat(timespec="seconds"),
            problem_type=problem_type,
        )
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(metadata.__dict__, f, indent=2)

        logger.info("Saved model to %s", model_path)
        return model_path, meta_path

    def list_models(self) -> list[dict]:
        results = []
        for meta_path in sorted(self.directory.glob("*.json")):
            model_path = meta_path.with_suffix(".joblib")
            if not model_path.exists():
                continue
            with open(meta_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)
            metadata["model_path"] = str(model_path)
            metadata["meta_path"] = str(meta_path)
            results.append(metadata)
        return results

    def load_model(self, model_path: str):
        path = Path(model_path)
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")
        return joblib.load(path)

    def delete_model(self, model_path: str) -> bool:
        path = Path(model_path)
        meta_path = path.with_suffix(".json")
        deleted = False
        if path.exists():
            path.unlink()
            deleted = True
        if meta_path.exists():
            meta_path.unlink()
        return deleted


def _json_safe(obj):
    import numpy as np
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj
