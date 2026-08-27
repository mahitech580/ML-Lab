"""
report_generator.py

Generates professional Markdown/text reports summarizing a training run:
dataset, model, metrics, parameters, feature importance, and a factual
(non-hyperbolic) conclusion.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

import config


class ReportGenerator:
    def __init__(self, directory: Path | None = None):
        self.directory = directory or config.REPORTS_DIR
        self.directory.mkdir(parents=True, exist_ok=True)

    def generate(
        self,
        dataset_name: str,
        problem_type: str,
        model_name: str,
        n_train: int,
        n_test: int,
        features: list[str],
        metrics: dict,
        parameters: dict,
        feature_importance: dict | None = None,
        baseline_metrics: dict | None = None,
        fmt: str = "md",
    ) -> Path:
        lines = []
        lines.append("# ML Lab Model Report\n")
        lines.append(f"*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n")

        lines.append("## Dataset")
        lines.append(f"- **Name:** {dataset_name}")
        lines.append(f"- **Problem type:** {problem_type}")
        lines.append(f"- **Training samples:** {n_train}")
        lines.append(f"- **Testing samples:** {n_test}")
        lines.append(f"- **Features used ({len(features)}):** {', '.join(features)}\n")

        lines.append("## Model")
        lines.append(f"- **Algorithm:** {model_name}")
        lines.append("- **Parameters:**")
        for k, v in parameters.items():
            lines.append(f"  - {k}: {v}")
        lines.append("")

        lines.append("## Metrics")
        for k, v in metrics.items():
            if v is None:
                continue
            v_str = f"{v:.4f}" if isinstance(v, float) else str(v)
            lines.append(f"- **{k}:** {v_str}")
        lines.append("")

        if baseline_metrics:
            lines.append("## Baseline Comparison")
            for k, v in baseline_metrics.items():
                if v is None:
                    continue
                v_str = f"{v:.4f}" if isinstance(v, float) else str(v)
                lines.append(f"- Baseline {k}: {v_str}")
            lines.append("")

        if feature_importance:
            lines.append("## Feature Importance (top 10)")
            for name, value in list(feature_importance.items())[:10]:
                lines.append(f"- {name}: {value:.4f}")
            lines.append("")

        lines.append("## Conclusion")
        conclusion = self._build_conclusion(model_name, metrics, problem_type)
        lines.append(conclusion)
        lines.append("")

        lines.append("## Limitations")
        lines.append(
            "- Metrics reflect performance on a single train/test split and may vary "
            "with different splits or random seeds.\n"
            "- Synthetic or small datasets may not generalize to real-world data.\n"
            "- No claim is made that this model is optimal beyond the models evaluated."
        )

        content = "\n".join(lines)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        ext = "md" if fmt == "md" else "txt"
        path = self.directory / f"report_{timestamp}.{ext}"
        path.write_text(content, encoding="utf-8")
        return path

    @staticmethod
    def _build_conclusion(model_name: str, metrics: dict, problem_type: str) -> str:
        if problem_type == "classification":
            primary = metrics.get("f1") or metrics.get("accuracy")
            metric_label = "F1 score" if metrics.get("f1") is not None else "accuracy"
        elif problem_type == "regression":
            primary = metrics.get("r2")
            metric_label = "R\u00b2"
        else:
            primary = metrics.get("silhouette_score")
            metric_label = "silhouette score"

        if primary is None:
            return f"{model_name} completed evaluation; refer to the metrics above for details."

        return (
            f"{model_name} achieved a test {metric_label} of {primary:.4f} on this dataset "
            f"and evaluation split. This reflects performance on the specific data and split "
            f"used, not a general claim about the algorithm."
        )
