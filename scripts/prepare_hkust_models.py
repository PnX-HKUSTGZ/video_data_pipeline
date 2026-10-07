"""复制指定 HKUST 权重并提取可移植推理配置，不执行源项目训练代码。"""

import argparse
import ast
import os
import pprint
import shutil
from pathlib import Path

import yaml


def literal(node):
    """源快照是展开后的字面量配置，只额外允许 dict(...) 构造。"""
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "dict":
        value = dict(literal(node.args[0])) if node.args else {}
        value.update({k.arg: literal(k.value) for k in node.keywords})
        return value
    if isinstance(node, ast.Dict):
        return {literal(k): literal(v) for k, v in zip(node.keys, node.values, strict=True)}
    if isinstance(node, ast.List):
        return [literal(v) for v in node.elts]
    if isinstance(node, ast.Tuple):
        return tuple(literal(v) for v in node.elts)
    return ast.literal_eval(node)


def prepare(source: Path, destination: Path):
    """保留模型图、点序、测试预处理与解码设置，移除训练路径及训练扩展导入。"""
    assets = {
        "marker": (
            "reports/precision_pillar_noflip_selection_v1/result/validation/pillar5/epoch_19/checkpoint_snapshot.pth",
            "reports/precision_pillar_noflip_selection_v1/result/validation/pillar5/epoch_19/config_snapshot.py",
        ),
        "station": (
            "reports/station7_sampling2x_v1/result/train_station7/iter_4000.pth",
            "reports/station7_sampling2x_v1/result/train_station7/runtime_config.py",
        ),
    }
    if destination.exists() and any(destination.iterdir()):
        raise ValueError(f"Destination must be empty: {destination}")
    for name, (weight, config) in assets.items():
        directory = destination / name
        directory.mkdir(parents=True)
        tree = ast.parse((source / config).read_text())
        selected = {}
        for node in tree.body:
            if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
                key = node.targets[0].id
                if key in ("model", "metainfo", "test_pipeline"):
                    selected[key] = literal(node.value)
        selected["model"]["backbone"]["init_cfg"] = None
        meta = selected["metainfo"]
        config_values = {
            "default_scope": "mmpose",
            "custom_imports": {
                "imports": ["video_dataset_pipeline.inference.visibility_head"],
                "allow_failed_imports": False,
            },
            "model": selected["model"],
            "train_dataloader": {"dataset": {"type": "CocoDataset", "metainfo": meta}},
            "test_dataloader": {
                "dataset": {
                    "type": "CocoDataset",
                    "metainfo": meta,
                    "pipeline": selected["test_pipeline"],
                }
            },
        }
        (directory / "inference.py").write_text(
            f"# Extracted from {config}\n"
            + "\n".join(
                f"{key} = {pprint.pformat(value, width=100, sort_dicts=False)}\n"
                for key, value in config_values.items()
            )
        )
        shutil.copy2(source / weight, directory / "checkpoint.pth")
    detector_dir = destination / "detector"
    detector_dir.mkdir()
    shutil.copy2(source / "checkpoints/yolo_v2/weights/best.pt", detector_dir / "best.pt")
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    from ultralytics import YOLO

    detector = YOLO(detector_dir / "best.pt")
    if detector.task != "detect":
        raise ValueError("Expected detection checkpoint")
    (detector_dir / "metadata.yaml").write_text(
        yaml.safe_dump({"task": detector.task, "names": detector.names}, allow_unicode=True)
    )
    print(f"Prepared {destination.resolve()}; detector names: {detector.names}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("~/code/tech_core_hkust").expanduser())
    parser.add_argument("--output", type=Path, default=Path("models/hkust"))
    args = parser.parse_args()
    prepare(args.source.resolve(), args.output.resolve())
