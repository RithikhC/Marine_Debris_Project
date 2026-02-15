"""
Train YOLOv8n-seg on marine debris dataset.
Uses data.yaml, saves best weights.
"""

import tempfile
from pathlib import Path

import yaml
from ultralytics import YOLO


def main():
    project_root = Path(__file__).resolve().parent
    datasets_dir = project_root / "datasets"

    # Load data.yaml and set absolute path so Ultralytics finds images in this project
    data_yaml = project_root / "data.yaml"
    with open(data_yaml) as f:
        data_cfg = yaml.safe_load(f)
    data_cfg["path"] = str(datasets_dir)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(data_cfg, f, default_flow_style=False, sort_keys=False)
        temp_data = f.name

    try:
        model = YOLO("yolov8n-seg.pt")
        model.train(
            data=temp_data,
            imgsz=640,
            epochs=50,
            save=True,
            project=str(project_root / "runs" / "segment"),
            name="train",
        )
    finally:
        Path(temp_data).unlink(missing_ok=True)


if __name__ == "__main__":
    main()
