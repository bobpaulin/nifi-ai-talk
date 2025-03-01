# table-cli

-----

## Installation

```console
python -m venv .env
source bin/activate
pip install -r requirements.txt
```

## Execution

### Execute Image Detection - Image Output

```console
python detect_image.py apple-chart-100px-whitespace.png
```

Check output/detect.png

### Execute Image Detection - Coco Output

```console
python detect_coco.py apple-chart-100px-whitespace.png
```

Check output/coco.json