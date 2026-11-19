# RiceDxNet – Rice Disease Classification

Fine-tuning an ImageNet-pretrained **ResNet18** to classify 14 rice classes
(13 diseases + healthy) from the *Philippines Rice Diseases* Kaggle dataset,
compared against a **small CNN trained from scratch** as the baseline.

## Setup

```bash
uv venv --python 3.12 .venv
source .venv/bin/activate        # fish: source .venv/bin/activate.fish
uv pip install -r requirements.txt
```
