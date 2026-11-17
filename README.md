# Orange Juice Impedance Spectroscopy: Machine Learning

The code behind the machine-learning results in our paper (the reference will be added once it is published). It trains classical ML models (scikit-learn, XGBoost, CatBoost, …) on impedance spectra of orange juices for three tasks:

| Notebook | Task |
|---|---|
| `mixture_regression_classification.ipynb` | Binary brand mixture: regress the mixture fractions and classify binned fractions |
| `adulteration_regression.ipynb` | Water adulteration: regress the water fraction |
| `adulteration_classification.ipynb` | Water adulteration: classify binned water levels |
| `orange_class_classification.ipynb` | Classify the brand (Brand I–IV), with train/test split by sample group |
| `result_summary.ipynb` | Average the per-iteration results into the summary tables in `res/` |

`experimentation.py` contains the model zoo and the evaluation loop, `exp.eval`: repeated K-fold cross-validation over several feature preprocessing pipelines, with one row written per model, fold and iteration.

## Data

The dataset is in a separate repository: [Orange-Juice-EIS-Dataset](https://github.com/Zeynep-Kara/Orange-Juice-EIS-Dataset). Download the three CSV files into `experiment_logs/`:

```
experiment_logs/
├── merged_mixtures_full_experiments_orange.csv
├── merged_adulteration_experiments.csv
└── merged_classification.csv
```

The dataset repository's README describes the format and the labels.

## Setup

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

## Running

1. Run the experiment notebooks. Each writes raw per-fold results to `iter_50/{mixture,adult,class}/`. With 50 iterations, a full run takes several hours.
2. Run `result_summary.ipynb` to average them into `res/`.

Two feature sets are evaluated:
- **x**: the raw spectrum (all frequency points).
- **xm**: summary statistics per measured quantity (std, mean, median, min and max).

## Results

`res/` contains the summary results reported in the paper.

> **Note:** the published results were obtained with two additional mixture runs that involved a fifth brand. Those runs are not in the public dataset, so re-running the notebooks may give slightly different scores, probably a little better.

## License

The code is released under the [MIT License](LICENSE).

The dataset is licensed separately under [CC BY 4.0](https://github.com/Zeynep-Kara/Orange-Juice-EIS-Dataset/blob/main/LICENSE). **If you use the data, with this code or otherwise, you must cite the accompanying paper.**

## Citation

If you use the dataset, please cite the accompanying paper. We would also appreciate a citation if this code helps your work. The full reference and BibTeX entry will be added here once the paper is published.
