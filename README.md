# Veritas AI Hackathon 2026: Health track

Prep work for the Veritas AI International High School Hackathon (Nov 14–15, 2026).

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/erjanowitz-code/veritas-health-hackathon/blob/main/baseline_test.ipynb)

`baseline_test.ipynb` is the notebook version: open it in Colab and use **Runtime → Run all**.

## baseline_test.py

Compares a "most frequent class" baseline with a Random Forest on real health data
loaded from Hugging Face, using cross-validation, a held-out test set, recall,
a per-group fairness check and feature importances.

```bash
pip install pandas scikit-learn
python baseline_test.py            # UCI Heart Disease (303 patients)
python baseline_test.py diabetes   # Pima Indians Diabetes (768 patients)
```

| Dataset | Baseline | Random Forest (5-fold CV) |
|---|---|---|
| Heart disease | 54% | 81% ± 2% |
| Diabetes | 65% | 77% ± 3% |

## Sources

- UCI Heart Disease (Cleveland): https://huggingface.co/datasets/skrishna/heart_disease_uci
  (original: https://archive.ics.uci.edu/dataset/45/heart+disease)
- Pima Indians Diabetes Database: https://huggingface.co/datasets/khoaguin/pima-indians-diabetes-database
- AI tools used: Claude (coding assistant)

> Prototype for learning only. Not medical advice.
