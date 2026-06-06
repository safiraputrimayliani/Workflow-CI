# Workflow-CI — Titanic Prediction

Repository ini berisi workflow CI menggunakan **GitHub Actions** dan **MLflow Project** untuk melakukan re-training model secara otomatis setiap kali trigger dipantik.

## Struktur Folder

```
Workflow-CI/
├── .github/
│   └── workflows/
│       └── ci.yml          ← GitHub Actions workflow
├── MLProject/
│   ├── modelling.py        ← Script training model
│   ├── conda.yaml          ← Conda environment
│   ├── MLProject           ← File konfigurasi MLflow Project
│   └── titanic_preprocessing.csv
└── README.md
```

## Cara Kerja

1. Setiap kali ada **push ke branch `main`** (atau trigger manual via `workflow_dispatch`), GitHub Actions akan otomatis berjalan.
2. Actions akan meng-install semua dependensi, lalu menjalankan `mlflow run .` di dalam folder `MLProject`.
3. Artefak hasil training (`confusion_matrix.png`, `classification_report.txt`, `mlruns/`) akan disimpan sebagai GitHub Actions Artifacts.

## Trigger Manual

Buka tab **Actions** di GitHub → pilih workflow **CI - MLflow Training** → klik **Run workflow**.
