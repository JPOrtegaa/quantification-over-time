# Methods

- **TOMS multi-regressor pipeline**: see [TOMS_multi_pipeline.md](TOMS_multi_pipeline.md).
- **`classification/`**: classification (Hugging Face, VADER, sklearn training in `trainingModel.py`).
- **`regression/`**: temporal regressors (`TimeSeriesMultinomialRegressor`, TOMS bundle in `toms_multi_regressor.py`).
- **`quantifiers.py`**: quantifiers (DyS, DyS-Opt, ACC, GPAC, EDy, etc.).
- **`ReadMe_Implement/`**: R project and support data (`data/<dataset>/...`) for quantifier `ReadMe2` and `utils/data_proc_toR.py`. Python path: `config.README_IMPLEMENT_DIR`.

Run scripts from the `project/` directory so `import config` and `import methods.*` resolve correctly.
