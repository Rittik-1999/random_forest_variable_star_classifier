# Random forest variable-star classifier

`mlcode.ipynb` contains three stages: assemble a labelled training table from public archives, train a random forest on nine features, and classify a prepared feature CSV. Each stage can be used independently with the required input files.

This folder contains code and this README. Training data, light curves, models and predictions are not bundled.

## Running the notebook

Open the notebook with its project folder as the working directory. Edit the setup cell to choose paths and enable the stage you need:

| Control | Action |
|---|---|
| `RUN_BUILD_TRAINING = True` | Query archives and build the training CSV. Requires internet access and may take many hours. |
| `RUN_TRAIN_MODEL = True` | Train and evaluate using an existing training CSV. |
| `RUN_PREDICT = True` | Load an existing model and classify a prepared feature CSV. |

All three controls default to `False`. Running all cells with the defaults defines the configuration and prints which stages were skipped; it does not download data, train a model or create output files. Set the desired controls in the setup cell before rerunning it and the relevant stage cells. Running a selected stage again can replace its configured output files.

Default locations are relative to `PROJECT_DIR = Path.cwd()`:

```text
project/
    mlcode.ipynb
    README.md
    data/
        variable_star_training_9features_complete.csv
        prediction_features.csv
    outputs/
        variable_star_classifier_9features.joblib
        variable_star_model_training_report_9features.csv
        variable_star_training_truth_table_9features.csv
        predictions.csv
    cache/
        catalogues/
        features/
        checkpoints/
        lightcurves/
```

Directories are created when an enabled stage needs them. Change the paths in the setup cell to use files elsewhere. To predict without rebuilding or retraining, supply `MODEL_PATH` and `PREDICTION_INPUT_CSV`, enable only `RUN_PREDICT`, and run the setup and prediction cells. Load only model files whose source you trust: joblib uses Python's pickle format.

## Environment

Install dependencies in the Python environment used by the notebook. The notebook checks for missing packages and reports them; it does not install packages or change certificate verification settings.

Training and prediction require `numpy`, `pandas`, `scikit-learn` and `joblib`. Archive harvesting additionally requires `astropy`, `astroquery`, `lightkurve`, `scipy`, `requests` and `tqdm`.

The training and prediction regression checks used Python 3.9 with NumPy 1.26.4, pandas 2.3.3, scikit-learn 1.6.1 and joblib 1.5.3. A saved scikit-learn model should be loaded in a compatible environment. The model bundle records the versions used for a new training run, its training-file SHA256 checksum, feature names, class counts, threshold and feature-definition version.

## Stage 1: archive training data

The builder requests up to `CANDIDATES_PER_CLASS` retained stars per class. The default is 2500. A class can finish with fewer rows if the archive or quality selection is exhausted. The output is `TRAINING_CSV`; a separate `combined_before_cleaning.csv` records processed candidates and quality flags under `OUTPUT_DIR`.

| Class | Reference labels and period/light-curve source |
|---|---|
| RRLYR | Gaia DR3 general-classifier RR candidates; Gaia RR Lyrae SOS periods and Fourier quantities |
| CEP | Gaia DR3 Cepheid SOS membership, using rows with a fundamental-mode period |
| DSCT, GDOR | Zhou catalogue labels; a TESS light curve supplies the searched period and Fourier quantities |
| EB | OGLE eclipsing binaries; published period and OGLE I-band light curve |
| MIRA | OGLE-IV Galactic-bulge Miras; published period and OGLE I-band light curve |
| SRV | OGLE-III Galactic-bulge semiregular variables; primary published period and OGLE I-band light curve |
| ROT | Gaia rotational-modulation catalogue; Gaia period and a TESS light curve |
| RS | Gaia DR3 general-classifier RS candidates; period searched in a TESS light curve |

Gaia supplies the mean photometry, parallax, extinction estimates and variability-summary statistics. The catalogue confidence field is used as a training weight. Some sources supply individual classifier scores; others receive the workflow's fixed catalogue weights. These weights are not uniformly calibrated probabilities of correct classification.

Source services and collections:

- [Gaia Archive](https://gea.esac.esa.int/archive/) and the [ARI Gaia TAP mirror](https://gaia.ari.uni-heidelberg.de/tap).
- [Zhou TESS-validated catalogue on Zenodo](https://zenodo.org/records/13840760).
- [OGLE-IV Galactic-bulge eclipsing binaries](https://www.astrouw.edu.pl/ogle/ogle4/OCVS/blg/ecl/), with Magellanic-Cloud catalogues as fallbacks.
- [OGLE-IV Galactic-bulge long-period variables](https://www.astrouw.edu.pl/ogle/ogle4/OCVS/blg/lpv/) and [OGLE-III Galactic-bulge long-period variables](https://www.astrouw.edu.pl/ogle/ogle3/OIII-CVS/blg/lpv/).
- [MAST](https://mast.stsci.edu/) for TESS light curves and TIC cross-identifiers.

The builder retains the original selection and feature definitions. It requires positive parallax and parallax error, finite required fields, available Gaia extinction/statistics and a fractional parallax error no larger than 1.0. “Complete” therefore describes the configured selection; it does not imply precise distances or uncontaminated light curves. Absolute magnitude uses inverse parallax without distance-error propagation.

Completed candidates and light-curve features are cached. An extraction-configuration fingerprint separates feature caches and checkpoints for the main feature settings. Catalogue files and downloaded light curves are kept separately. If the extraction implementation changes, update `FEATURE_DEFINITION_VERSION` or use a new `CACHE_ROOT` before rebuilding. A new network harvest need not select the identical sample as a previous run; archive contents, returned products, failures and completion order can differ. Preserve a frozen training CSV and its checksum when reproducing a reported model.

The default is one download worker to avoid concurrent library output/connection problems. Network retries are finite, but third-party archive calls may still take a long time or resist immediate cancellation. Interrupt and rerun to resume completed checkpoints. No automatic cache-deletion action is provided; use a new cache location for a fresh run. Inspect partial training counts before training.

## Stage 2: model and evaluation

The estimator retains these settings:

```text
RandomForestClassifier(
    n_estimators=500,
    max_features="sqrt",
    min_samples_leaf=2,
    class_weight="balanced_subsample",
    random_state=42,
    n_jobs=-1,
)
```

Catalogue confidence values supply sample weights. The default requires all nine target classes and at least five finite training examples per class. Invalid weights, duplicate non-missing source IDs and classes with zero total weight are rejected. The notebook prints a stratified 20% holdout report where the sample is large enough, fits the final model on all accepted rows, and writes five-fold out-of-fold predictions for each training star.

These reports measure agreement with the harvested reference labels within this sample. Class-dependent feature extraction remains part of the workflow: for example, the period search is restricted to 0.02–0.35 days for known DSCT training stars, 0.20–6 days for GDOR, and 0.2–27 days for RS. Other classes may use catalogue periods. Random folds do not test a class-agnostic period search on a new survey. Ground-based cadence, passband, errors, aliases and different populations require separate validation.

The default confidence threshold is retained at 0.45. It is an abstention rule on random-forest class probabilities, not a fitted probability calibration or a validated detector of all unfamiliar classes. Confidently incorrect assignments remain possible. The out-of-fold table and per-class probabilities should be inspected alongside independent evidence.

## Stage 3: prediction CSV

The input CSV must have these exact feature names, with finite numeric values in every row:

| Column | Convention retained by the training workflow |
|---|---|
| `logP` | Base-10 logarithm of period in days. |
| `amplitude` | Gaia SOS `peak_to_peak_g` for RRLYR/CEP; 95th minus 5th percentile of magnitudes for the TESS/OGLE light curves. These are different passbands and estimators. |
| `MG` | `G + 5*log10(parallax_mas) - 10 - A_G`, using Gaia mean G magnitude and GSP-Phot extinction. |
| `BP_RP0` | Gaia `bp_rp - ebpminrp_gspphot`. |
| `skewness` | Gaia variability-summary `skewness_mag_g_fov`. |
| `stetson_J` | The workflow's column name for Gaia `stetson_mag_g_fov`; do not assume a differently computed local statistic is equivalent. |
| `delta_wesenheit_cepheid` | Residual defined below from the dereddened `MG`, `BP_RP0` and `logP`. |
| `R21` | Second-harmonic amplitude divided by first-harmonic amplitude. |
| `R31` | Third-harmonic amplitude divided by first-harmonic amplitude. |

The retained Wesenheit-residual definition is:

```text
W = MG - 1.90 * BP_RP0
W_PL = -3.853058319974742 * logP - 1.0689761833751459
delta_wesenheit_cepheid = (W - W_PL) / 1.4891618950080006
```

For Gaia SOS sources the Fourier ratios come from the published G-band parameters. The other classes use a weighted three-harmonic magnitude fit at the adopted period. These source-specific definitions are retained for compatibility with existing models; they should not be described as survey-independent features.

Optional identifier columns are `star_id`, `source_id` or `gaia_id`. Identifiers are read as strings to preserve their digits. Without an identifier, a row-based ID is assigned. Prediction writes `star_id`, `gaia_id`, `star_type`, `confidence_score`, `recognised`, `best_guess_type`, and one `prob_<CLASS>` column per class in the model. A row below the threshold receives `UNKNOWN`, while its best guess and probabilities remain available. `PREDICTION_CONFIDENCE_OVERRIDE` can override the threshold saved in the bundle.

The companion photometry project produces scientific measurements, but this classifier does not automatically join its tables or convert them into the nine-feature CSV. Its phase-binned amplitude, locally computed variability statistics or observed-photometry Wesenheit value must not be substituted silently for the training conventions above. Prepare a consistent feature table and assess transfer to that dataset before interpreting model labels. No raw-light-curve feature-preparation stage is included here.

## Numerical compatibility

Cleanup retains the original feature formulae, period ranges, forest settings and default threshold. It changes configuration, file locations, cache handling, dependency checks and invalid-input guards. The Cepheid `label_source` metadata now correctly identifies the dedicated SOS catalogue; that correction does not change a feature or a training weight.

Against the existing frozen training table, the cleaned training stage reproduces the saved 21,666-row out-of-fold table and holdout report. With the existing model and 90-row prediction input, assigned labels and identifiers are unchanged, and class probabilities agree to floating-point precision. These checks cover training and prediction using those inputs. The cleaned network-harvesting stage has not been rerun in full.

## License

This project is distributed under the BSD 3-Clause License.
See [LICENSE](LICENSE) for details.

## Archived release

Version 1.0.0 is archived on Zenodo:
[View the archived release](https://doi.org/10.5281/zenodo.23597239).
