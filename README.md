# Eccentricity confound in EEG-based visual attention decoding

Python code for the paper *Eccentricity Confound in EEG-based Visual Attention Decoding from Gaze-Fixated Neural Tracking of Motion in Natural Videos* [1] (Accepted by Journal of Neural Engineering).

## Experimental protocol

14 subjects (`Subj_1` … `Subj_15`, subject 12 excluded because of an eye-tracker failure) watched 7 single-shot videos of 180 s each, at 30 Hz, under three conditions:

| Task | Condition | Attention |
| --- | --- | --- |
| 1 | ignore, eccentric | away from the moving object, which appears in the periphery |
| 2 | attend, eccentric | on the moving object, which appears in the periphery |
| 3 | attend, central | on the moving object, which appears at the fixation point |

In all three tasks the participants keep fixating a cross in the center of the screen for the whole
video, so gaze position is held constant and only the locus of covert attention and the eccentricity
of the moving object change. EEG, EOG and eye-tracker gaze are recorded simultaneously.

## Data and processing pipeline

Raw recordings live outside the repository (`Experiments/data/SOMove_MultiTask/`); the code caches
one pickle per task in `data/` (`data_task_{1,2,3}.pkl`), which `utils.load_subj` reads.

* **EEG** — `utils.preprocessing`: MNE-based, `biosemi64` montage, bad-channel interpolation,
  average re-reference, 0.5 Hz high-pass, 50 Hz notch, resample to 30 Hz, per-minute
  normalization. EOG is then regressed out of the EEG (`utils.regress_out`), giving the `EEG-EOG`
  modality used throughout; the plain `EEG` modality (no regression) is kept for comparison.
* **Stimulus** — object-based optical flow (column 8 of the precomputed feature files), NaN
  interpolated and envelope-smoothed by `utils.clean_features`.
* **Gaze** — x/y coordinates plus saccade and blink flags. `utils.fixation_cluster` (DBSCAN) finds
  the dominant fixation cluster; `utils.get_mask_from_gaze` marks samples that fall outside it, or
  that are saccades, as invalid. `utils.create_event_masks` additionally masks the cross/circle
  events; it reads their frame indices from `mounted_videos_info.csv` (tasks 1 and 2) and
  `mounted_videos_info_3.csv` (task 3), which will be provided together with the dataset published
  on Zenodo. Masked samples are dropped only from the test data, so the eccentricity condition is
  not contaminated by gaze shifts.

## Analyses

`algo.py` holds the correlation analysis functions, `utils.py` the pre-processing, statistics and plotting helpers.

* `CanonicalCorrelationAnalysis` — stimulus–EEG CCA with (spatio-)temporal filters
  (`L_EEG = 3`, `offset_EEG = 1`, `L_Stim = 15`). Evaluated by **match–mismatch** classification on
  45 s trials with leave-one-video-out cross-validation (7 folds), bootstrapped trial start points,
  and a binomial significance level; `permutation_test` gives the chance-level bounds.
  `mm_blocks` / `mm_blocks_global_mismatch` run the same test per spatial block.
* `GeneralizedCCA` — group analysis (GCCA / CorrCA) across subjects, yielding inter-subject
  correlation (ISC) per task and the corresponding forward models.
* `BlockCCA` — exploratory model with filters shared per view set [Not used in the paper].

Statistics: Wilcoxon signed-rank tests with rank-biserial effect sizes across tasks
(`utils.wilcoxon_effect`), FDR-corrected. 

## Notebooks

* **`Analysis_SO_MultiTask.ipynb`** — the main analysis. Loads and checks the multi-task data
  (EOG/gaze alignment), reports gaze-eccentricity statistics in degrees of visual angle, runs the
  per-subject CCA match–mismatch analysis across the three tasks, the grand-average topoplots, and
  the GCCA/ISC group analysis. The *CCA-Block Analysis* section tests whether coupling strength
  varies with stimulus eccentricity; those results were inconclusive and are **not** part of the
  manuscript.
* **`Compare.ipynb`** — compares the new protocol against the dataset of the first protocol [2]
  (19 subjects, `data/1stprotocol/`), for both match–mismatch accuracy and ISC, with and without
  EOG regression. The first-protocol recordings are temporally re-aligned to the current stimulus
  features before the comparison.

Results are written to `tables/<MOD>/` (CSV and pickle, one file per run) and figures to
`figures/`.

## Reference

[1] Yao, Y., González, C. S., Geirnaert, S., Gillebert, C. R., Tuytelaars, T., & Bertrand, A. (2026). Eccentricity Confound in EEG-based Visual Attention Decoding from Gaze-Fixated Neural Tracking of Motion in Natural Videos. arXiv preprint arXiv:2604.15223.

[2] Yao, Y., Stebner, A., Tuytelaars, T., Geirnaert, S., & Bertrand, A. (2024). Identifying temporal correlations between natural single-shot videos and EEG signals. Journal of Neural Engineering, 21(1), 016018.
