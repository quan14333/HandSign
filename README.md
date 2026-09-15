# HandSign

Practice evaluation for Vietnamese sign language. The project makes three
separate decisions so a weak motion score is never presented as a wrong sign:

1. `recognizer.py` optionally recognizes the sign label from a video.
2. `sign_eval` checks landmark tracking quality and scores form against the
   requested label.
3. The evaluator returns label-specific, time-localized feedback for each hand
   and the face.

## Project layout

```text
sign_eval/
  landmarks.py       Landmark validation, normalization, DTW, and tracking checks
  calibration.py     Builds per-label distributions and regional thresholds
  evaluator.py       Structured semantic, form, and feedback evaluation
record.py            Webcam capture; writes sequence.npy and record.npz
build_calibration.py Creates artifacts/label_calibration.json
build_validation.py  Fits optional thresholds from labelled user clips
evaluate.py          Main evaluation command
recognizer.py        Optional VideoMAE label recognition
data/landmarks (1)/ Reference landmark corpus
result2/info.npy    Cached reference-reference DTW pairs
```

`dtw.py`, `baseline.py`, and `Video-Text.py` remain as lightweight
backwards-compatible commands. `Value_threshold.ipynb` is an older exploratory
notebook and is not part of the evaluation pipeline.

## Setup

```powershell
python -m pip install -r requirements.txt
python build_calibration.py
```

The calibration builder removes invalid all-zero sequences and exact duplicates
before generating the score distributions. It uses `result2/info.npy`, so run
it again whenever the reference corpus changes.

Calibration format 3 also saves `required_regions`, `region_thresholds`, and
`region_threshold_sample_count` for every label in `artifacts/label_calibration.json`.
The builder computes the same P90 regional limits from the selected reference
pairs that evaluation previously computed on demand. Evaluation reads these
limits directly; it does not recompute them for a new user recording. Labels
with fewer than three selected pairs store null thresholds and remain unscored.

After upgrading from format 2, run `python build_calibration.py` once. Old files
or missing/inconsistent regional thresholds produce a rebuild instruction.
Rebuild after changing the reference data, normalization, or distance algorithm;
the pair cache in `result2/info.npy` must also match those references and distances.
The builder uses that cache, but does not regenerate it.

## Capture and evaluate

```powershell
python record.py
python evaluate.py --target-label "Cảm ơn"
```

The capture command writes both a legacy-compatible
`data/user/sequence_user/sequence.npy` and a mask-aware
`data/user/sequence_user/record.npz`. The second format lets the evaluator
report tracking coverage and warn when missing landmarks may affect the score.

To also verify that the user performed the requested label:

```powershell
python evaluate.py --target-label "Cảm ơn" --video data/user/video_user/trimmed_video.mp4
```

When manually labelled correct/incorrect recordings exist for a label, the
following command can still fit a global DTW threshold for analysis. These
legacy validation thresholds do not control the regional practice score.

```powershell
python build_validation.py --label "Cảm ơn" --correct-dir data/eval_set/correct --incorrect-dir data/eval_set/incorrect
```

Results are written to `data/user/evaluation.json`. The relevant fields are:

If recognition predicts a different target label with confidence >= 70%, the
result is `incorrect_label`: form score is null and feedback explains the label
mismatch instead of grading regions. A different label below 70% still allows
practice scoring with a recognition warning and `needs_label_confirmation`.
Matching labels retain the existing confidence >= 50% and margin >= 12% checks
(when margin is provided). Without `--video`, recognition is not run.

- `status`: `correct`, `incorrect_label`, `right_label_needs_practice`, or
  `needs_label_confirmation`; `not_scored` for all-zero landmark data or
  insufficient regional calibration.
- `tracking`: landmark detection quality. All detection percentages, including
  0%, allow scoring of otherwise valid landmark sequences. Low coverage remains
  visible as `low_quality` and/or a feedback warning; it does not block scores
  or regional feedback. Missing landmarks can still affect the computed score.
- `form.score`: a weighted regional practice score (0–100), using the same
  distances and thresholds as feedback. Active hands share 80% and face gets
  20%; a single active hand gets the full 80%. If face is excluded, hands share
  100%. No score is returned if a required region lacks calibration.
- `form.region_scores`, `form.region_weights`: the components of the weighted
  score. Each regional feedback item also includes `score` and `weight`.
- `form.reference_percentile`, `form.comparison_distance`: DTW diagnostics,
  retained for reference only; neither determines the practice score.
- `form.required_regions`, `form.ignored_regions`, and `form.hand_mode`: explain
  whether the label was evaluated as a one-hand or two-hand sign. The builder
  saves the inferred `required_regions` with thresholds for exactly those regions.
  Automatic inference includes a hand when its median presence across reference
  clips is at least 10% (previously 20%). Custom region selections need matching
  thresholds recomputed with `calculate_region_thresholds`; do not change the
  region list alone.
- `form.decision_source`: `weighted_region_thresholds` for the new score, or
  `insufficient_region_calibration` when no regional score can be computed.
- `feedback`: per-region distance, label-specific threshold, and the most
  problematic time range of the motion.

Use `--rebuild-calibration` after adding or removing reference files.
User-to-reference DTW still runs for every recording. The diagnostic reference
distribution for a non-default region selection may also be computed at runtime;
that distribution is separate from the saved regional scoring thresholds.

## Regional practice score

For each required region, let `r = distance / threshold`:

```text
r <= 0.3:       regional score = 100
0.3 < r <= 1:   regional score = 100 - 30 * (r - 0.3) / 0.7
r > 1:         regional score = 70 * 2^(1 - r)
form.score = sum(regional score * regional weight)
```

If a threshold is zero, an exact match gets 100 and a nonzero distance gets 0.
Regional scores are rounded to two decimals before aggregation; the total is
also rounded to two decimals. Status uses that displayed total: `excellent`
at 85+, `good` at 70+, `needs_practice` at 30+, otherwise `far_from_reference`.
Low sample counts remain visible in calibration quality and feedback warnings.
These are practice-score design choices, not probabilities of correctness.
Individual regions can still need attention when the weighted total is good.
