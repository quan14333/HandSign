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
  calibration.py     Builds per-label leave-one-out calibration data
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

## Capture and evaluate

```powershell
python record.py
python evaluate.py --target-label "Cảm ơn"
```

The capture command writes both a legacy-compatible
`data/user/sequence_user/sequence.npy` and a mask-aware
`data/user/sequence_user/record.npz`. The second format lets the evaluator
reject low-quality tracking rather than treating missing landmarks as a pose.

To also verify that the user performed the requested label:

```powershell
python evaluate.py --target-label "Cảm ơn" --video data/user/video_user/trimmed_video.mp4
```

When manually labelled correct/incorrect recordings exist for a label, fit a
camera-aware practice threshold. This is especially useful because the corpus
and a webcam user can have different framing and signing styles.

```powershell
python build_validation.py --label "Cảm ơn" --correct-dir data/eval_set/correct --incorrect-dir data/eval_set/incorrect
```

Results are written to `data/user/evaluation.json`. The relevant fields are:

- `status`: `correct`, `incorrect_label`, `right_label_needs_practice`, or
  `needs_label_confirmation`.
- `tracking`: landmark detection quality. The score is blocked when hands were
  missing for too much of a new recording.
- `form.score`: a label-specific leave-one-out percentile, not a global DTW
  threshold. Higher is closer to the reference set.
- `form.decision_source`: records whether the verdict used a labelled user
  validation threshold or only the reference corpus distribution.
- `feedback`: per-region distance, label-specific threshold, and the most
  problematic time range of the motion.

Use `--rebuild-calibration` after adding or removing reference files.
