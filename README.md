# HandSign

Practice evaluation for Vietnamese sign language. The project makes three
separate decisions so a weak motion score is never presented as a wrong sign:

1. `recognizer.py` recognizes the sign label from a video. A matching label is
   required before form scoring; confidence and margin are diagnostic only.
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
  framing.py         Body framing guidance and pre-recording readiness checks
  desktop.py         Target selection, processing window, score/feedback display
  sample_player.py   Optional demo playback inside the lesson window
  samples.py         Finds label-matched local videos and checks decoding
  paths.py           Checkout-relative asset and output paths
record.py            Webcam capture followed by automatic desktop evaluation
build_calibration.py Creates artifacts/label_calibration.json
build_validation.py  Fits optional thresholds from labelled user clips
evaluate.py          Main evaluation command
recognizer.py        VideoMAE label recognition required for practice scores
data/landmarks (1)/ Reference landmark corpus
artifacts/reference_landmarks.npz  Bundled references for fresh clones
sample_videos/      One portable demonstration video for each label
result2/info.npy    Author's reference-reference DTW cache (only for rebuilding)
```

`dtw.py`, `baseline.py`, and `Video-Text.py` remain as lightweight
backwards-compatible commands. `Value_threshold.ipynb` is an older exploratory
notebook and is not part of the evaluation pipeline.

## Setup

Use 64-bit Python 3.10–3.12; Python 3.11 is a suitable default. Install a new
environment on each machine: do not copy `.venv` from someone else's computer.
On Windows, after cloning and entering the repository:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\setup.ps1
powershell -ExecutionPolicy Bypass -File .\run.ps1
```

The setup script creates `.venv-handsign`, installs `requirements.txt`, and
checks model assets and the bundled reference pack. `run.ps1` always uses that
environment. Paths are anchored to the project directory, so the app can also
be launched from another working directory. No `C:\HandSign` or user-specific
paths are required. To run with a preselected word:

```powershell
powershell -ExecutionPolicy Bypass -File .\run.ps1 --target-label "Bế mạc"
```

For a manual setup (including Linux/macOS), create a virtual environment,
install the requirements, and run `python scripts/check_setup.py` followed by
`python record.py` using that environment. Linux may need the OS `python3-tk`
and OpenCV desktop libraries; available dependency wheels depend on OS/CPU.
The chosen Python range is supported by [MediaPipe](https://pypi.org/project/mediapipe/0.10.21/).
The installed versions are bounded to avoid unintended major API upgrades.
OpenCV handles both recognition video loading and playback; Decord is no longer
required. Pillow is declared explicitly for the embedded player.

Normal use **does not require downloading the demo videos or rebuilding
calibration**. The checked-in calibration and `artifacts/reference_landmarks.npz`
contain the 2,047 calibrated reference sequences (~14 MiB compressed). When the
original landmark folders are absent, evaluation reads this pack. The pack is
checked against the calibration checksum to reject mismatched data. It contains
no user recordings and no demonstration videos.

Internet is needed to install dependencies and download the recognition model
from Hugging Face on the first evaluation. Subsequent evaluations reuse its
cache. A desktop session, camera access, and sufficient memory for the model
are still needed; hardware/driver support cannot be guaranteed on every machine.

### Rebuilding reference data (maintainers only)

Only rebuild if the reference corpus/scoring calibration changes. Obtain the
original landmark corpus and its matching pairwise cache first, then run:

```powershell
python build_calibration.py
python scripts/package_references.py
```

Commit the updated calibration **and** reference pack together. Optional video
datasets, virtual environments and generated user recordings stay ignored.

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
```

Type a prefix in the target selector: `B` suggests words starting with B, and
`Bế` or `be` narrows the list. Click a suggestion to fill the entry; the Down
key and Enter also select suggestions. Choose **Xem động tác mẫu** to view the
lesson, then **Luyện tập với AI** to open the camera. **Chọn từ khác** returns
to the target selector without restarting `record.py`. You can
also preselect it: `python record.py --target-label "Bế mạc"`.
After recording, press **Q** (or close the camera window). The camera closes;
an **AI đang xử lí…** window appears while the app saves the matching video and
landmarks, recognizes the sign and evaluates it. The result window shows the
target, predicted label, total score, regional score/weight/distance/threshold,
feedback and a tab with tracking and comparison diagnostics. Unscored attempts
show the retry explanation instead of a numeric score. **Quay lại động tác**
starts another recording of the same target; **Đóng** exits.

### Optional demonstration videos

The lesson shows **Động tác mẫu**, the selected label, and a central video
player with pause/replay controls. A clip is displayed only after opening and
decoding its first frame succeeds. Missing, inaccessible or unreadable videos
show **Hiện không có video**; **Luyện tập với AI** remains available. Videos are
silent demonstrations and are independent of the references used for scoring.

The project bundles one representative clip for each of its 100 labels in
`sample_videos/<label>.mp4`, so the lesson works after copying the whole
project to another machine. The configured root also supports
`<label>/*.(mp4|avi|mov|mkv|webm)`. Exact label matching ignores casing and
Unicode composition, without guessing labels from numeric filenames. To use a
different video collection, run:

```powershell
python record.py --sample-video-root "D:\SignVideos"
```

This is an optional example path, not a path embedded in the program.
`HANDSIGN_SAMPLE_VIDEO_ROOT` is an equivalent environment override; the CLI
option takes priority. No absolute machine-specific path is stored.

The desktop UI uses Python's Tkinter (included with standard Windows Python).
Model work runs in a background thread and all UI updates run on the main
thread. Errors appear in the result window; cancelled/invalid recordings never
evaluate an older saved clip. Results still save to `data/user/evaluation.json`.

For terminal-only evaluation of an existing recording:

```powershell
python evaluate.py --target-label "Bế mạc" --video data/user/video_user/trimmed_video.mp4
```

The capture command writes both a legacy-compatible
`data/user/sequence_user/sequence.npy` and a mask-aware
`data/user/sequence_user/record.npz`. The second format lets the evaluator
report tracking coverage and warn when missing landmarks may affect the score.

Supply a video and landmark sequence from the **same recording**. To evaluate
files outside the default capture directory:

```powershell
python evaluate.py --target-label "Bế mạc" --sequence path/to/record.npz --video path/to/video.mp4
```

Before capture, `record.py` checks the face with `face_landmarker.task` and the
two shoulder points with `pose_landmarker_lite.task`. Adjust the camera until
your face and both shoulders are visible: the border turns green and shows
"GOC CAMERA OK - Nhan S de quay". Press **S** while green for a **3-second
countdown**, then perform the sign when "DANG QUAY" appears. Losing face or
shoulder visibility during countdown cancels it; adjust and press S again.
There is no extra stability hold timer. Pressing S before the preview is ready
does not queue a recording. Repeated S does not reset an active countdown.
Press **Q** to stop or exit. Guidance uses Vietnamese without diacritics to
work with OpenCV's built-in font. Text appears in separate panels above and
below the camera image. Overlays are never saved into the model video.

The preflight only requires a detected face and two finite, in-image shoulder
points with visibility/presence >= 0.60. Hips, hands, torso size, body centering
and camera distance are not start requirements. The border indicates readiness,
not a required body size. During signing, framing remains advisory;
it does not cut the motion, discard frames or block the eventual score.
See the [MediaPipe Pose guide](https://developers.google.com/edge/mediapipe/solutions/vision/pose_landmarker/python)
for landmark coordinates and visibility/presence outputs.

When manually labelled correct/incorrect recordings exist for a label, the
following command can still fit a global DTW threshold for analysis. These
legacy validation thresholds do not control the regional practice score.

```powershell
python build_validation.py --label "Cảm ơn" --correct-dir data/eval_set/correct --incorrect-dir data/eval_set/incorrect
```

Results are written to `data/user/evaluation.json`. The relevant fields are:

If the predicted label differs from the target, the result is `incorrect_label`
and `form.score` is null, regardless of confidence. Feedback asks the learner
to repeat the target sign; no regional scores are generated. If the names match
(ignoring case, extra whitespace and Unicode composition differences), form
scoring proceeds. Confidence and margin do not gate scoring.

Without `--video`, or when recognition returns no usable label, the result is
`needs_label_confirmation` and `form.score` is null. The same rule applies to
direct `SignEvaluator.evaluate` callers and the legacy `dtw.py` wrapper.
Use `evaluate.py --video ...` for scored practice. These gates do not change
the regional score formula or make low hand-tracking coverage a blocker.

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
- `form.decision_source`: `weighted_region_thresholds` for a score,
  `insufficient_region_calibration` when regional calibration is insufficient,
  or `recognition_mismatch`, `recognition_not_run`, `recognition_uncertain`
  when recognition prevents scoring.
- `feedback`: per-region distance, label-specific threshold, and the most
  problematic time range of the motion.

Use `--rebuild-calibration` after adding or removing reference files.
User-to-reference DTW runs only after a matching label is confirmed. The diagnostic reference
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

## Tests

```powershell
python -X utf8 -B -m unittest discover -s tests -v
```

UTF-8 mode supports Vietnamese test output on Windows terminals. Automated
tests cover prefix suggestions, missing/unreadable videos, a relocated checkout
without datasets, recognition gating, countdown, capture/evaluation flow and
desktop results using simulated poses and model outputs;
actual webcam framing and recognition accuracy require real recordings.
