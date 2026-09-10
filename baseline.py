"""Backward-compatible alias for the current calibration builder.

The previous version emitted a pairwise-distance distribution but ``dtw.py``
never consumed it. ``build_calibration.py`` now generates the actual
leave-one-out top-k distribution used during evaluation.
"""

from sign_eval.calibration import main


if __name__ == "__main__":
    main()
