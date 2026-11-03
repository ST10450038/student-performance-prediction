"""Predict a single student's final mark and pass/fail result.

Example:
    python predict.py --study-hours 15 --attendance 85 --previous-grade 65 \
        --assignment-avg 72 --sleep-hours 7 --activities 1 --internet 1 --support 2
"""
import argparse
from pathlib import Path

import joblib
import pandas as pd

MODELS = Path(__file__).parent / "models"


def main() -> None:
    p = argparse.ArgumentParser(description="Student performance predictor")
    p.add_argument("--study-hours", type=float, required=True, help="hours per week")
    p.add_argument("--attendance", type=float, required=True, help="percent")
    p.add_argument("--previous-grade", type=float, required=True, help="percent")
    p.add_argument("--assignment-avg", type=float, required=True, help="percent")
    p.add_argument("--sleep-hours", type=float, default=7)
    p.add_argument("--activities", type=int, default=1)
    p.add_argument("--internet", type=int, choices=[0, 1], default=1)
    p.add_argument("--support", type=int, choices=[0, 1, 2], default=1,
                   help="parental support: 0 low, 1 medium, 2 high")
    a = p.parse_args()

    reg = joblib.load(MODELS / "regressor.joblib")
    clf = joblib.load(MODELS / "classifier.joblib")

    row = pd.DataFrame([{
        "study_hours_per_week": a.study_hours,
        "attendance_pct": a.attendance,
        "previous_grade": a.previous_grade,
        "assignment_avg": a.assignment_avg,
        "sleep_hours": a.sleep_hours,
        "extracurricular_activities": a.activities,
        "internet_access": a.internet,
        "parental_support": a.support,
    }])[reg["features"]]

    mark = float(reg["model"].predict(row)[0])
    prob = float(clf["model"].predict_proba(row)[0][1])
    print(f"Predicted final mark: {mark:.1f}%")
    print(f"Pass probability:     {prob:.0%}")
    print(f"Prediction:           {'PASS' if prob >= 0.5 else 'AT RISK / FAIL'}")


if __name__ == "__main__":
    main()
