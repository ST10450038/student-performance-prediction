"""Generate a synthetic student dataset for the performance prediction project."""
from pathlib import Path

import numpy as np
import pandas as pd


def generate(n: int = 1000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    study_hours = np.clip(rng.normal(12, 5, n), 0, 40)          # per week
    attendance = np.clip(rng.normal(80, 12, n), 30, 100)        # %
    prev_grade = np.clip(rng.normal(60, 14, n), 20, 100)        # %
    assignments = np.clip(rng.normal(70, 15, n), 0, 100)        # avg %
    sleep_hours = np.clip(rng.normal(7, 1.2, n), 3, 10)
    extracurricular = rng.integers(0, 4, n)                     # activities
    internet_access = rng.choice([0, 1], n, p=[0.15, 0.85])
    parental_support = rng.choice([0, 1, 2], n, p=[0.2, 0.5, 0.3])  # low/med/high

    score = (
        0.30 * prev_grade
        + 0.25 * assignments
        + 0.15 * attendance
        + 0.9 * study_hours
        + 1.2 * (sleep_hours - 7)
        + 2.0 * internet_access
        + 1.5 * parental_support
        - 0.8 * np.maximum(extracurricular - 2, 0)
        + rng.normal(0, 4, n)
    )
    final_mark = np.clip(score - 5, 0, 100)

    df = pd.DataFrame(
        {
            "study_hours_per_week": study_hours.round(1),
            "attendance_pct": attendance.round(1),
            "previous_grade": prev_grade.round(1),
            "assignment_avg": assignments.round(1),
            "sleep_hours": sleep_hours.round(1),
            "extracurricular_activities": extracurricular,
            "internet_access": internet_access,
            "parental_support": parental_support,
            "final_mark": final_mark.round(1),
        }
    )
    df["result"] = np.where(df["final_mark"] >= 50, "Pass", "Fail")
    return df


if __name__ == "__main__":
    out = Path(__file__).parent / "data" / "students.csv"
    out.parent.mkdir(exist_ok=True)
    generate().to_csv(out, index=False)
    print(f"Saved {out}")
