"""Train and evaluate models that predict student final marks (regression)
and pass/fail outcome (classification)."""
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (accuracy_score, classification_report,
                             mean_absolute_error, r2_score)
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).parent
DATA = ROOT / "data" / "students.csv"
MODELS = ROOT / "models"
OUTPUTS = ROOT / "outputs"
MODELS.mkdir(exist_ok=True)
OUTPUTS.mkdir(exist_ok=True)


def main() -> None:
    if not DATA.exists():
        from generate_data import generate

        DATA.parent.mkdir(exist_ok=True)
        generate().to_csv(DATA, index=False)

    df = pd.read_csv(DATA)
    features = [c for c in df.columns if c not in ("final_mark", "result")]
    X = df[features]

    # ---- Regression: predict final mark ----
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, df["final_mark"], test_size=0.2, random_state=42
    )
    results = {}
    for name, model in {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=200, random_state=42),
    }.items():
        model.fit(X_tr, y_tr)
        pred = model.predict(X_te)
        results[name] = (model, r2_score(y_te, pred), mean_absolute_error(y_te, pred))

    print("=== Final mark prediction ===")
    for name, (_, r2, mae) in results.items():
        print(f"{name:18s} R2={r2:.3f}  MAE={mae:.2f} marks")

    best_name = max(results, key=lambda k: results[k][1])
    best_reg = results[best_name][0]
    joblib.dump({"model": best_reg, "features": features}, MODELS / "regressor.joblib")
    print(f"Best regressor: {best_name}")

    # ---- Classification: Pass / Fail ----
    y_cls = (df["result"] == "Pass").astype(int)
    Xc_tr, Xc_te, yc_tr, yc_te = train_test_split(
        X, y_cls, test_size=0.2, random_state=42, stratify=y_cls
    )
    clf = RandomForestClassifier(n_estimators=200, random_state=42)
    clf.fit(Xc_tr, yc_tr)
    pred_c = clf.predict(Xc_te)
    print("\n=== Pass/Fail classification ===")
    print(f"Accuracy: {accuracy_score(yc_te, pred_c):.3f}")
    print(classification_report(yc_te, pred_c, target_names=["Fail", "Pass"]))
    joblib.dump({"model": clf, "features": features}, MODELS / "classifier.joblib")

    # ---- Feature importance chart ----
    rf = results["Random Forest"][0]
    imp = pd.Series(rf.feature_importances_, index=features).sort_values()
    ax = imp.plot.barh(figsize=(8, 5), color="#2563eb")
    ax.set_title("What drives student performance? (Random Forest)")
    ax.set_xlabel("Feature importance")
    plt.tight_layout()
    plt.savefig(OUTPUTS / "feature_importance.png", dpi=150)
    plt.close()

    # ---- Predicted vs actual chart ----
    pred = best_reg.predict(X_te)
    plt.figure(figsize=(6, 6))
    plt.scatter(y_te, pred, alpha=0.5, color="#2563eb")
    plt.plot([0, 100], [0, 100], "r--")
    plt.xlabel("Actual final mark")
    plt.ylabel("Predicted final mark")
    plt.title(f"Predicted vs actual ({best_name})")
    plt.tight_layout()
    plt.savefig(OUTPUTS / "predicted_vs_actual.png", dpi=150)
    plt.close()
    print(f"\nSaved models to {MODELS} and charts to {OUTPUTS}")


if __name__ == "__main__":
    main()
