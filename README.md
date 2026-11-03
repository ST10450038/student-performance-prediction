# Student Performance Prediction System

A machine learning project that predicts a student's **final mark** (regression) and whether they will **pass or fail** (classification) from study habits and background factors. It can be used to flag at-risk students early.

## Features
- Synthetic dataset generator (1,000 students, 8 features)
- Regression: Linear Regression vs Random Forest
- Classification: Random Forest pass/fail model
- Feature importance and predicted-vs-actual charts
- Command-line predictor for a single student

## Tech Stack
Python, pandas, NumPy, scikit-learn, matplotlib, joblib

## Project Structure
```
├── generate_data.py   # creates data/students.csv
├── train.py           # trains, evaluates, saves models + charts
├── predict.py         # CLI prediction for one student
├── data/              # dataset
├── outputs/           # charts
├── models/            # saved models (git-ignored)
└── requirements.txt
```

## Getting Started
```bash
git clone <your-repo-url>
cd student-performance-prediction
pip install -r requirements.txt
python generate_data.py
python train.py
python predict.py --study-hours 15 --attendance 85 --previous-grade 65 --assignment-avg 72
```

## Features Used
| Feature | Description |
|---|---|
| study_hours_per_week | Weekly study time |
| attendance_pct | Class attendance % |
| previous_grade | Previous term average % |
| assignment_avg | Assignment average % |
| sleep_hours | Average nightly sleep |
| extracurricular_activities | Number of activities |
| internet_access | 0 = no, 1 = yes |
| parental_support | 0 low, 1 medium, 2 high |

## Results
| Task | Model | Score |
|---|---|---|
| Final mark | Linear Regression | R² 0.76, MAE 3.3 marks |
| Final mark | Random Forest | R² 0.70, MAE 3.7 marks |
| Pass/Fail | Random Forest | Accuracy 86.5% |

Charts are saved in `outputs/`.

## Note
The dataset is synthetic, generated for learning and demonstration. To use real data, replace `data/students.csv` with a file that has the same columns (for example, the UCI Student Performance dataset after mapping columns).

## Author
Sandisiwe Seluleko Manyathi — Diploma in Software Development, IIE Rosebank College
