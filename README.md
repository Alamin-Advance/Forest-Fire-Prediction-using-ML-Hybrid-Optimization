# Forest-Fire-Prediction-using-ML-Hybrid-Optimization

🔥 Overview
This repository explores machine learning and hybrid optimization techniques for forest fire (FF) prediction, comparing:

Random Forest (RF)

PSO-Optimized Random Forest (RF-PSO)

Long Short-Term Memory (LSTM)

🔹 Key Findings:
✅ RF & RF-PSO outperform LSTM in accuracy, precision, and AUC-ROC.
✅ LSTM excels in recall, making it better for reducing false negatives in fire detection.
✅ PSO optimization improves RF marginally, but standard RF remains highly robust.
✅ Statistical tests (Wilcoxon, Friedman, Holm-corrected Wilcoxon) confirm significance.

📊 Performance Metrics (Avg. over 100 Trials)
Model	Accuracy	Precision	Recall	F1-Score	AUC-ROC	RMSE
RF	
RF-PSO	
LSTM	
(Bold = Best performance in metric)

⚙️ Methodology
1. Models Compared
Random Forest (RF) – Ensemble of decision trees for structured environmental data.

RF-PSO – Hyperparameter-optimized RF using Particle Swarm Optimization.

LSTM – Deep learning model for sequential data (tested despite limited temporal resolution).

2. Dataset Features
Meteorological (temperature, humidity, wind speed)

Environmental (soil moisture, vegetation index)

Historical fire occurrence records

3. Evaluation Approach
100 independent trials for statistical robustness.

Metrics: Accuracy, Precision, Recall, F1, AUC-ROC, RMSE.

Statistical Tests:

Paired t-test

Wilcoxon Signed-Rank

Friedman Test

Holm-corrected Wilcoxon

📂 Repository Structure
text
├── /data/                  # Forest fire dataset (CSV)  
├── /notebooks/             # Jupyter notebooks (EDA, training)  
│   ├── EDA.ipynb  
│   ├── RF_PSO_Training.ipynb  
│   └── LSTM_Training.ipynb  
├── /src/  
│   ├── model_training.py   # RF, RF-PSO, LSTM training scripts  
│   ├── pso_optimizer.py    # PSO hyperparameter tuning  
│   └── utils.py            # Data preprocessing & metrics  
├── /results/               # Performance logs & plots  
├── requirements.txt        # Python dependencies  
└── README.md  
🚀 Installation & Usage
1. Clone & Install Dependencies
bash
git clone https://github.com/yourusername/forest-fire-prediction.git  
cd forest-fire-prediction  
pip install -r requirements.txt  
2. Run Experiments
Train RF / RF-PSO:

bash
python src/model_training.py --model rf_pso --trials 100
Train LSTM:

bash
python src/model_training.py --model lstm --epochs 50
3. Evaluate Results
python
from src.utils import evaluate_model
metrics = evaluate_model("RF-PSO", test_data)  
📜 Citation
If this work aids your research, please cite:

bibtex
@article{FFPrediction2024,
  title = {Forest Fire Prediction Using PSO-Optimized Random Forest vs. LSTM},
  author = {Hossain},
  year = {2024},
  journal = {},
  url = {https://github.com/Alamin-Advance/forest-fire-prediction}
}
🔍 Key Takeaways for Practitioners
✔ Use RF/RF-PSO for balanced performance (accuracy, precision).
✔ Prefer LSTM if minimizing false negatives (high recall) is critical.
✔ PSO optimization offers marginal gains but adds computational cost.
✔ Statistical testing is crucial—don’t rely solely on raw metrics!

🤝 Contributing & Contact
Issues/Pull Requests: Welcome!

Contact: alaminh1411@gmail.com

🌲 Built for Environmental Safety | Python + Scikit-learn + TensorFlow 🌲
