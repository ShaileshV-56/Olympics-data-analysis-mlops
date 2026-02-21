# 🏅 Olympic Athlete Performance Prediction Engine

**A production-ready MLOps project for analyzing Olympic athlete data and predicting medal outcomes using machine learning.**

Combines data analysis, interactive visualizations, and a trained Random Forest classifier to predict whether an athlete will win a medal based on age, height, weight, sport, event, team, and Olympic year.

## Features

- 🎯 **Medal Prediction**: Binary classification model (medal/no medal) with ~90% accuracy
- 📊 **Interactive Dashboards**: Streamlit web app with multiple analysis pages
- 🏆 **Multi-page UI**:
  - Index: Historical medal tallies and participation trends
  - Medal Prediction: Manual entry or CSV batch prediction with probabilities
- 🐳 **Production Ready**: Docker support, CI/CD pipeline, and Streamlit server configuration
- 📈 **Comprehensive Analytics**: Country analysis, athlete statistics, participation trends

## Tech Stack

| Component | Technology |
|-----------|-----------|
| **Frontend** | Streamlit |
| **ML Framework** | scikit-learn (Random Forest) |
| **Data Processing** | pandas, NumPy |
| **Visualization** | Plotly, Matplotlib, Seaborn |
| **Deployment** | Docker, GitHub Actions |
| **Language** | Python 3.9 |

## Project Structure

```
Olympics-data-analysis-mlops/
├── app.py                           # Streamlit main page (index)
├── pages/
│   └── 01_Medal_Prediction.py      # Prediction page with manual/batch modes
├── data/
│   ├── athlete_events.csv          # Complete Olympic dataset (~272K rows)
│   └── noc_regions.csv             # Country-region mappings
├── src/
│   ├── train.py                    # Model training with 8 features
│   ├── predict.py                  # Inference module with category filters
│   ├── preprocessor.py             # Data preprocessing utilities
│   └── helper.py                   # Analytics helper functions
├── models/
│   └── model.pkl                   # Trained Random Forest (88 features after encoding)
├── scripts/
│   └── ensure_requirements.py       # CI helper to ensure streamlit in requirements
├── .github/workflows/
│   └── ci.yml                       # GitHub Actions CI/CD (test, build Docker)
├── requirements.txt                 # Python dependencies
├── Dockerfile                       # Multi-stage Streamlit container
└── README.md                        # This file
```

## Quick Start

### Local Development

```bash
# Clone and setup
git clone <repository-url>
cd Olympics-data-analysis-mlops
python -m venv Olympics
source Olympics/bin/activate  # On Windows: Olympics\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Train the model
python src/train.py
# Output: Model Accuracy: 0.8973, Features: 88, Saved at: models/model.pkl

# Run Streamlit app
streamlit run app.py
# Open browser to http://localhost:8501
```

### Using Docker

```bash
# Build image
docker build -t olympics-analysis:latest .

# Run container (exposed on port 8501)
docker run -p 8501:8501 olympics-analysis:latest
# Open browser to http://localhost:8501
```

## Model Details

### Training Features (8)
- **Physical**: Age, Height, Weight
- **Athlete**: Sex (M/F)
- **Event**: Sport, Event, Team, Year

### Model Specifications
- **Algorithm**: Random Forest Classifier
- **Hyperparameters**: 50 trees, max_depth=10
- **Final Features**: 88 (after one-hot encoding; limited to top 15 sports, 40 events, 50 teams)
- **Data**: 33,927 samples (after filtering)
- **Accuracy**: 89.73% on held-out test set

### Prediction Output
For each athlete profile, model returns:
- Binary prediction (0=No Medal, 1=Medal Won)
- Probability score (0–100%)

## Usage Guide

### 1. Medal Prediction Page

#### Manual Mode
1. Navigate to **Medal Prediction** page
2. Enter athlete profile (Age, Height, Weight, Sex)
3. Select Sport, Event, Team, Year from dropdowns
4. Click **Predict Medal**
5. View prediction & probability

#### Batch Mode (CSV Upload)
1. Prepare CSV with columns: `Age, Height, Weight, Sex, Sport, Event, Team, Year`
2. Upload file in **Upload CSV** mode
3. Click **Predict on All Records**
4. Download results as CSV

### 2. Index Page (Analytics)
- **Medal Tally**: Filter by year/country
- **Overall Analysis**: Sports heatmap, athlete participation trends
- **Country Analysis**: Medals over time per nation
- **Athlete Analysis**: Age/height distribution, gender participation

## CI/CD Pipeline

GitHub Actions workflow (`.github/workflows/ci.yml`) runs on every push/PR:

1. **Setup**: Python 3.9, dependencies
2. **Validate**: Run `scripts/ensure_requirements.py` to add `streamlit` if missing
3. **Test**: Run pytest (if tests/ folder exists)
4. **Build**: Docker image via Buildx (multi-platform support)

## Development

### Adding Tests
Create `tests/` folder with `test_*.py` files. CI will auto-detect and run them:

```bash
mkdir tests
echo "def test_example(): assert True" > tests/test_example.py
python -m pytest tests/ -v
```

### Model Retraining
To retrain with new data or parameters:

```bash
python src/train.py
# Updates models/model.pkl with new accuracy metrics
```

### Running Prediction Script
Batch predict on the full dataset:

```bash
python src/predict.py
# Outputs predictions (first 5 rows shown)
```

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `ModuleNotFoundError: streamlit` | Run `pip install -r requirements.txt` |
| `FileNotFoundError: models/model.pkl` | Run `python src/train.py` to train first |
| Port 8501 already in use | Use `streamlit run app.py --server.port 8502` |
| Docker build fails | Ensure Docker daemon is running; check `docker version` |

## Dataset Info

**athlete_events.csv** (271,116 rows × 15 cols)
- Historical Olympic data from 1896–2024
- Athlete details: Name, Age, Height, Weight, Sex, Team
- Event details: Year, Season, Sport, Event, Medal (Gold/Silver/Bronze/NaN)

**noc_regions.csv**
- Mapping of NOC codes to country names

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Commit changes (`git commit -m "Add your feature"`)
4. Push to branch (`git push origin feature/your-feature`)
5. Open a Pull Request

The CI pipeline will automatically test and build your changes.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

## Authors

- **Data Scientists / ML Engineers**: Olympics MLOps Team
- **Built with**: Python, Streamlit, scikit-learn

---

**Questions?** Open an issue or contact the team.

**Latest Update**: February 21, 2026

