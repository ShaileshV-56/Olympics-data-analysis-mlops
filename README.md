# Olympics Data Analysis MLOps

A machine learning operations project for analyzing Olympic athlete performance data.

## Project Structure

```
Olympics-data-analysis-mlops/
├── app.py                      # Main application entry point
├── data/                       # Data directory
│   ├── athlete_events.csv      # Olympic events dataset
│   └── noc_regions.csv         # NOC regions mapping
├── src/                        # Source code
│   ├── preprocessor.py         # Data preprocessing logic
│   ├── helper.py               # Helper utilities
│   ├── train.py               # Model training script
│   └── predict.py             # Prediction script
├── models/                     # Trained models
│   └── model.pkl              # Serialized model
├── notebooks/                  # Jupyter notebooks for exploration
├── requirements.txt            # Python dependencies
├── Dockerfile                  # Docker configuration
├── .github/workflows/ci.yml   # CI/CD pipeline
└── README.md                   # This file
```

## Setup

### Prerequisites
- Python 3.9+
- Docker (optional)

### Installation

1. Clone the repository
```bash
git clone <repository-url>
cd Olympics-data-analysis-mlops
```

2. Create a virtual environment
```bash
python -m venv Olympics
source Olympics/bin/activate  # On Windows: Olympics\Scripts\activate
```

3. Install dependencies
```bash
pip install -r requirements.txt
```

## Usage

### Training
```bash
python src/train.py
```

### Prediction
```bash
python src/predict.py
```

### Running the Application
```bash
python app.py
```

### Docker

Build and run the application in Docker:
```bash
docker build -t olympics-analysis:latest .
docker run olympics-analysis:latest
```

## Development

Run tests and checks through the CI pipeline or locally:
```bash
# Install development dependencies
pip install -r requirements.txt

# Run linting/formatting as needed
```

## License

This project is licensed under the MIT License.
