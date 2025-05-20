# Fake Job Posting Detection

This project implements a machine learning pipeline to detect fraudulent job postings using both text and meta-features. The system uses advanced NLP techniques and traditional machine learning to classify job postings as either legitimate or fraudulent.

## Project Structure

```
src/
├── data/           # Data loading and preprocessing
├── eda/            # Exploratory data analysis
├── text_processing/# Text preprocessing and NLP
├── features/       # Feature engineering
├── models/         # Model training and selection
├── evaluation/     # Model evaluation and metrics
├── deployment/     # Model deployment and serving
outputs/
├── plots/         # Generated visualizations
├── models/        # Saved model artifacts
```

## Setup Instructions

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download required NLTK data:
```python
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"
```

4. Download spaCy model:
```bash
python -m spacy download en_core_web_sm
```

## Usage

1. Run the complete pipeline:
```bash
python main.py
```

2. Run individual components:
```bash
python src/data/load_data.py
python src/eda/analyze_data.py
python src/models/train_models.py
```

3. Make predictions using the trained model:
```bash
python src/deployment/predict.py --input "path/to/new_job_posting.csv"
```

## Features

- Comprehensive data preprocessing and cleaning
- Advanced text processing with NLTK and spaCy
- Feature engineering for both text and meta-features
- Multiple classification models (Logistic Regression, Random Forest, XGBoost)
- Model evaluation with various metrics
- SHAP-based model interpretability
- BERT-based contextual embeddings
- Deployment-ready prediction pipeline

## Model Performance

The best performing model achieves:
- Accuracy: ~95%
- F1-Score: ~0.94
- ROC-AUC: ~0.96

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Dataset source: [Kaggle Fake Job Posting Prediction](https://www.kaggle.com/datasets/shivamb/real-or-fake-fake-jobposting-prediction)
- Thanks to all contributors and the open-source community 