# Censusbureau-Data-Analysis

.
├── run.py                 # Main runnable pipeline
├── census-bureau.data              # Dataset (not included, please upload it)
├── census-bureau.columns           # Column names (included)
├── notebooks/                      # (optional) original exploration notebooks
└── README.md

## Requirement 
python 3.8
numpy==1.23.5
pandas==1.5.3
scikit-learn==1.2.2
scipy==1.10.1
xgboost==1.7.6

Other version of python and sklearns should also work, but may need to change `OneHotEncoder(sparse=True)` to `sparce_output=True`

## Run model
`run.py` file will run logistic regression and XGBoost for income level classification, and run Kmeans for customer segmentation with specified parameter. 
`python run.py`
