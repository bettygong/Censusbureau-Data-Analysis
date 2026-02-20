# Censusbureau-Data-Analysis
The objective of this project is to develop a predictive model that classifies individuals into income groups (<50K vs. >50K) using demographic, education, and employment-related attributes from the Census Bureau dataset. To be specific, I utilized a XGBoost model to conduct binary classification, achieving a test accuracy of 0.88 and recall rate of 0.87. Furthermore, I build a customer segmentation unsupervised model clustering 5 groups for marketing purposes. 5 customer segments are Traditional Households, Growth-oriented Households, Next-Gen Saver, Aggressive Investors, and Affluent Retirees.  From a business perspective, these models can support targeted marketing campaigns and product personalization.  
[View the report](customer_data_analysis_report.pdf)  

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

