"""
    Imported Libraries
"""
import os
import sys
import warnings

# Add the parent directory to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(parent_dir)

from release.classifier.predict import Prediction
from release.load.load_data import Loading
from release.preprocess.custom_transformers import transformations_pipeline
from release.preprocess.preprocess import Preprocessing
from release.train.train import training
from release.utilities import custom_logging
from sklearn.metrics import classification_report

logger = custom_logging.CustomLogging(__name__)

warnings.filterwarnings("ignore")

SEED_SPLIT = 42
TARGET = 'Class'
COLUMNS_NAME = [
    'Time',
    'V1',
    'V2',
    'V3',
    'V4',
    'V5',
    'V6',
    'V7',
    'V8',
    'V9',
    'V10',
    'V11',
    'V12',
    'V13',
    'V14',
    'V15',
    'V16',
    'V17',
    'V18',
    'V19',
    'V20',
    'V21',
    'V22',
    'V23',
    'V24',
    'V25',
    'V26',
    'V27',
    'V28',
    'Amount',
    'Class']
URL = "C:\\Users\\jcriollo\\Desktop\\Respado\\ITESM\\Maestría Inteligencia " \
    "Artificial\\Z - MLOPS\\Proyecto\\4. Despliegue de modelos de ML\\" \
    "mlops\\project\\release\\data\\"

logger.info("Setup established")

df = Loading.loadingData(URL)

df = transformations_pipeline.fit_transform(df)

X_train, X_test, X_val, y_train, y_test, y_val = Preprocessing.robustScaler(df)

log_reg = training(X_train, y_train)

y_pred_log_reg_val = Prediction.predict(X_val)
print(classification_report(y_val, y_pred_log_reg_val, digits=4))
os.remove('creditcard.csv')
