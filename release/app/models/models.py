import os
import sys

# Add the parent directory to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
parent_dir = os.path.abspath(os.path.join(parent_dir, ".."))
sys.path.append(parent_dir)

from pydantic import BaseModel
from utilities import custom_logging

logger = custom_logging.CustomLogging(__name__)

class Classifier(BaseModel):
    """
    Represents a credit card transaction.
    
    Attributes:
        The attributes have been anonymized to be used from V1 to V28 (all float),
        we only have the Time (int) and Amount (float) attributes without 
        anonymization.
    """
    Time : float
    V1 : float
    V2 : float
    V3 : float
    V4 : float
    V5 : float
    V6 : float
    V7 : float
    V8 : float
    V9 : float
    V10 : float
    V11 : float
    V12 : float
    V13 : float
    V14 : float
    V15 : float
    V16 : float
    V17 : float
    V18 : float
    V19 : float
    V20 : float
    V21 : float
    V22 : float
    V23 : float
    V24 : float
    V25 : float
    V26 : float
    V27 : float
    V28 : float
    Amount : float

logger.info("Executed correctly")