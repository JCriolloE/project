import os
import sys

# Add the parent directory to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(parent_dir)

import zipfile
import pandas as pd
from utilities import custom_logging

logger = custom_logging.CustomLogging(__name__)

class Loading:

    def __init__(self, model_path):
        self.model_path = model_path

    def loadingData(p1):
        """
            Unzipping and loading data
        """
        # Define the name of the ZIP file you want to extract
        zip_file = p1 + "creditcard.zip"

        # Create an `Zipfile` object for the ZIP file
        with zipfile.ZipFile(zip_file, "r") as zip_ref:

            # Get the file list in the ZIP file
            files = zip_ref.namelist()

            # Itera on the files in the ZIP file
            for file in files:

                # If the file is a CSV file, extra be the current directory
                if file.endswith(".csv"):
                    with zip_ref.open(file) as f:
                        with open(file, "wb") as out_file:
                            out_file.write(f.read())

        df = pd.read_csv(zip_file)
        logger.info("Data loaded")
        return df
