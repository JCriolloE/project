# ITESM MLOPs Project

## Introduction of the project

Welcome to the final project focused on MLOps, where the key concepts of ML frameworks and their application will be applied in a practical approach. Throughout this project, the basic concepts and fundamental tools for developing software in the field of MLOps are shown, covering everything from configuring the environment to best practices for creating ML models and deploying them.

## About the project

The overall goal of this project is to build a robust and reproducible MLOps workflow for developing, training, and deploying machine learning models. A linear regression model will be used as a proof of concept due to its simplicity, and it will be applied to the Credit Card Fraud data set to predict the probability that the transaction is a fraud or non-fraud based on certain characteristics.

This project covers the following topics:

1. **Key concepts of ML systems**  
The objective of this module is to give an introduction to MLOps, life cycle and architecture examples is also given.

2. **Basic concepts and tools for software development**  
This module focuses on introducing the principles of software development that will be used in MLOps. Consider the configuration of the environment, tools to use, and best practices, among other things.

3. **Development of ML models**  
This module consists of showing the development of an ML model from experimentation in notebooks, and subsequent code refactoring, to the generation of an API to serve the model.

4. **Deployment of ML models**  
The objective of this module is to show how a model is served as a web service to make predictions.

5. **Integration of concepts**  
This module integrates all the knowledge learned in the previous modules. A demo of Continuous Delivery is implemented.

### Baseline

This MLOps project is focused on demonstrating the implementation of a complete workflow that ranges from data preparation to exposing a local web service to make predictions using a linear regression model. The chosen dataset is the famous Credit Card Fraud dataset, which contains information about the transactions and whether or not they are fraud.

The purpose is to establish a starting point or "baseline" that will serve as a reference to evaluate future improvements and not only more complex algorithms but more complex components and further deployments.

### Scope

This project is planned to cover the topics seen in the course syllabus, which was designed to include technical capacity levels 0, 1 and a small part of 2 of [Machine Learning operations maturity model - Azure Architecture Center | Microsoft Learn](https://learn.microsoft.com/en-us/azure/architecture/example-scenario/mlops/mlops-maturity-model).

In other words, knowledge is integrated regarding the learning of good software development practices and Dev Ops (Continuous Integration) applied to the deployment of ML models.

### Links to experiments like notebooks

You can find the Credit Card Fraud experiment here:

* [proyect_Notebook.ipynb](release\proyect_Notebook.ipynb)

## Setup

### Python version and packages to install

* Change the directory to the root folder.

* Create a virtual environment with Python 3.10+:

    ```bash
    py -3.10 -m venv envProyecto
    ```

* Activate the virtual environment

    ```bash
    envProyecto/Scripts/activate.bat
    ```

* Install libraries
Run the following command to install the libraries/packages.

    ```bash
    pip install -r requirements.txt
    ```

## Model training from a main file

To train the Logistic Model, only run the following code:

```bash
python release\main.py
```

Output:

```bash
Logistic Regression:
              precision    recall  f1-score   support

           0     0.9062    0.9775    0.9405        89
           1     0.9787    0.9109    0.9436       101

    accuracy                         0.9421       190
   macro avg     0.9425    0.9442    0.9421       190
weighted avg     0.9448    0.9421    0.9422       190
```

## Execution of unit tests (Pytest)

### Test location

You can find the test location in the [test](tests) folder, and the following tests:

* Test `test_csv_file_existence`:
Test if there is a CSV file on Data route.

* Test `test_if_model_exists`:
Test if there is a `.joblib` file on Models route.

* Test `test_missing_indicator_transform`:  
Test the `transform` method of the `MissingIndicator` transformer

* Test `test_missing_numerical_imputer`:  
Test the `transform` method of the `NumericalImputer` transformer

### Execution instructions

#### Test `loading` class

The following test validates the [load_data.py](release\load\load_data.py) module, with the `loading` class.

Follow the next steps to run the test.

* Run in the terminal:

    ```bash
    pytest ./tests/test_csv_file_existence.py::test_csv_file_exists -v
    ```

#### Test model existence

The following test validates the model's existence after the training.

Follow the next steps to run the test.

* Run in the terminal:

    ```bash
    pytest ./tests/test_if_model_exists.py::test_joblib_file_exists -v
    ```

#### Test `MissingIndicator` class - `transform` method

The following test validates the [custom_transformers.py](release\preprocess\custom_transformers.py) module, with the `MissingIndicator` class in the `transform` method.

Follow the next steps to run the test.

* Run in the terminal:

    ```bash
    pytest ./tests/test_missing_indicator_transform.py::test_missing_indicator_transform -v
    ```

#### Test `NumericalImputer` class - `transform` method

The following test validates the [custom_transformers.py](release\preprocess\custom_transformers.py) module, with the `NumericalImputer` class in the `transform` method.

Follow the next steps to run the test.

* Run in the terminal:

    ```bash
    pytest ./tests/test_missing_numerical_imputer.py::test_NumericalImputer_transform -v
    ```

## Usage

### Individual Fastapi and Use Deployment

*Go to `release` folder and run the next command to start the Credit Card API locally

    ```bash
    uvicorn main_api:app --reload
    ```

#### Checking endpoints

1. Access `http://127.0.0.1:8000/`, you will see a message like this `"Classifier is all ready to go!"`
2. Access `http://127.0.0.1:8000/docs`, the browser will display something like this:
![FastAPI Docs](release\images\API.JPG)
3. Try running the following predictions with the endpoint by writing the following values:
    * **Prediction 1**  
        Request body

        ```bash
        {
            "Time": 0,
            "V1": 0,
            "V2": 0,
            "V3": 0,
            "V4": 0,
            "V5": 0,
            "V6": 0,
            "V7": 0,
            "V8": 0,
            "V9": 0,
            "V10": 0,
            "V11": 0,
            "V12": 0,
            "V13": 0,
            "V14": 0,
            "V15": 0,
            "V16": 0,
            "V17": 0,
            "V18": 0,
            "V19": 0,
            "V20": 0,
            "V21": 0,
            "V22": 0,
            "V23": 0,
            "V24": 0,
            "V25": 0,
            "V26": 0,
            "V27": 0,
            "V28": 0,
            "Amount": 0
        }
        ```

        Response body
        The output will be:

        ```bash
        "Resultado predicción: [0]"
        ```

### Individual deployment of the API with Docker and usage

#### Build the image

* Ensure you are in the `project/` directory (root folder).
* Run the following code to build the image:

    ```bash
    docker build -t classifier
    ```

* Inspect the image created by running this command:

    ```bash
    docker images
    ```

    Output:

    ```bash
    REPOSITORY      TAG       IMAGE ID       CREATED       SIZE
    classifier      latest    9948abd70c53   16 hours ago  3.25GB
    ```

#### Run Titanic REST API

1. Run the next command to start the `classifier` image in a container.

    ```bash
    docker run -d --rm --name classifier -p 8000:8000 classifier
    ```

2. Check the container running.

    ```bash
    docker ps -a
    ```

    Output:

    ```bash
    CONTAINER ID   IMAGE           COMMAND                  CREATED          STATUS          PORTS                    NAMES
    9261ad9538bd   classifier   "uvicorn main_api:app --…"   3 hours ago   Exited (0) 3 hours ag   8000/tcp   classifier
    ```

#### Checking endpoints for app

1. Access `http://127.0.0.1:8000/`, and you will see a message like this `"Classifier classifier is all ready to go!"`
2. A file called `CustomLogging.log` will be created automatically inside the container in the `utilities` folder. We will inspect it below.
3. Access `http://127.0.0.1:8000/docs`, the browser will display something like this:
    ![FastAPI Docs](release\images\API.JPG)

4. Try running the following predictions with the endpoint by writing the following values:
    * **Prediction 1**  
        Request body

        ```bash
        {
            "Time": 0,
            "V1": 0,
            "V2": 0,
            "V3": 0,
            "V4": 0,
            "V5": 0,
            "V6": 0,
            "V7": 0,
            "V8": 0,
            "V9": 0,
            "V10": 0,
            "V11": 0,
            "V12": 0,
            "V13": 0,
            "V14": 0,
            "V15": 0,
            "V16": 0,
            "V17": 0,
            "V18": 0,
            "V19": 0,
            "V20": 0,
            "V21": 0,
            "V22": 0,
            "V23": 0,
            "V24": 0,
            "V25": 0,
            "V26": 0,
            "V27": 0,
            "V28": 0,
            "Amount": 0
        }
        ```

        Response body
        The output will be:

        ```bash
        "Resultado predicción: [0]"
        ```

        ![Prediction](release\images\RESULTS.JPG)
#### Opening the logs

1. Run the command

    ```bash
    docker exec -it classifier bash
    ```

    Output:

    ```bash
    root@9261ad9538bd:#
    ```

2. Check the existing files:

    ```bash
    ls
    ```

    Output:

    ```bash
    Dockerfile  __init__.py  __pycache__  api  classifier  data  load  main.py  main_api.py  models  preprocess  requirements.txt  train  utilities
    ```

3. Open the file `CustomLogging.log` and inspect the logs with this command:

    ```bash
    vim CustomLogging.log
    ```

    Output:

    ```log
    2023-08-22 00:44:34,221:__main__:custom_logging:INFO:Executed correctly
    2023-08-22 00:45:37,485:models.models:custom_logging:INFO:Executed correctly
    2023-08-22 00:49:11,034:models.models:custom_logging:INFO:Executed correctly
    2023-08-22 01:09:20,603:__main__:custom_logging:INFO:Setup established
    2023-08-22 01:09:23,262:load.load_data:custom_logging:INFO:Data loaded
    2023-08-22 01:09:23,262:preprocess.custom_transformers:custom_logging:INFO:fit MissingIndicator executed
    2023-08-22 01:09:23,281:preprocess.custom_transformers:custom_logging:INFO:transform MissingIndicator executed
    2023-08-22 01:09:23,282:preprocess.custom_transformers:custom_logging:INFO:fit NumericalImputer executed
    2023-08-22 01:09:23,301:preprocess.custom_transformers:custom_logging:INFO:transform NumericalImputer executed
    2023-08-22 01:09:23,644:preprocess.preprocess:custom_logging:INFO:train, test and val data was generated
    2023-08-22 01:09:24,176:train.train:custom_logging:INFO:Model trained and saved
    2023-08-22 01:09:24,176:classifier.predict:custom_logging:INFO:Resultado predicci�n: [0 1 1 1 0 0 1 0 0 0 1 0 0 1 0 1 1 0 0 1 0 0 0 0 1 0 1 1 1 1 1 1 0 0 0 1 0
    1 0 1 1 1 0 1 0 0 0 1 1 1 0 0 0 0 0 0 0 0 1 1 0 0 0 0 1 1 1 1 1 0 0 0 0 0
    0 0 0 0 1 0 1 1 0 1 0 1 0 1 1 1 1 1 1 1]

    ```

4. Copy the logs to the root folder:

    ```bash
    docker cp classifier:/CustomLogging.log .
    ```

    Output:

    ```bash
    Successfully copied 26kB to .../utilities/.
    ```

#### Delete container and image

* Stop the container:

    ```bash
    docker stop classifier
    ```

* Verify it was deleted

    ```bash
    docker ps -a
    ```

    Output:

    ```bash
    CONTAINER ID   IMAGE     COMMAND   CREATED   STATUS    PORTS     NAMES
    ```

* Delete the image

    ```bash
    docker rmi classifier
    ```

    Output:

    ```bash
    Deleted: sha256:bb48551cf5423bad83617ad54a8194501aebbc8f3ebb767de62862100d4e7fd2
    ```

### Complete deployment of all containers with Docker Compose and usage

#### Create the network

First, create the network AIService by running this command:

```bash
docker network create AIservice
```

#### Run Docker Compose

* Ensure you are in the directory where the docker-compose.yml file is located

* Run the next command to start the App and Frontend APIs

    ```bash
    docker-compose up --build
    ```

    You will see something like this:

    ```bash
    Use 'docker scan' to run Snyk tests against images to find vulnerabilities and learn how to fix them
    Creating project-app ... done
    Creating project-app2 ... done
    Creating project-frontend ... done
    project-app       | INFO:     Will watch for changes in these directories: ['/']
    project-app       | INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
    project-app       | INFO:     Started reloader process [1] using StatReload

    project-app2      | INFO:     Will watch for changes in these directories: ['/']
    project-app2      | INFO:     Uvicorn running on http://0.0.0.0:7000 (Press CTRL+C to quit)
    project-app2      | INFO:     Started reloader process [1] using StatReload

    itesm_mlops_project-frontend  | INFO:     Will watch for changes in these directories: ['/']
    itesm_mlops_project-frontend  | INFO:     Uvicorn running on http://0.0.0.0:3000 (Press CTRL+C to quit)
    itesm_mlops_project-frontend  | INFO:     Started reloader process [1] using StatReload
    project-app         | INFO:     Started server process [8]
    project-app         | INFO:     Waiting for application startup.
    project-app         | INFO:     Application startup complete.
    project-app-2       | INFO:     Started server process [9]
    project-app-2       | INFO:     Waiting for application startup.
    project-app-2       | INFO:     Application startup complete.
    project-frontend    | INFO:     Started server process [10]
    project-frontend    | INFO:     Waiting for application startup.
    project-frontend    | INFO:     Application startup complete.
    ```

#### Checking endpoints in Frontend

1. Access `http://127.0.0.1:3000/`, and you will see a message like this `"Front-end is all ready to go!"`
2. A file called `frontend.log` will be created automatically inside the container. We will inspect it below.
3. Access `http://127.0.0.1:3000/docs`

4. Try running the following predictions with the endpoint `classify` by writing the following values:
    * **Prediction 1**  
        Request body

        ```bash
        {
            "Time": 0,
            "V1": 0,
            "V2": 0,
            "V3": 0,
            "V4": 0,
            "V5": 0,
            "V6": 0,
            "V7": 0,
            "V8": 0,
            "V9": 0,
            "V10": 0,
            "V11": 0,
            "V12": 0,
            "V13": 0,
            "V14": 0,
            "V15": 0,
            "V16": 0,
            "V17": 0,
            "V18": 0,
            "V19": 0,
            "V20": 0,
            "V21": 0,
            "V22": 0,
            "V23": 0,
            "V24": 0,
            "V25": 0,
            "V26": 0,
            "V27": 0,
            "V28": 0,
            "Amount": 0
        }
        ```
        Response body
        The output will be:

        ```bash
        "Resultado predicción: [0]"
        ```

## Resources

Here you will find information about this project and more.

### Information sources

* [MNA - Master in Applied Artificial Intelligence](https://learn.maestriasydiplomados.tec.mx/pos-programa-mna-v-)
* [ITESM MLOps Course GitHub Repository](https://github.com/carloslme/itesm-mlops)

## Contact information

* **Credits**

    ------------

  * **Development Lead**

    * Jorge Criollo <jcriollo.e@gmail.com>
    * [GitHub Profile](https://github.com/JCriolloE/)
    * [LinkedIn](https://www.linkedin.com/in/jorge-criollo-934406120/)
