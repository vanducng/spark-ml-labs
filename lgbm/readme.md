# Basic step
* Extraction: extract features from raw data
* Transformation: scaling, converting, or modyfing the features
* Selection: select subset of selected features
* Locality Sensitivity Hasing (LSH): this class of algorithms combines aspects of feature transformation with other algorithms

# Notebook setup
* Setup driver
    ```bash
    export SPARK_HOME=/opt/spark
    export PATH=$PATH:$SPARK_HOME/bin:$SPARK_HOME/sbin
    export PYSPARK_PYTHON=/home/vanducng/anaconda3/bin/python
    export PYTHONPATH=$SPARK_HOME/python:$SPARK_HOME/python/lib/py4j-0.10.7-src.zip:$PYTHONPATH
    export PYSPARK_DRIVER_PYTHON=jupyter
    export PYSPARK_DRIVER_PYTHON_OPTS='notebook'
    ```

* Run below command
    ```shell
    pyspark --packages com.microsoft.ml.spark:mmlspark_2.11:0.18.1
    ```

# References
* Fine tuning parameters: https://towardsdatascience.com/lightgbm-hyper-parameters-tuning-in-spark-6b8880d98c85
* Fraud detection with lgbm: https://towardsdatascience.com/how-to-perform-credit-card-fraud-detection-at-large-scale-with-apache-spark-and-lightgbm-61a982de8672