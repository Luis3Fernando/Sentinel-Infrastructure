# SENTINEL 

![Version](https://img.shields.io/badge/version-beta-orange) ![Language](https://img.shields.io/badge/language-Yaml-orange) ![Status](https://img.shields.io/badge/status-beta-orange)
![Stars](https://img.shields.io/github/stars/Luis3Fernando/Sentinel-Infrastructure?style=social)

This project defines an infrastructure with Docker Compose to build an Apache Spark cluster with a Spark Master, two Spark Workers, a Jupyter Notebook environment to run PySpark jobs and a PostgreSQL database to store results. Everything is configured with shared volumes, which allows to easily work with datasets, scripts and external libraries.


📌 Apache Spark Cluster with Docker Compose

This project defines a complete infrastructure for running an Apache Spark cluster with Docker Compose, which includes:

- Spark Master
- Spark Workers
- Jupyter Notebook with PySpark support
- PostgreSQL as database

The configuration is intended for development and test environments, allowing to run distributed jobs and store results in a relational database.

bash```
spark-submit --jars /libs/postgresql-42.6.0.jar /jobs/pipeline_robos.py
```
