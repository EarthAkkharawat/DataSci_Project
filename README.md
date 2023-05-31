# TofuFondue Project

## Overview
Using API requests to obtain real-time information on problems reported through the [TraffyFondue website](https://www.traffy.in.th/?page_id=4434). From there, we can use ours multilabel classification model to predict images associated with the reported problems. Finally, the results are visualized in a PowerBI dashboard by sending an API.

Model Deploy: https://github.com/AkiraSitdhi/TofuFondue-API
Visualization (PowerBI): https://shorturl.at/eIJR4
Model REST API: https://tofu-api-nj2eo5v2pq-as.a.run.app

## Details
### Data Engineering Part
- Web scraping: scraped images to enlarge train dataset
- Implemented GET Request Api from [Traffy Fondue](https://www.traffy.in.th/?page_id=27351)
- Airflow: use to create project pipeline consisting of 4 tasks
  1. Using api GET request the reported problems in realtime (daily) 
  2. Call the REST API of our deployed model to predict the images obtained from the previous task
  3. Sending the prediction result and problems infomation to visualize via PowerBI using API POST request
  4. Clear all metadata on airflow XComs database

## Machine Learning Part
- Multilabel image classification: 
  - Train: train a neural network model using 9376 images sourced from [Traffy Fondue](https://www.traffy.in.th/?page_id=27351) and scraped images from the Data Engineering Part. These images have been classified into 10 categories: sanitary, sewer, stray, canal, light, flooding, electric, traffic, road, and sidewalk.
  - Predict: realtime predict images that obtained from Data Engineering Part
- MLFlow: use to save parameters and artifacts, as well as monitor loss and macro F1 scores during model training.
- Onnx: use to optimize and reduce the size of the model when deployed.
- Google Cloud Services: use to deploy model and can call by REST API

## Visualization Part
- Power BI streaming dataset
- Power BI Dashboard
- Geospatial visualization

## How to run airflow
1. Open terminal in /airflow-local and run `docker-compose up airflow-init`

2. Launch Airflow `docker-compose up`

Wait for scheduler and webserver to get healthy, then go to `localhost:8080` 

```python
username: airflow
password: airflow
```
