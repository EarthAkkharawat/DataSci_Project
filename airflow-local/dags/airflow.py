from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
from airflow.models import DagRun, TaskInstance, XCom
from airflow import settings
from airflow.utils.db import provide_session
from sqlalchemy.orm.session import make_transient

from PIL import Image
import numpy as np
import os

import onnxruntime as rt

import urllib.request
import requests
from pathlib import Path

import onnxruntime as rt

BASE_DIR = Path(__file__).resolve(strict=True).parent


@provide_session
def cleanup_xcom(session=None, **context):
    dag = context["dag"]
    dag_id = dag._dag_id
    session.query(XCom).filter(XCom.dag_id == dag_id).delete()


def fetch_data_from_traffyapi(ti):
    today = datetime.now().date()
    yesterday = today - timedelta(days=1)
    formatted_date = yesterday.strftime("%Y-%m-%d")

    url = "https://publicapi.traffy.in.th/share/teamchadchart/search?start=" + \
        formatted_date+"&end="+formatted_date+"&sort=ASC"
    response = requests.get(url)
    data = response.json()
    data_body = []
    if "results" in data:
        print(len(data['results']))
        for item in data.get("results", []):
            photo_url = item.get("photo_url")
            coords = item.get("coords")
            ticket_id = item.get("ticket_id")
            timestamp = item.get("timestamp").split(
                " ")[0]  # Extract date from timestamp

            location = f"{coords[1]},{coords[0]}"
            location_round = f"{round(float(coords[1]), 2)},{round(float(coords[0]), 2)}"

            data_body.append({
                "ticket_id": ticket_id,
                "photo_url": photo_url,
                "timestamp": timestamp,
                "location": location,
                "location_round": location_round,
            })

    ti.xcom_push(key='traffyapi_data', value=data_body)


def feed_data_to_model(ti):
    payload = ti.xcom_pull(
        key='traffyapi_data', task_ids='fetch_data')
    prediction_list = []

    for i, item in enumerate(payload[:50]):
        urllib.request.urlretrieve(item['photo_url'], f"image{i}.jpg")
        inputImage = Image.open(f'image{i}.jpg')
        inputImage = inputImage.resize((224, 224))
        inputImage = np.array(inputImage.convert('RGB'))

        inputTensor = ((inputImage / 255) -
                       [0.4303, 0.4301, 0.4139]) / [0.2186, 0.2140, 0.2205]
        inputTensor = inputTensor.transpose(
            2, 0, 1)[np.newaxis].astype(np.float32)
        sessOptions = rt.SessionOptions()
        sessOptions.graph_optimization_level = rt.GraphOptimizationLevel.ORT_ENABLE_ALL
        model = rt.InferenceSession(os.path.join(
            BASE_DIR, "model.onnx"), sessOptions)

        def thresholding_lambda(x): return 1 if x > 0.5 else 0

        output = model.run([], {'input': inputTensor})[0]
        output = np.vectorize(thresholding_lambda)(output)
        prediction_list.append(output[0].tolist())
    ti.xcom_push(key='model_prediction', value=prediction_list)


def send_prediction_to_vis(ti):
    model_prediction = ti.xcom_pull(
        key='model_prediction', task_ids='feed_data_to_model')
    payload = ti.xcom_pull(
        key='traffyapi_data', task_ids='fetch_data')
    ti.xcom_push(key='vis0', value=len(payload))
    payload = payload[:50]
    for i in range(len(payload)):
        for j in range(10):
            payload[i][str(j)] = model_prediction[i][j]

    ti.xcom_push(key='vis1', value=payload)

    if payload:
        post_url = "https://api.powerbi.com/beta/271d5e7b-1350-4b96-ab84-52dbda4cf40c/datasets/dba5b78e-aaeb-4914-9df3-931a56dcd817/rows?key=Vca7Fx48SMhUmUnO48xZre8GaGC%2FEegcb7Fruc4Coi4IBAVHNYhauufsDpMarYJF5U79I6rE%2BvOYubkZhk1eYg%3D%3D"
        post_response = requests.post(post_url, json=payload)

        if post_response.status_code == 200:
            ti.xcom_push(key='status', value="Data posted successfully.")
        else:
            ti.xcom_push(key='status', value="Failed to post data.")


default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2023, 5, 16),
    'retries': 1,
    'retry_delay': timedelta(minutes=60)
}

dag = DAG('TraffyfonduePredictionPipeline',
          description='A DAG to fetch data from the Traffy API, then predict (multi-label) image classes and POST via API to visualize the results using PowerBI',
          default_args=default_args,
          schedule_interval='@hourly',
          catchup=False)

fetch_task = PythonOperator(
    task_id='fetch_data',
    python_callable=fetch_data_from_traffyapi,
    provide_context=True,
    dag=dag
)

delete_metadata = PythonOperator(
    task_id="delete_metadata",
    python_callable=cleanup_xcom,
    provide_context=True,
    dag=dag
)

feed_and_predict_task = PythonOperator(
    task_id='feed_data_to_model',
    python_callable=feed_data_to_model,
    provide_context=True,
    dag=dag
)

visualize_task = PythonOperator(
    task_id='send_prediction_to_vis',
    python_callable=send_prediction_to_vis,
    provide_context=True,
    dag=dag
)

fetch_task >> feed_and_predict_task >> visualize_task >> delete_metadata
