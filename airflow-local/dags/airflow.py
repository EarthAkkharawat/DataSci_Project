from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
from airflow.models import DagRun, TaskInstance, XCom
from airflow import settings
from airflow.utils.db import provide_session
from sqlalchemy.orm.session import make_transient
import numpy as np
import requests
from pathlib import Path

BASE_DIR = Path(__file__).resolve(strict=True).parent
headers = {
    'Content-Type': 'application/json' 
}

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
    data = ti.xcom_pull(
        key='traffyapi_data', task_ids='fetch_data')
    prediction_list = []
    error_list = []
    i=0
    for i,item in enumerate(data):
        if item['photo_url'].split("/")[-1].split(".")[0][:-1] == 'corruption_photo':
            error_list.append(i)
            continue
        
        payload = {
            "url":item['photo_url']
        }

        response = requests.post("https://tofu-api-nj2eo5v2pq-as.a.run.app", json=payload, headers=headers)
        if response.status_code != 200 or 'prediction' not in response.json():
            error_list.append(i)
            continue
        prediction_list.append(response.json()['prediction'])
        i+=1

    ti.xcom_push(key='error_images', value=error_list)
    ti.xcom_push(key='model_prediction', value=prediction_list)


def send_prediction_to_vis(ti):
    model_prediction = ti.xcom_pull(
        key='model_prediction', task_ids='feed_data_to_model')
    payload = ti.xcom_pull(
        key='traffyapi_data', task_ids='fetch_data')
    
    error_images = ti.xcom_pull(
        key='error_images', task_ids='feed_data_to_model')
    
    data = []
    idx = 0
    for i in range(len(payload)):
        if i in error_images:
            continue
        for j in range(10):
            payload[i][str(j)] = model_prediction[idx][j]
        idx+=1
        data.append(payload[i])

    if data:
        post_url = "https://api.powerbi.com/beta/271d5e7b-1350-4b96-ab84-52dbda4cf40c/datasets/dba5b78e-aaeb-4914-9df3-931a56dcd817/rows?key=Vca7Fx48SMhUmUnO48xZre8GaGC%2FEegcb7Fruc4Coi4IBAVHNYhauufsDpMarYJF5U79I6rE%2BvOYubkZhk1eYg%3D%3D"
        post_response = requests.post(post_url, json=data)

        if post_response.status_code == 200:
            ti.xcom_push(key='status', value="Data posted successfully.")
        else:
            ti.xcom_push(key='status', value="Failed to post data.")


default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2023, 5, 1),
    'retries': 1,
    'retry_delay': timedelta(minutes=5)
}

dag = DAG('TraffyfonduePredictionPipeline',
          description='A DAG to fetch data from the Traffy API, then predict (multi-label) image classes and POST via API to visualize the results using PowerBI',
          default_args=default_args,
          schedule_interval='@daily',
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