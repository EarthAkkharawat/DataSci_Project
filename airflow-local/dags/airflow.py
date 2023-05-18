from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta


def fetch_data_from_traffyapi():
    # Code to fetch data from the API
    return


def feed_data_to_model():
    # Code to feed the data to the PyTorch model
    return

def send_prediction_to_vis():
    
    return


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
          schedule_interval='@daily',
          catchup=False)

fetch_task = PythonOperator(
    task_id='fetch_data',
    python_callable=fetch_data_from_traffyapi,
    dag=dag
)

feed_and_predict_task = PythonOperator(
    task_id='feed_data_to_model',
    python_callable=feed_data_to_model,
    dag=dag
)

post_task = PythonOperator(
    task_id='send_prediction_to_vis',
    python_callable=send_prediction_to_vis,
    dag=dag
)

fetch_task >> feed_and_predict_task >> post_task
