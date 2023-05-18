import shutil
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
from airflow.models import DagRun, TaskInstance, XCom
from airflow import settings
from airflow.utils.db import provide_session
# import cv2
from sqlalchemy.orm.session import make_transient

from PIL import Image
from io import BytesIO

import pandas
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import os

from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
import onnxruntime as rt

import urllib.request
import requests
from pathlib import Path

import onnxruntime as rt

# num_classes = 10
# device = torch.device("cpu")

# efficientnet_version = 'b3'
# model_ft = torchvision.models.efficientnet_b3(weights=True)

# model_ft.classifier[-1] = nn.Sequential(
#     nn.Linear(
#         in_features=model_ft.classifier[-1].in_features, out_features=num_classes),
#     nn.Sigmoid()
# )

# model = model_ft.to(device)

# model.load_state_dict(torch.load('train_resources/best_model.pth'))


# class TraffyFondueImages(Dataset):

#     def __init__(self,
#                  img_dir,
#                  image_names_list,
#                  transforms=None):

#         super().__init__()
#         self.input_dataset = list()

#         _, _, files = next(os.walk(os.path.join(img_dir)))

#         for image_name in image_names_list:
#             input = [os.path.join(img_dir, image_name), image_name]
#             self.input_dataset.append(input)

#         self.transforms = transforms

#     def __len__(self):
#         return len(self.input_dataset)

#     def __getitem__(self, idx):
#         img = Image.open(self.input_dataset[idx][0]).convert('RGB')
#         x = self.transforms(img)
#         return x, self.input_dataset[idx][1]


def delete_all_xcom_entries():
    session = settings.Session()

    # Delete all XCom entries
    session.query(XCom).delete()

    # Make sure the deleted entries are not stored in the session
    session.flush()
    for obj in session:
        make_transient(obj)

    session.commit()
    session.close()


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

    # ti.xcom_push(key='traffyapi_data', value=data_body)
    delete_all_xcom_entries()
    session = settings.Session()

    # Delete all XCom entries
    session.query(XCom).delete()


# def convert_images_to_jpg(ti):
#     payload = ti.xcom_pull(key='traffyapi_data', task_ods='fetch_data')
#     image_names_list = []

#     # Create the empty directory
#     current_directory = os.path.dirname(os.path.abspath('airflow.py'))
#     new_directory = os.path.join(
#         os.path.dirname(current_directory), 'daily_images')
#     os.mkdir(new_directory)

#     for i, item in enumerate(payload):
#         response = requests.get(item['photo_url'])
#         image = Image.open(BytesIO(response.content))
#         image.save(f"../daily_images/img_{i}.jpg", 'JPEG')
#         image_names_list.append(f"img_{i}.jpg")
#     ti.xcom_push(key='image_names', value=image_names_list)


def feed_data_to_model(ti):
    # transform = transforms.Compose(
    #     [transforms.Resize((224, 224)),
    #      transforms.ToTensor(),
    #      transforms.Normalize(mean=[0.4303, 0.4301, 0.4139], std=[
    #          0.2186, 0.2140, 0.2205])
    #      ])

    # image_names_list = ti.xcom_pull(key='image_names', task_ids='convert_data')
    # realtime_uploaded_images = TraffyFondueImages(
    #     '../daily_images', transform, image_names_list)
    # realtime_uploaded_images_loader = torch.utils.data.DataLoader(
    #     realtime_uploaded_images, batch_size=32, shuffle=False)

    # predict = list()
    # model.eval()
    # with torch.no_grad():
    #     for inputs, _ in tqdm(realtime_uploaded_images_loader):
    #         inputs = inputs.to(device)
    #         outputs = model(inputs)
    #         predict += list(outputs.argmax(dim=1).cpu().numpy())

    ti.xcom_push(key='model_prediction', value="test1")
    urllib.request.urlretrieve(
        'https://storage.googleapis.com/traffy_public_bucket/attachment/2023-05/870f383a266b2d8fb101c550f46ca732e0a00963.jpg', "image.jpg")
    inputImage = Image.open('image.jpg')
    inputImage = inputImage.resize((224, 224))
    inputImage = np.array(inputImage.convert('RGB'))

    ti.xcom_push(key='model_prediction', value="test2")
    inputTensor = ((inputImage / 255) -
                   [0.4303, 0.4301, 0.4139]) / [0.2186, 0.2140, 0.2205]
    inputTensor = inputTensor.transpose(2, 0, 1)[np.newaxis].astype(np.float32)
    BASE_DIR = Path(__file__).resolve(strict=True).parent
    ti.xcom_push(key='model_prediction', value="test3")
    sessOptions = rt.SessionOptions()
    sessOptions.graph_optimization_level = rt.GraphOptimizationLevel.ORT_ENABLE_ALL
    model = rt.InferenceSession(os.path.join(BASE_DIR,"model.onnx"), sessOptions)
    thresholding_lambda = lambda x: 1 if x > 0.5 else 0
    ti.xcom_push(key='model_prediction', value="test4")

    output = model.run([], {'input': inputTensor})[0]
    print(output)
    output = np.vectorize(thresholding_lambda)(output)
    ti.xcom_push(key='model_prediction', value=str(output[0]))


def send_prediction_to_vis(ti):
    model_prediction = ti.xcom_pull(
        key='model_prediction', task_ids='feed_data_to_model')
    payload = ti.xcom_pull(key='traffyapi_data', task_ods='fetch_data')

    # Delete the directory and delete all data on metadata database
    shutil.rmtree('../daily_images')
    delete_all_xcom_entries()


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

# convert_task = PythonOperator(
#     task_id='convert_data',
#     python_callable=convert_images_to_jpg,
#     provide_context=True,
#     dag=dag
# )

delete_xcom = PythonOperator(
    task_id="delete_xcom",
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

post_task = PythonOperator(
    task_id='send_prediction_to_vis',
    python_callable=send_prediction_to_vis,
    provide_context=True,
    dag=dag
)

fetch_task >> delete_xcom >> feed_and_predict_task >> post_task
