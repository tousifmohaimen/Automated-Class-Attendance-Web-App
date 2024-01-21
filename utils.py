import streamlit as st
import redis
import numpy as np
import pandas as pd
import cv2
from datetime import datetime, timedelta
from insightface.app import FaceAnalysis
from sklearn.metrics import pairwise
import os
from streamlit_webrtc import webrtc_streamer
import av
import re
import time
import base64

redis_hostname = 'redis-14895.c74.us-east-1-4.ec2.cloud.redislabs.com'
redis_portnumber = 14895
redis_password = 'GNAkOLmU26w6MuyLBfqfAWJcHAXdba4r'
redis_client = redis.StrictRedis(
    host=redis_hostname,
    port=redis_portnumber,
    password=redis_password
)
faceapp = FaceAnalysis(name='buffalo_sc', root='insightface_model', providers=['CPUExecutionProvider'])
faceapp.prepare(ctx_id=0, det_size=(640, 640), det_thresh=0.5)

def ml_search_algorithm(dataframe, feature_column, test_vector,
                        name_id=['Name', 'Matric'], thresh=0.5):
    """
    cosine similarity base search algorithm
    """
    # step-1: take the dataframe (collection of data)
    dataframe = dataframe.copy()
    # step-2: Index face embeding from the dataframe and convert into array
    X_list = dataframe[feature_column].tolist()
    x = np.asarray(X_list)

    # step-3: Cal. cosine similarity
    similar = pairwise.cosine_similarity(x, test_vector.reshape(1, -1))
    similar_arr = np.array(similar).flatten()
    dataframe['cosine'] = similar_arr

    # step-4: filter the data
    data_filter = dataframe.query(f'cosine >= {thresh}')
    if len(data_filter) > 0:
        # step-5: get the person name
        data_filter.reset_index(drop=True, inplace=True)
        argmax = data_filter['cosine'].argmax()
        person_name, person_role = data_filter.loc[argmax][name_id]

    else:
        person_name = 'Unknown'
        person_role = 'Unknown'

    return person_name, person_role


class RealTimePred:
    def __init__(self, username):
        self.username = username
        self.logs_key = f'attendance:logs:{username}'
        self.logs = {'name': [], 'matric': [], 'current_time': []}
        self.last_update_time = time.time()

    def reset_dict(self):
        self.logs = {'name': [], 'matric': [], 'current_time': []}
        self.last_update_time = time.time()

    def save_logs_redis(self):
        if self.logs['name']:
            # Create a DataFrame from the logs
            logs_df = pd.DataFrame(self.logs)

            # Filter out duplicate entries based on the 'name' column
            logs_df.drop_duplicates('name', inplace=True)

            # Prepare the data for Redis
            encoded_data = []
            for _, row in logs_df.iterrows():
                name, matric, current_time = row['name'], row['matric'], row['current_time']
                concat_string = f"{name} : {matric} : {current_time}"
                encoded_data.append(concat_string)

            # Push data to the user-specific log key in Redis
            if encoded_data:
                redis_client.lpush(self.logs_key, *encoded_data)

            # Reset the logs dictionary
            self.reset_dict()

    def face_prediction(self, test_image, dataframe, feature_column,
                        name_id=['Name', 'Matric'], thresh=0.5):
        # Find the time
        current_time = str(datetime.now())

        # Use for loop and extract each embedding and pass to ml_search_algorithm
        for res in faceapp.get(test_image):
            x1, y1, x2, y2 = res['bbox'].astype(int)
            embeddings = res['embedding']
            person_name, person_role = ml_search_algorithm(dataframe,
                                                           feature_column,
                                                           test_vector=embeddings,
                                                           name_id=name_id,
                                                           thresh=thresh)
            if person_name == 'Unknown':
                color = (0, 0, 255)  # bgr
            else:
                color = (0, 255, 0)

            cv2.rectangle(test_image, (x1, y1), (x2, y2), color)
            text_gen = f"{person_name} - {person_role}"
            cv2.putText(test_image, text_gen, (x1, y1 - 10), cv2.FONT_HERSHEY_DUPLEX, 0.7, color, 2)
            cv2.putText(test_image, current_time, (x1, y2 + 10), cv2.FONT_HERSHEY_DUPLEX, 0.7, color, 2)

            # Save info in logs dict
            self.logs['name'].append(person_name)
            self.logs['matric'].append(person_role)
            self.logs['current_time'].append(current_time)

        return test_image

    def update_report(self):
        # Create a DataFrame from the logs
        logs_df = pd.DataFrame(self.logs)

        # Filter out duplicate entries based on the 'name' column
        logs_df.drop_duplicates('name', inplace=True)

        # Add the data to the report in Redis
        for _, row in logs_df.iterrows():
            name, matric, current_time = row['name'], row['matric'], row['current_time']
            concat_string = f"{name} : {matric} : {current_time}"
            redis_client.lpush('attendance:logs', concat_string)

        # Reset the logs dictionary
        self.reset_dict()
 

class RegistrationForm:
    def __init__(self):
        self.sample = 0

    def reset(self):
        self.sample = 0

    def get_embedding(self, frame):
        # get results from insightface model
        results = faceapp.get(frame, max_num=1)
        embeddings = None
        for res in results:
            self.sample += 1
            x1, y1, x2, y2 = res['bbox'].astype(int)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)
            # put text samples info
            text = f"samples = {self.sample}"
            cv2.putText(frame, text, (x1, y1), cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 0), 2)

            # facial features
            embeddings = res['embedding']

        return frame, embeddings

    def save_data_in_redis_db(self, name, matric):
        # validation name
        if name is not None:
            if name.strip() != '':
                key = f'{name} : {matric}'
            else:
                return 'name_false'
        else:
            return 'name_false'

        # if face_embedding.txt exists
        if 'face_embedding.txt' not in os.listdir():
            return 'file_false'

        # step-1: load "face_embedding.txt"
        x_array = np.loadtxt('face_embedding.txt', dtype=np.float32)  # flatten array

        # step-2: convert into array (proper shape)
        received_samples = int(x_array.size / 512)
        x_array = x_array.reshape(received_samples, 512)
        x_array = np.asarray(x_array)

        # step-3: cal. mean embeddings
        x_mean = x_array.mean(axis=0)
        x_mean = x_mean.astype(np.float32)
        x_mean_bytes = x_mean.tobytes()

        # step-4: save this into redis database
        # redis hashes
        redis_client.hset(name='academy:register', key=key, value=x_mean_bytes)

        #
        os.remove('face_embedding.txt')
        self.reset()

        return True
