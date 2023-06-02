#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import joblib 
from keras.models import load_model


model = load_model('Ukrain_and_Russia_victims_and_subevent_predictor.h5')
label_encoders = joblib.load('Label_Encoders')


def predictor (category_arr, numeric_arr):

    x_cat_features = ['disorder_type', 'event_type', 'actor1', 'assoc_actor_1', 'actor2', 'assoc_actor_2',
                      'country', 'admin1', 'admin2', 'admin3', 'location']
    x_num_val = np.array(list(numeric_arr.values()), dtype=np.float32)
    x_cat_val = []

    # Encode the values from category_arr to labels
    for i, (feature, val) in enumerate(zip(x_cat_features, category_arr)):
        encoded_val = label_encoders[feature].transform([val])[0]
        x_cat_val.append(encoded_val)

    x_cat_val = np.array(x_cat_val, dtype=np.int32)
    x_cat_val = np.reshape(x_cat_val, (1, -1))
    x_num_val = np.reshape(x_num_val, (1, -1))
    
    # Predict the values using the model
    batch_size = 32
    predictions = model.predict([x_cat_val, x_num_val], batch_size=batch_size)
    sub_event_type = np.argmax(predictions[0], axis=1)
    civilian_targeting = np.argmax(predictions[1], axis=1)

    # Convert the results to string
    sub_event_type = label_encoders['sub_event_type'].inverse_transform(sub_event_type)
   
    if civilian_targeting == 0:
        civilian_targeting_resp = 'No'
    else:
        civilian_targeting_resp = 'Yes'

    return sub_event_type[0], civilian_targeting_resp

