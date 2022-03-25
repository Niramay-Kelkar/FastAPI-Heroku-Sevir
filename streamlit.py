import streamlit as st
import pandas as pd
import numpy as np
import os
from LocationFetch2 import prediction
from apinew import *

st.title('NOWCASTING')

#Text
title = st.text_input('Location', ' ')
st.write('The selected Location is', title)
year=2019
month=12
date=12

def get_pred(location_name: str , year : int , month :int, day : int):
    #location_name = location_name
    #this_file_path = "./images/"+ str(location_name) + "/images"
    this_file_path = "./images/"+ str(location_name) +"/"
    p = prediction(location_name, year, month, day)
    filep = os.path.join(this_file_path,"files/fin.jpg")
    if os.path.exists(filep):
      return FileResponse(filep , media_type="image/jpeg", filename= "myfin.jpg")
    return {"error:file not found"}

img = get_pred(title,year,month,date)

def prediction(location_name, year, month, day):
  # Calling the location function to get the path of the downloaded h5 file
  data = Location(location_name)
      
  if __name__ == '__main__':
      log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
      logging.basicConfig(level=logging.INFO, format=log_fmt)
      main()

  model = "./models/mse_model.h5"
  mse_model = tf.keras.models.load_model(model,compile=False,custom_objects={"tf":tf})

  x_test, y_test = read_data('./nowcast_testing.h5', end=50)

  loc = randint(10,19)
  y_pred = mse_model.predict(x_test)
  if isinstance(y_pred,(list,)):
    y_pred=y_pred[0]
  y_preds.append(y_pred+norm['scale']+norm['shift'])

  res = imgsave(loc ,location_name, y_preds)
  return res

from PIL import Image
image = Image.open('images/'+str(title)+ '/files/fin.jpg')

st.image(image, caption='Nowcasting for '+str(title))

video_file = open('images/'+str(title)+'/myvideo.mp4', 'rb')
video_bytes = video_file.read()

st.video(video_bytes)
st.button('Download')
    