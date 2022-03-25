import streamlit as st
import pandas as pd
import numpy as np
import os
#from apinew import *

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

from PIL import Image
image = Image.open('images/'+str(title)+ '/files/fin.jpg')

st.image(image, caption='Nowcasting for '+str(title))

video_file = open('images/'+str(title)+'/myvideo.mp4', 'rb')
video_bytes = video_file.read()

st.video(video_bytes)
st.button('Download')
    