import streamlit as st
import pandas as pd
import numpy as np
from apinew3 import *
from PIL import Image


import streamlit_authenticator as stauth



names = ['Akanksha','Briggs']
usernames = ['ak','rbriggs']
passwords = ['123','456']

hashed_passwords = stauth.Hasher(passwords).generate()

authenticator = stauth.Authenticate(names,usernames,hashed_passwords,
    'some_cookie_name','some_signature_key')

name, authentication_status, username = authenticator.login('Login','main')

if authentication_status:

    st.write('Welcome *%s*' % (name))
    st.title('NOWCASTING')



    #Text
    title = st.text_input('Location', ' ')
    st.write('The selected Location is', title)
    year=2019
    month=12
    date=12


    flag=0
    flag2=0
    def get_pred(location_name: str , year : int , month :int, day : int):
        #location_name = location_name
        #this_file_path = "./images/"+ str(location_name) + "/images"
        this_file_path = "./images/"+ str(location_name) +"/"
        p, flag = prediction(location_name, year, month, day)
        if flag ==1:
            flag2=1
            filep = os.path.join(this_file_path,"files/fin.jpg")
            if os.path.exists(filep):
                return FileResponse(filep , media_type="image/jpeg", filename= "myfin.jpg")
        elif flag ==0:
            return {"error:file not found"}

    img = get_pred(title,year,month,date)

    print("In streamlit 3--------------------")
    print(flag2)

    if os.path.exists('images/'+str(title)+ '/files/fin.jpg'):
        image = Image.open('images/'+str(title)+ '/files/fin.jpg')

        st.image(image, caption='Nowcasting for '+str(title))

        video_file = open('images/'+str(title)+'/myvideo.mp4', 'rb')
        video_bytes = video_file.read()

        st.video(video_bytes)
    else:
        st.write("Beyond range. Try again!")

elif authentication_status == False:
    st.error('Username/password is incorrect')
elif authentication_status == None:
    st.warning('Please enter your username and password')

        