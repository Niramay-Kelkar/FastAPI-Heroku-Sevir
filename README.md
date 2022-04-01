# FastAPI-Heroku-Sevir
Deployment of SEVIR Nowcasting System for Insurance companies using Heroku

#### Fast api:
$ uvicorn apinew:app --reload

- Takes Location and Timestamp.
- Fetches corresponding sevir data using boto3 from aws 
- Generates nowcast_testing.h5 file using "nowcast test generator"
- Uses mse model to predict the next 12 images
- Creates a merged image and a gif
- API gives an option to download the image



#### Streamlit:
$ streamlit run streamlit.py

- It will run on http://34.134.17.190:8501/ 
- Streamlit shows the visual outputs(merged image and gif) upon providing the location from the client


### References:
- https://docs.streamlit.io/library/api-reference/media/st.video
- https://www.youtube.com/watch?v=lzp6YvJMRL4


“WE ATTEST THAT WE HAVEN’T USED ANY OTHER STUDENTS’ WORK IN OUR ASSIGNMENT AND ABIDE BY THE POLICIES LISTED IN THE STUDENT HANDBOOK”
