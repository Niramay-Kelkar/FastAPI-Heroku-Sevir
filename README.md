# FastAPI-Heroku-Sevir
Deployment of SEVIR Nowcasting System for Insurance companies using Heroku

#### Fast api:
- Takes Location and Timestamp.
- Fetches corresponding sevir data using boto3 from aws 
- Generates nowcast_testing.h5 file using "nowcast test generator"
- Uses mse model to predict the next 12 images
- Creates a merged image and a gif
- API gives an option to download the image
$ uvicorn apinew:app --reload


#### Streamlit:
$ streamlit run streamlit.py

- It will run on http://localhost:8501/ 
- Streamlit shows the visual outputs upon providing the location from the client


### References:
- https://docs.streamlit.io/library/api-reference/media/st.video
- https://www.youtube.com/watch?v=lzp6YvJMRL4
