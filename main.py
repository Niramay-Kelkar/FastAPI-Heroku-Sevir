# main.py

from datetime import date, datetime
#from Assignment4.LocationFetch2 import prediction
from fastapi import FastAPI
from pydantic import BaseModel
import datetime
from LocationFetch2 import *

class Item(BaseModel):
    location: str
    year: int
    month: int
    day: int

app = FastAPI()

@app.get("/location")
def get_loc(item: Item):
    #datem = datetime.datetime.strptime(item.date, "%Y-%m-%d %H:%M:%S")
    this_file_path = "./images/"+ str(item.location) +"/"
    p = prediction(location_name=item.location, year=item.year, month=item.month, day=item.day)
    filep = os.path.join(this_file_path,"files/fin.jpg")
    if os.path.exists(filep):
        return FileResponse(filep , media_type="image/jpeg", filename= "myfin.jpg")
    return {"error:file not found"}
