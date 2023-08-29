from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import tensorflow as tf
from io import BytesIO
from PIL import Image
import numpy as np


app = FastAPI()

origins = [
    "http://localhost",
    "http://localhost:3000",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# product_model = "E:/Deep Learning/Projects/6_Potato_decease_recognition/My_Model/1"
# alpha_model = "E:/Deep Learning/Projects/6_Potato_decease_recognition/My_Model/2"
# beta_model = "E:/Deep Learning/Projects/6_Potato_decease_recognition/My_Model/3"

path = "E:/deep_learning/projects/potato_decease_recognition/saved_models/1"
MODEL = tf.keras.models.load_model(path)
CLASS_NAMES = ['Early_blight', 'Late_blight', 'healthy']

@app.get('/ping')
async def ping():
    return "This is FastAPI"

def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    return image

@app.post('/predict')
async def predict(
    file: UploadFile = File(...)
): 

   image = read_file_as_image(await file.read())
   img_batch = np.expand_dims(image, 0)
   predictions = MODEL.predict(img_batch)
   
   index =  np.argmax(predictions[0])

   predicted_class = CLASS_NAMES[index]
   confidence = np.max(predictions[0])
   
   print(confidence, predicted_class)
   return {
            "class": predicted_class, 
            "Confidence": f"{np.round((float(confidence)*100), 1)}%"
    }


if __name__ == '__main__':
    uvicorn.run(app, host='localhost', port=8000)