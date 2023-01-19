import sys
sys.path.append('./src/models')

from fastapi import FastAPI
from http import HTTPStatus
from fastapi import UploadFile, File
from typing import Optional
from torchvision.transforms import transforms
from predict_model import predict

app = FastAPI()

@app.get("/")
def root():
    """ Health check."""
    response = {
        "message": HTTPStatus.OK.phrase,
        "status-code": HTTPStatus.OK,
    }
    return response


@app.post("/predict/")
async def cv_model(data: UploadFile = File(...), h: Optional[int] = 28, w: Optional[int] = 28):
    totensor = transforms.ToTensor()
    with open('image.jpg', 'wb') as image:
        content = await data.read()
        image.write(content)
        image.close()
    
    prob, label = predict('models/lightning/epoch=107-step=108.ckpt',"image.jpg")
   
    return {'Label': label, 'Probability': prob}




