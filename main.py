from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse
import cv2
import numpy as np
import io

app = FastAPI()

def convert_to_sketch(image_data):
    # Read the image data
    image_array = np.frombuffer(image_data, np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    
    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Invert the grayscale image
    inverted_gray_image = 255 - gray_image
    
    # Apply Gaussian blur
    blurred_image = cv2.GaussianBlur(inverted_gray_image, (21, 21), 0)
    
    # Invert the blurred image
    inverted_blurred_image = 255 - blurred_image
    
    # Convert to sketch
    sketch = cv2.divide(gray_image, inverted_blurred_image, scale=256.0)
    
    # Encode the sketch image to JPEG format
    _, sketch_data = cv2.imencode(".jpg", sketch)
    
    return sketch_data.tobytes()

@app.post("/convert_to_sketch/")
async def convert_to_sketch_api(file: UploadFile = File(...)):
    try:
        # Read the uploaded image
        image_data = await file.read()
        
        # Convert the image to sketch
        sketch_data = convert_to_sketch(image_data)
        
        # Return the sketch image as a downloadable file
        return StreamingResponse(io.BytesIO(sketch_data), media_type="image/jpeg", headers={"Content-Disposition": "attachment;filename=sketch.jpg"})
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
