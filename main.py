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

def convert_to_cartoon(image_data: bytes) -> bytes:
    # Convert bytes to a numpy array
    nparr = np.frombuffer(image_data, np.uint8)
    # Convert numpy array to image
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Apply cartoon effect
    # Step 1: Apply a bilateral filter to reduce the color palette of the image.
    color = cv2.bilateralFilter(img, d=9, sigmaColor=75, sigmaSpace=75)

    # Step 2: Convert to grayscale and apply median blur
    gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 7)

    # Step 3: Use adaptive thresholding to create an edge mask
    edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, blockSize=9, C=2)

    # Step 4: Combine the edge mask with the color image
    cartoon = cv2.bitwise_and(color, color, mask=edges)

    # Step 5 (Optional): Increase color vibrancy
    # This step increases the vibrancy of the colors in the cartoon
    # by converting to HSV, scaling up the saturation, and then converting back to BGR.
    cartoon_hsv = cv2.cvtColor(cartoon, cv2.COLOR_BGR2HSV)
    cartoon_hsv[:, :, 1] = cv2.multiply(cartoon_hsv[:, :, 1], 1.3)
    cartoon = cv2.cvtColor(cartoon_hsv, cv2.COLOR_HSV2BGR)

    # Encode the cartoon image to bytes
    _, cartoon_data = cv2.imencode(".jpg", cartoon)
    
    return cartoon_data.tobytes()

@app.post("/convert_to_cartoon/")
async def convert_to_cartoon_api(file: UploadFile = File(...)):
    try:
        # Read the uploaded image
        image_data = await file.read()
        
        # Convert the image to cartoon
        cartoon_data = convert_to_cartoon(image_data)
        
        # Return the cartoon image as a downloadable file
        return StreamingResponse(io.BytesIO(cartoon_data), media_type="image/jpeg", headers={"Content-Disposition": "attachment;filename=cartoon.jpg"})
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
