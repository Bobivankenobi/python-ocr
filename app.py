import os
from flask import Flask, request, jsonify
import cv2
import numpy as np
from paddleocr import PaddleOCR

app = Flask(__name__)

# Initialize the OCR model outside the route to avoid loading it multiple times
ocr_model = PaddleOCR(lang='en')

@app.route('/')
def hello_world():
	return 'Hello World!'

@app.route('/extract-text', methods=['POST'])
def extract_text_from_image():
    if 'file' not in request.files:
        return jsonify({'error': 'no file provided'}), 400
    
    file = request.files['file']
    # Convert the uploaded file to a numpy array
    nparr = np.fromstring(file.read(), np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)

    # Use tempfile to create a temporary image file
    # This step is required as PaddleOCR's `ocr` method expects a file path
    import tempfile
    temp_filepath = tempfile.mktemp(suffix=".png")
    cv2.imwrite(temp_filepath, image)

    # Process the image using PaddleOCR
    result = ocr_model.ocr(temp_filepath)
    print(result)
    text_results = []

    for item in result:
        for line in item:
            # Extract the text part from the line
            text_part = line[1][0]
            if isinstance(text_part, str):
                text_results.append(text_part)
            elif isinstance(text_part, list):
                # Join the list into a single string
                text_results.append(' '.join(text_part))

    full_text = ' '.join(text_results)
    print(full_text)
    
    # Clean up the temporary file
    os.remove(temp_filepath)

    return jsonify({'text': full_text})

if __name__ == '__main__':
    app.run()

