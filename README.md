# License Plate Detection

This project is aimed at detecting license plates in images using computer vision techniques. It utilizes OpenCV for image processing and pytesseract for optical character recognition (OCR).

## Requirements

- Python 3.x
- OpenCV
- pytesseract
- Tesseract OCR Engine
- openpyxl

You can install the Python dependencies using pip:

```
pip install opencv-python pytesseract
```

For Tesseract OCR, you need to install it separately. You can download it from [here](https://github.com/tesseract-ocr/tesseract) and follow the installation instructions for your operating system.

## Usage

1. Clone the repository or download the source code.

2. Install the required Python packages as mentioned above.

3. Make sure you have the trained Haar cascade XML file for license plate detection. You can find one for Russian license plates in the OpenCV repository.

4. Run the `plate_detection_image.py` script:

## Output

The script will display the input image with detected license plates highlighted by rectangles. Additionally, it will print the detected text from each license plate region in an excel sheet called "detected_text.xlsx" and the image of the vehicle will be store in a folder called "detected images".
