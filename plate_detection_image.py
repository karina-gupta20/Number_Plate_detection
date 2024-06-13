import cv2
import pytesseract
import openpyxl
import os

# Define the width and height of the video frames
width = 640
height = 480

# Load the Haar cascade classifier for license plate detection
plate_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_russian_plate_number.xml')

# Read the input image
image_path = r"E:\pythonProject\face_detection\face_data\Licence_plate_detection-main\car_plate.jpg"
image = cv2.imread(image_path)

# Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Detect license plates in the image
plates = plate_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

# Create a new Excel workbook
workbook = openpyxl.Workbook()
sheet = workbook.active
sheet["A1"] = "Image Path"
sheet["B1"] = "Detected Text"
sheet["C1"] = "Fine"

# Create a directory to store the images
image_dir = "detected_images"
os.makedirs(image_dir, exist_ok=True)

# Iterate over the detected license plates
for idx, (x, y, w, h) in enumerate(plates, start=2):  # Start from row 2
    cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
    # Extract the detected license plate region
    plate_region = gray[y:y + h, x:x + w]

    cv2.imshow('Number Plate: ', plate_region)

    # Use pytesseract to perform OCR on the grayscale image
    pytesseract.pytesseract.tesseract_cmd = r'E:\pythonProject\tesseract.exe'
    text = pytesseract.image_to_string(plate_region)

    # Save the extracted region image with detected text
    image_filename = os.path.join(image_dir, f"plate_{idx}.jpg")
    cv2.imwrite(image_filename, image)

    # Write image path and detected text to Excel
    sheet[f"A{idx}"] = image_filename
    sheet[f"B{idx}"] = text
    sheet[f"C{idx}"] = "500"

# Save the Excel workbook
workbook.save("detected_text.xlsx")

cv2.imshow('Plate Detection', image)

# Release resources
workbook.close()
cv2.waitKey(0)
cv2.destroyAllWindows()
