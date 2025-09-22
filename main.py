import cv2
import numpy as np
import pytesseract

# Set the Tesseract path (required for Windows)
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Define the license plate recognition function
def get_license_plate(frame, car_box):
    # ... (paste the code from above) ...
    x, y, w, h = car_box
    car_roi = frame[y:y+h, x:x+w]
    gray_roi = cv2.cvtColor(car_roi, cv2.COLOR_BGR2GRAY)
    blurred_roi = cv2.GaussianBlur(gray_roi, (5, 5), 0)
    edged_roi = cv2.Canny(blurred_roi, 100, 200)
    contours, _ = cv2.findContours(edged_roi.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

    plate_text = ""
    for c in contours:
        perimeter = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * perimeter, True)
        if len(approx) == 4:
            x_plate, y_plate, w_plate, h_plate = cv2.boundingRect(approx)
            aspect_ratio = w_plate / float(h_plate)
            if aspect_ratio > 2.5 and aspect_ratio < 6.0:
                license_plate_image = gray_roi[y_plate:y_plate+h_plate, x_plate:x_plate+w_plate]
                plate_text = pytesseract.image_to_string(license_plate_image, config='--psm 8')
                break
    return plate_text

# Load the Haar Cascade for car detection
car_cascade = cv2.CascadeClassifier('haarcascade_car.xml')
if car_cascade.empty():
    print("Error: Could not load the Haar Cascade file.")
    exit()

# Step 2: Open the video file
video_path = 'traffic.mp4'  # <-- Use your video file name
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print(f"Error: Could not open video file at {video_path}")
    exit()

# Initialize vehicle count
vehicle_count = 0
tracked_vehicles = []
counting_line_y = 400

# Step 3: The main loop to process frames
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (640, 480))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cars = car_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    cv2.line(frame, (0, counting_line_y), (frame.shape[1], counting_line_y), (255, 0, 0), 2)

    for (x, y, w, h) in cars:
        center_x = x + w // 2
        center_y = y + h // 2

        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Get the license plate text
        plate_text = get_license_plate(frame, (x, y, w, h))
        if plate_text:
            cleaned_plate = "".join(filter(str.isalnum, plate_text))
            cv2.putText(frame, cleaned_plate, (x, y - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        if center_y > counting_line_y:
            already_counted = False
            for tracked_vehicle in tracked_vehicles:
                distance = np.sqrt(((center_x - tracked_vehicle[0])**2) + ((center_y - tracked_vehicle[1])**2))
                if distance < 50:
                    already_counted = True
                    break
            if not already_counted:
                vehicle_count += 1
                tracked_vehicles.append((center_x, center_y))
                tracked_vehicles = [(center_x, center_y)]

    cv2.putText(frame, f'Vehicles: {vehicle_count}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.imshow('Vehicle Detection and LPR', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
