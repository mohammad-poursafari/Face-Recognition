import cv2

# Load the pre-trained YuNet model for face detection
model_path = "face_detection_yunet_2026may.onnx"
detector = cv2.FaceDetectorYN.create(
    model_path, 
    "", 
    (320, 320), 
    score_threshold=0.9, 
    nms_threshold=0.3, 
    top_k=5000
)

# Initialize webcam capture
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Webcam not found!")
    exit()

print("Webcam started. Press 'q' to quit.")

# Real-time detection loop
while True:
    ret, frame = cap.read()
    if not ret:
        break
        
    # Get frame dimensions
    h, w = frame.shape[:2]
    
    # Set input size for the detector based on current frame size
    detector.setInputSize((w, h))
    
    # Perform face detection
    _, faces = detector.detect(frame)
    
    # If faces are detected, draw bounding boxes
    if faces is not None:
        for face in faces:
            # Extract coordinates and dimensions of the face
            x, y, w_face, h_face = map(int, face[:4])
            
            # Draw a green rectangle around the detected face
            cv2.rectangle(frame, (x, y), (x + w_face, y + h_face), (0, 255, 0), 2)
            
    # Display the resulting frame
    cv2.imshow("Face Detection (YoNet)", frame)
    
    # Exit the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
