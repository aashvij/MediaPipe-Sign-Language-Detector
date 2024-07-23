import cv2
import mediapipe as mp
import os

#init mediapipe hands
handsmp = mp.solutions.hands
hands = handsmp.Hands(
    static_image_mode=True,
    max_num_hands = 2,
    min_detection_confidence = 0.8,
    min_tracking_confidence = 0.6,
)
#drawing utils
drawingmp = mp.solutions.drawing_utils
drawingmp_styles = mp.solutions.drawing_styles

# STATIC IMAGE LANDMARKS
inputDir = "./practiceImgs"
outDir = "./annotated"


for file in os.listdir(inputDir):
    if file.endswith(".jpeg"):
        path = os.path.join(inputDir, file)
        image = cv2.flip(cv2.imread(path), 1) # 1 for flipping on y-axis
        # image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        annotatedImage = image.copy()

        if not results.multi_hand_landmarks:
            continue

        for hand_landmarks in results.multi_hand_landmarks:
            drawingmp.draw_landmarks(annotatedImage, hand_landmarks, handsmp.HAND_CONNECTIONS)
            print(file)
        
        outPath = os.path.join(outDir, file)
        cv2.imwrite(outPath, annotatedImage)
        print(outPath)
        cv2.imshow('Hand Landmarks', annotatedImage)
        cv2.waitKey(0)



# # VIDEO CAPTURE LANDMARKS # #
# cap = cv2.VideoCapture(0)
# while cap.isOpened():
#     data, image = cap.read()
#     if not data: 
#         break

#     image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
#     results = hands.process(image)
#     # draw landmarks
#     image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
#     if results.multi_hand_landmarks:
#         for hand_landmarks in results.multi_hand_landmarks:
#             drawingmp.draw_landmarks(
#                 image,
#                 hand_landmarks,
#                 handsmp.HAND_CONNECTIONS,
#             )
#     cv2.imshow("Mediapipe Hands", image)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
# cap.release()



