import cv2
import numpy as np

def detect_text_regions(image_path, east_model_path="utils/frozen_east_text_detection.pb"):
    """Detects and segments text regions from an image using EAST."""
    
    # Load the image
    image = cv2.imread(image_path)
    orig = image.copy()
    (H, W) = image.shape[:2]

    # Convert to grayscale and enhance contrast
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    gray = clahe.apply(gray)

    # Apply thresholding and dilation
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    dilated = cv2.dilate(thresh, kernel, iterations=2)

    # Load EAST model
    layer_names = ["feature_fusion/Conv_7/Sigmoid", "feature_fusion/concat_3"]
    net = cv2.dnn.readNet(east_model_path)

    # Prepare image for EAST model
    newW, newH = (320, 320)
    rW, rH = W / float(newW), H / float(newH)
    image_resized = cv2.resize(image, (newW, newH))
    blob = cv2.dnn.blobFromImage(image_resized, 1.0, (newW, newH), (123.68, 116.78, 103.94), swapRB=True, crop=False)

    # Perform inference
    net.setInput(blob)
    (scores, geometry) = net.forward(layer_names)

    # Decode EAST output
    numRows, numCols = scores.shape[2:4]
    rects, confidences = [], []

    for y in range(numRows):
        scoresData = scores[0, 0, y]
        xData0, xData1, xData2, xData3, anglesData = geometry[0, 0:5, y]

        for x in range(numCols):
            if scoresData[x] < 0.2:
                continue

            offsetX, offsetY = x * 4.0, y * 4.0
            angle = anglesData[x]
            cos, sin = np.cos(angle), np.sin(angle)

            h, w = xData0[x] + xData2[x], xData1[x] + xData3[x]
            endX, endY = int(offsetX + (cos * xData1[x]) + (sin * xData2[x])), int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
            startX, startY = int(endX - w), int(endY - h)

            # Scale to original size
            startX, startY, endX, endY = int(startX * rW), int(startY * rH), int(endX * rW), int(endY * rH)
            rects.append((startX, startY, endX, endY))
            confidences.append(float(scoresData[x]))

    # Apply Non-Maximum Suppression (NMS)
    boxes = cv2.dnn.NMSBoxes(rects, confidences, 0.2, 0.4)
    detected_boxes = [rects[i] for i in boxes.flatten()]

    # Merge boxes to create full lines
    def merge_boxes(boxes, threshold=30):
        merged_boxes = []
        boxes = sorted(boxes, key=lambda x: x[1])

        for box in boxes:
            x1, y1, x2, y2 = box
            merged = False

            for i, mbox in enumerate(merged_boxes):
                mx1, my1, mx2, my2 = mbox

                if abs(y1 - my1) < threshold and abs(y2 - my2) < threshold:
                    merged_boxes[i] = (min(mx1, x1), min(my1, y1), max(mx2, x2), max(my2, y2))
                    merged = True
                    break

            if not merged:
                merged_boxes.append(box)

        return merged_boxes

    merged_boxes = merge_boxes(detected_boxes, threshold=30)

    # Expand bounding boxes slightly
    expansion_x, expansion_y = 10, 5
    segmented_images = []

    for (startX, startY, endX, endY) in merged_boxes:
        startX, startY = max(0, startX - expansion_x), max(0, startY - expansion_y)
        endX, endY = min(W, endX + expansion_x), min(H, endY + expansion_y)
        
        roi = orig[startY:endY, startX:endX]
        segmented_images.append(roi)

    return segmented_images
