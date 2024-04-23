import cv2
import numpy as np

cap = cv2.VideoCapture(0)

lower_red = np.array([0,88,210])
upper_red = np.array([15,242,255])

lower_blue = np.array([83,36,155])
upper_blue = np.array([134,255,255])

lower_green = np.array([45,38,80])
upper_green = np.array([76,255,255])

lower_yellow = np.array([10,86,200])
upper_yellow = np.array([84,170,255])

List_Colors = [(lower_red, upper_red),
               (lower_blue, upper_blue),
               (lower_green, upper_green),
               (lower_yellow, upper_yellow)]

str_color = ["Red", "Blue", "Green", "Yellow"]

while True:
    ret, frame = cap.read()

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    if ret:
        roi = frame[10:450, 180:400]
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        for idx_color, (low, high) in enumerate(List_Colors):
            mask =cv2.inRange(hsv, low, high)
            contours , _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

            for i, c in enumerate(contours):
                area = cv2.contourArea(c)
                if area < 6500:
                    continue

                rect = cv2.minAreaRect(c)
                box = cv2.boxPoints(rect)
                box = np.intp(box)
                cv2.drawContours(roi, [box], 0, (221, 160, 221), 3)
            
                M = cv2.moments(c)
                if M["m00"] != 0:
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])
                    cv2.circle(roi, (cX, cY), 3, (0, 0, 0), -1)
                    cv2.putText(roi, str_color[idx_color], (cX - 25, cY - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 1)

        cv2.imshow("Frame", frame)
        cv2.imshow("Roi", roi)
     
cap.release()
cv2.destroyAllWindows()
