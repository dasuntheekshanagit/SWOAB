import cv2
import numpy as np

def main():
    print("NotePad 1.1")

    img = cv2.imread("Images/20230212_134143.jpg")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.medianBlur(gray, 5)

    thresh = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,2)
    kernel = np.ones((5,5),np.uint8)
    errosion = cv2.erode(thresh,kernel,iterations=1)

    kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (2,2))
    dilate = cv2.dilate(errosion, kernel2, iterations=3)
    cnts = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]

    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)

    for c in cnts:
        x,y,w,h = cv2.boundingRect(c)
        ROI = img[y:y+h, x:x+w]
        #cv2.imwrite('ROI.png', ROI)
        break

    cv2.drawContours(dilate, cnts, -1, (0, 255, 0), 3)
    cv2.namedWindow('dialate', cv2.WINDOW_NORMAL)
    cv2.imshow("dialate",ROI)
    cv2.namedWindow('imagegray', cv2.WINDOW_NORMAL)
    cv2.imshow("imagegray",dilate)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
if __name__ == '__main__':
    main()
