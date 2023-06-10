import cv2
import numpy as np

def main():
    print("NotePad 1.1")

    img = cv2.imread("Images/20230212_134132.jpg")

    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    #cv2.imshow("image",img)
    hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    #cv2.imshow("image",hsv)
    lower_blue = np.array([20,50,50])
    upper_blue = np.array([110,255,255])
    mask = cv2.inRange(hsv,lower_blue,upper_blue)
    th2 = cv2.adaptiveThreshold(mask,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,2)
    kernel = np.ones((5,5),np.uint8)
    errosion = cv2.erode(mask,kernel,iterations=1)
    opening = cv2.morphologyEx(errosion, cv2.MORPH_OPEN, kernel)

    cnts = cv2.findContours(errosion, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]

    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
    img2 = np.full(img.shape,255, np.uint8)
    cv2.drawContours(img2, cnts, -1, (0, 0, 0), -1)
    cv2.imshow("image",img2)
    #for i in img:
     #   print(i)

    cv2.waitKey(0)

    cv2.destroyAllWindows()
if __name__ == '__main__':
    main()
