import cv2
import numpy as np

class PlateDetector():

    def __init__(self, dy1=0.06, dy2=0.06):
        self._dy1 = dy1
        self._dy2 = dy2
        pass

    def _filter_blue(self, img):
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        l_range = np.array([100, 50, 100])
        u_range = np.array([130, 255, 140])

        mask = cv2.inRange(hsv, l_range, u_range)
        res = cv2.bitwise_and(img, img, mask=mask)
        return mask, res

    def _filter_gray(self, img):

        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        l_range = np.array([0, 0, 100])
        u_range = np.array([255, 20, 250])

        mask = cv2.inRange(hsv, l_range, u_range)
        res = cv2.bitwise_and(img, img, mask=mask)
        return mask, res

    def _check_contour(self, contour, img, debug=False):
        x, y, w, h = cv2.boundingRect(contour)

        cond1 = w > 35 and w < 210
        cond2 = h > 50 and h < 250

        area = cv2.contourArea(contour)
        cond3 = area > 5000 and area < 80000
        # cond4 = float(h) / w > 1
        cond5 = x > 0 and x+w > 0
        if debug == True:
            print("Checking: ")
            print(cond1)
            print(cond2)
            print(cond3)
            # print(cond4)
            print(float(h) / w)
            print(cond5)
            print(w*(h+int(self._dy2*y)))
        if cond1 and cond2 and cond3 and cond5:
            return True
        else:
            return False

    def validate(self, license_plate):
        bw = cv2.threshold(license_plate, 225, 255, cv2.THRESH_BINARY)[1]

        if np.sum(bw) > 30:
            return False
        else:
            return True

    def get_label_img(self, img, debug=False):
        mask, res = self._filter_gray(img)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        road = cv2.threshold(gray, 225, 255, cv2.THRESH_BINARY)[1]

        if debug == True:
            cv2.imshow("get_label_img", mask)
            cv2.waitKey(1000)

        # get contours of the masked image
        image, contours, hierarchy = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if len(contours) != 0:
            max_c = max(contours, key=cv2.contourArea)
            if self._check_contour(max_c, img, debug=debug):
                x, y, w, h = cv2.boundingRect(max_c)
                y1 = y+int(self._dy1*y)
                y2 = y+h+int(self._dy2*y)
                x1 = x
                x2 = x+w
                license_plate = img[y1:y2, x:x+w]

                if self.validate(license_plate):
                    if debug == True:
                        cv2.imshow("get_label_img", license_plate)
                        cv2.waitKey(1000)

                    return license_plate

        return None

    def draw_contour(self, img, debug=False):
        mask, res = self._filter_gray(img)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        road = cv2.threshold(gray, 225, 255, cv2.THRESH_BINARY)[1]

        # get contours of the masked image
        image, contours, hierarchy = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if len(contours) != 0:
            max_c = max(contours, key=cv2.contourArea)
            if self._check_contour(max_c, img):
                x, y, w, h = cv2.boundingRect(max_c)
                y1 = y+int(self._dy1*y)
                y2 = y+h+int(self._dy2*y)
                x1 = x
                x2 = x+w
                license_plate = img[y1:y2, x:x+w]

                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                if debug == True:
                    cv2.imshow("draw_contour", img)
                    cv2.waitKey(1000)

        return img


if __name__ == "__main__":
    bob = PlateDetector()

    for i in range(1, 9):
        img = cv2.imread(
            '/home/fizzer/Videos/Lisence_detector_data/{}.jpg'.format(i))
        bob.get_label_img(img, debug=True)
