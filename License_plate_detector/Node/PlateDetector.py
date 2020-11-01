import cv2
import numpy as np


class PlateDetector():

    def __init__(self, dy1=0.06, dy2=0.08):
        """
        keyword arguments:

        dy1 -- given a bounding rectangle, determines how much to
        shift the top edge of the rectangle down to remove white space

        dy2 -- given a bounding rectangle, determines how much to 
        shift the bottom edge down so the liscene plate is included
        """
        self._dy1 = dy1
        self._dy2 = dy2
        pass

    def _get_blue_mask(self, img):
        hsv = cv2.cvtColor(img.copy(), cv2.COLOR_RGB2HSV)

        l_range = np.array([100, 50, 100])
        u_range = np.array([130, 255, 140])

        mask = cv2.inRange(hsv, l_range, u_range)

        return mask

    def _get_gray_mask(self, img):

        hsv = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2HSV)

        l_range = np.array([0, 0, 100])
        u_range = np.array([255, 20, 250])

        mask = cv2.inRange(hsv, l_range, u_range)
        return mask

    def _check_contour(self, contour, img, debug=False):

        area = cv2.contourArea(contour)

        x, y, w, h = cv2.boundingRect(contour)

        cond1 = w > 50 and w < 210
        cond2 = h > 60 and h < 250
        cond3 = area > 4000 and area < 80000
        cond4 = x > 0 and x+w > 0
        cond5 = float(h)/w < 1.4

        if debug == True:
            print("Checking: ")
            print(cond1)
            print(cond2)
            print(cond3)
            print(area)
            print(cond4)
            print(cond5)
        if cond1 and cond2 and cond3 and cond4 and cond5:
            return True
        else:
            return False

    def _validate(self, license_plate, debug=False):
        bw = cv2.threshold(license_plate.copy(), 225,
                           255, cv2.THRESH_BINARY)[1]
        bmask = self._get_blue_mask(license_plate)

        blue_hist = np.sum(bmask, axis=1)
        max_blue_row = np.argmax(blue_hist)

        max_blue_near_edge = max_blue_row < int(
            len(blue_hist)/4.) or max_blue_row > int(3.0*len(blue_hist)/4.)

        if np.sum(bw) > 0 or np.sum(bmask) < 7000 or max_blue_near_edge:
            return False
        else:
            return True

    def get_label_img(self, img, debug=False):
        mask = self._get_gray_mask(img)

        if debug == True:
            cv2.imshow("get_label_img", mask)
            cv2.waitKey(1000)

        # get contours of the masked image
        image, contours, hierarchy = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if len(contours) >= 5:
            max_list = sorted(contours, key=cv2.contourArea, reverse=True)[:5]
            for mac_c in max_list:

                if self._check_contour(mac_c, img, debug=debug):
                    x, y, w, h = cv2.boundingRect(mac_c)
                    y1 = y+int(self._dy1*y)
                    y2 = y+h+int(self._dy2*y)
                    x1 = x
                    x2 = x+w
                    license_plate = img[y1:y2, x:x+w]
                    if debug == True:
                        cv2.imshow("get_label_img", license_plate)
                        cv2.waitKey(1000)

                    if self._validate(license_plate, debug=True):
                        return license_plate

        return None

    def draw_contour(self, img, debug=False):

        mask = self._get_gray_mask(img)

        # get contours of the masked image
        image, contours, hierarchy = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if len(contours) >= 5:
            max_list = sorted(contours, key=cv2.contourArea, reverse=True)[:5]
            for mac_c in max_list:

                if self._check_contour(mac_c, img, debug=debug):
                    x, y, w, h = cv2.boundingRect(mac_c)
                    y1 = y+int(self._dy1*y)
                    y2 = y+h+int(self._dy2*y)
                    x1 = x
                    x2 = x+w
                    license_plate = img[y1:y2, x:x+w]
                    if self._validate(license_plate):
                        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        if debug == True:
                            cv2.imshow("draw_contour", img)
                            cv2.waitKey(1000)

                        return img

        return img


if __name__ == "__main__":

    bob = PlateDetector()
    
    import imageio
    import logging
    logging.basicConfig(level=logging.DEBUG)
    
    for v in [1, 2, 3, 4, 5]:

        vid = imageio.get_reader("/home/fizzer/ENPH353_competition_CNN/License_plate_detector/Detector_data/Test{}.mp4".format(v),
                                 'ffmpeg')

        result_path = "/home/fizzer/ENPH353_competition_CNN/License_plate_detector/Detector_data/Test{}_result".format(
            v)

        for i, im in enumerate(vid):
            result = bob.get_label_img(im)
            if result is not None:
                cv2.imwrite(result_path + "/label_{}.jpg".format(i),
                            cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
        logging.debug("Test {} completed".format(v))