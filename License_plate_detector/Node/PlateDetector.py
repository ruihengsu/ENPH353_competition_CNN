import cv2
import numpy as np


class PlateDetector():

    def __init__(self, dy1=0.07, dy2=0.09):
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
        hsv = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2HSV)

        l_range = np.array([100, 50, 50])
        u_range = np.array([130, 255, 255])

        mask = cv2.inRange(hsv, l_range, u_range)

        return mask

    def _get_gray_mask(self, img):

        hsv = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2HSV)

        l_range = np.array([0, 0, 100])
        u_range = np.array([0, 20, 250])

        mask = cv2.inRange(hsv, l_range, u_range)
        return mask

    def _check_contour(self, contour, img, debug=False):
        """
        This function needs to be hand-tuned
        """
        area = cv2.contourArea(contour)

        x, y, w, h = cv2.boundingRect(contour)

        cond1 = w > 50 and w < 400
        cond2 = h > 50 and h < 300
        cond3 = area > 3000 and area < 80000
        cond4 = x > 0 and x+w > 0
        cond5 = float(h)/w < 1.6

        if debug == True:
            print("Checking: ")
            print("w: {} h:{}".format(w,h))
            print("Area: {}".format(area))
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
        # cv2.imshow("val", bw)
        # cv2.waitKey(300)
        bmask = self._get_blue_mask(license_plate)

        blue_hist = np.sum(bmask, axis=1)
        max_blue_row = np.argmax(blue_hist)

        max_blue_near_top_edge = max_blue_row < int(
            len(blue_hist)/4.)
        
        try:
            blue_bottom_edge = blue_hist[-1] + blue_hist[-2] + blue_hist[-3] != 0
        except:
            return False 
        # print(np.sum(bw))
        # print(np.sum(bmask))
        # print(max_blue_near_top_edge)
        # print(blue_bottom_edge)

        if np.sum(bw) > 4000 or np.sum(bmask) < 5000 or max_blue_near_top_edge or blue_bottom_edge:
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
            max_list = sorted(contours, key=cv2.contourArea, reverse=True)[:3]
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
            max_list = sorted(contours, key=cv2.contourArea, reverse=True)[:3]
            for mac_c in max_list:

                if self._check_contour(mac_c, img, debug=debug):
                    x, y, w, h = cv2.boundingRect(mac_c)
                    y1 = y+int(self._dy1*y)
                    y2 = y+h+int(self._dy2*y)
                    x1 = x
                    x2 = x+w
                    license_plate = img[y1:y2, x:x+w]

                    # self._validate(license_plate)
                    # return license_plate
                    
                    if self._validate(license_plate):
                        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        if debug == True:
                            cv2.imshow("draw_contour", img)
                            cv2.waitKey(1000)

                        return img

        return None


if __name__ == "__main__":

    bob = PlateDetector()
    # img = cv2.imread("/home/fizzer/ENPH353_competition_CNN/License_plate_detector/Detector_data/9.jpg")
    # result = bob.get_label_img(img, True)
    # print(result)

    import imageio
    import logging
    import os
    import glob

    logging.basicConfig(level=logging.DEBUG)
    

    for v in [1,2,3,4,5,6]:
         
        vid = imageio.get_reader("/home/fizzer/ENPH353_competition_CNN/License_plate_detector/Detector_data/Test{}.mp4".format(v),
                                 'ffmpeg')

        result_path = "/home/fizzer/ENPH353_competition_CNN/License_plate_detector/Detector_data/Test{}_result".format(
            v)
        try:
            os.mkdir(result_path)
        except:
            pass
        files = glob.glob(result_path + "/*")
        for f in files:
            # print(f)
            os.remove(f)

        for i, im in enumerate(vid):
            img = cv2.cvtColor(im.copy(), cv2.COLOR_RGB2BGR)
            result = bob.get_label_img(img,False)
            if result is not None:
                cv2.imwrite(result_path + "/label_{}.jpg".format(i),
                            result)
        logging.debug("Test {} completed".format(v))