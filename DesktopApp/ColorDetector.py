import numpy as np
import cv2


class ColorDetector(object):

    def __init__(self):  # Basic Colors - added +-25 tolerance ratio
        self.black_lower = np.array([0, 0, 0], dtype="uint8")
        self.black_upper = np.array([25, 25, 25], dtype="uint8")

        self.white_lower = np.array([225, 225, 225], dtype="uint8")
        self.white_upper = np.array([255, 255, 255], dtype="uint8")

        self.red_lower = np.array([0, 0, 230], dtype="uint8")
        self.red_upper = np.array([25, 25, 255], dtype="uint8")

        self.lime_lower = np.array([0, 230, 0], dtype="uint8")
        self.lime_upper = np.array([25, 255, 25], dtype="uint8")

        self.blue_lower = np.array([230, 0, 0], dtype="uint8")
        self.blue_upper = np.array([255, 25, 25], dtype="uint8")

        self.yellow_lower = np.array([0, 230, 230], dtype="uint8")
        self.yellow_upper = np.array([25, 255, 255], dtype="uint8")

        self.cyan_lower = np.array([230, 230, 0], dtype="uint8")
        self.cyan_upper = np.array([255, 255, 25], dtype="uint8")

        self.magenda_lower = np.array([230, 0, 230], dtype="uint8")
        self.magenda_upper = np.array([255, 25, 255], dtype="uint8")

        self.silver_lower = np.array([167, 167, 167], dtype="uint8")  # Silver(192,192,192) +-25
        self.silver_upper = np.array([217, 217, 217], dtype="uint8")

        self.gray_lower = np.array([103, 103, 103], dtype="uint8")  # gray(128,128,128) +-25
        self.gray_upper = np.array([153, 153, 153], dtype="uint8")

        self.maroon_lower = np.array([0, 0, 103], dtype="uint8")  # maroon(128,0,0) +-25
        self.maroon_upper = np.array([25, 25, 153], dtype="uint8")

        self.olive_lower = np.array([0, 103, 103], dtype="uint8")  # olive(128,128,0) +-25
        self.olive_upper = np.array([25, 153, 153], dtype="uint8")

        self.green_lower = np.array([0, 103, 0], dtype="uint8")  # green(0,128,0) +-25
        self.green_upper = np.array([25, 153, 25], dtype="uint8")

        self.purple_lower = np.array([103, 0, 103], dtype="uint8")  # purple(128,0,128) +-25
        self.purple_upper = np.array([153, 25, 153], dtype="uint8")

        self.teal_lower = np.array([103, 103, 0], dtype="uint8")  # teal(0,128,128) +-25
        self.teal_upper = np.array([153, 153, 25], dtype="uint8")

        self.navy_lower = np.array([103, 0, 0], dtype="uint8")  # navy(0,0,128) +-25
        self.navy_upper = np.array([153, 25, 25], dtype="uint8")

        self.skyblue_lower = np.array([210, 130, 60], dtype="uint8")  # skyblue(135,206,235) +-25
        self.skyblue_upper = np.array([255, 230, 160], dtype="uint8")

        # Reference: https://www.rapidtables.com/web/color/RGB_Color.html

        self.kernal = np.ones((5, 5), "uint8")

    def black_detection(self, image, hsv, image_det=True):

        if image_det == False:
            black_mask = cv2.inRange(hsv, self.black_lower, self.black_upper)
        else:
            black_mask = cv2.inRange(image, self.black_lower, self.black_upper)

        black_mask = cv2.dilate(black_mask, self.kernal)
        output = cv2.bitwise_and(image, image, mask=black_mask)
        # Tracking the black Color
        im2, contours, hierarchy = cv2.findContours(black_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        for pic, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            if (area > 300):
                x, y, w, h = cv2.boundingRect(contour)
                image = cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
                cv2.putText(image, "BLACK", (x, y), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 0, 255))

        return image

    def white_detection(self, image, hsv, image_det=True):

        if image_det == False:
            white_mask = cv2.inRange(hsv, self.white_lower, self.white_upper)
        else:
            white_mask = cv2.inRange(image, self.white_lower, self.white_upper)

        white_mask = cv2.dilate(white_mask, self.kernal)
        output = cv2.bitwise_and(image, image, mask=white_mask)

        # Tracking the Red Color
        im2, contours, hierarchy = cv2.findContours(white_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        for pic, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            if (area > 300):
                x, y, w, h = cv2.boundingRect(contour)
                image = cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
                cv2.putText(image, "WHITE", (x, y), cv2.FONT_HERSHEY_DUPLEX, 0.9, (255, 255, 255))

        return image

    def red_detection(self, image, hsv, image_det=True):

        if image_det == False:
            red_mask = cv2.inRange(hsv, self.red_lower, self.red_upper)
        else:
            red_mask = cv2.inRange(image, self.red_lower, self.red_upper)

        red_mask = cv2.dilate(red_mask, self.kernal)
        output = cv2.bitwise_and(image, image, mask=red_mask)
        # Tracking the Red Color
        im2, contours, hierarchy = cv2.findContours(red_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        for pic, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            if (area > 300):
                x, y, w, h = cv2.boundingRect(contour)
                image = cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
                cv2.putText(image, "RED", (x, y), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 0, 255))

        return image

    def lime_detection(self, image, hsv, image_det=True):

        if image_det == False:
            lime_mask = cv2.inRange(hsv, self.lime_lower, self.lime_upper)
        else:
            lime_mask = cv2.inRange(image, self.lime_lower, self.lime_upper)

        lime_mask = cv2.dilate(lime_mask, self.kernal)
        output = cv2.bitwise_and(image, image, mask=lime_mask)
        # Tracking the lime Color
        im2, contours, hierarchy = cv2.findContours(lime_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        for pic, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            if (area > 300):
                x, y, w, h = cv2.boundingRect(contour)
                image = cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
                cv2.putText(image, "LIME", (x, y), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 0, 255))

        return image

    def blue_detection(self, image, hsv, image_det=True):

        if image_det == False:
            blue_mask = cv2.inRange(hsv, self.blue_lower, self.blue_upper)
        else:
            blue_mask = cv2.inRange(image, self.blue_lower, self.blue_upper)

        blue_mask = cv2.dilate(blue_mask, self.kernal)
        output = cv2.bitwise_and(image, image, mask=blue_mask)

        # Tracking the Red Color
        im2, contours, hierarchy = cv2.findContours(blue_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        for pic, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            if (area > 300):
                x, y, w, h = cv2.boundingRect(contour)
                image = cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
                cv2.putText(image, "BLUE", (x, y), cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 0, 0))

        return image

    def yellow_detection(self, image, hsv, image_det=True):

        if image_det == False:
            yellow_mask = cv2.inRange(hsv, self.yellow_lower, self.yellow_upper)
        else:
            yellow_mask = cv2.inRange(image, self.yellow_lower, self.yellow_upper)

        yellow_mask = cv2.dilate(yellow_mask, self.kernal)
        output = cv2.bitwise_and(image, image, mask=yellow_mask)
        # Tracking the yellow Color
        im2, contours, hierarchy = cv2.findContours(yellow_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        for pic, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            if (area > 300):
                x, y, w, h = cv2.boundingRect(contour)
                image = cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
                cv2.putText(image, "YELLOW", (x, y), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 0, 255))

        return image

    def cyan_detection(self, image, hsv, image_det=True):

        if image_det == False:
            cyan_mask = cv2.inRange(hsv, self.cyan_lower, self.cyan_upper)
        else:
            cyan_mask = cv2.inRange(image, self.cyan_lower, self.cyan_upper)

        cyan_mask = cv2.dilate(cyan_mask, self.kernal)
        output = cv2.bitwise_and(image, image, mask=cyan_mask)
        # Tracking the cyan Color
        im2, contours, hierarchy = cv2.findContours(cyan_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        for pic, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            if (area > 300):
                x, y, w, h = cv2.boundingRect(contour)
                image = cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
                cv2.putText(image, "CYAN/AQUA", (x, y), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 0, 255))

        return image

    def magenda_detection(self, image, hsv, image_det=True):

        if image_det == False:
            magenda_mask = cv2.inRange(hsv, self.magenda_lower, self.magenda_upper)
        else:
            magenda_mask = cv2.inRange(image, self.magenda_lower, self.magenda_upper)

        magenda_mask = cv2.dilate(magenda_mask, self.kernal)
        output = cv2.bitwise_and(image, image, mask=magenda_mask)
        # Tracking the magenda Color
        im2, contours, hierarchy = cv2.findContours(magenda_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        for pic, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            if (area > 300):
                x, y, w, h = cv2.boundingRect(contour)
                image = cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
                cv2.putText(image, "MAGENDA/FUCHSIA", (x, y), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 0, 255))

        return image

    def silver_detection(self, image, hsv, image_det=True):

        if image_det == False:
            silver_mask = cv2.inRange(hsv, self.silver_lower, self.silver_upper)
        else:
            silver_mask = cv2.inRange(image, self.silver_lower, self.silver_upper)

        silver_mask = cv2.dilate(silver_mask, self.kernal)
        output = cv2.bitwise_and(image, image, mask=silver_mask)
        # Tracking the silver Color
        im2, contours, hierarchy = cv2.findContours(silver_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        for pic, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            if (area > 300):
                x, y, w, h = cv2.boundingRect(contour)
                image = cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
                cv2.putText(image, "SILVER", (x, y), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 0, 255))

        return image

    def gray_detection(self, image, hsv, image_det=True):

        if image_det == False:
            gray_mask = cv2.inRange(hsv, self.gray_lower, self.gray_upper)
        else:
            gray_mask = cv2.inRange(image, self.gray_lower, self.gray_upper)

        gray_mask = cv2.dilate(gray_mask, self.kernal)
        output = cv2.bitwise_and(image, image, mask=gray_mask)
        # Tracking the gray Color
        im2, contours, hierarchy = cv2.findContours(gray_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        for pic, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            if (area > 300):
                x, y, w, h = cv2.boundingRect(contour)
                image = cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
                cv2.putText(image, "GRAY", (x, y), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 0, 255))

        return image

    def maroon_detection(self, image, hsv, image_det=True):

        if image_det == False:
            maroon_mask = cv2.inRange(hsv, self.maroon_lower, self.maroon_upper)
        else:
            maroon_mask = cv2.inRange(image, self.maroon_lower, self.maroon_upper)

        maroon_mask = cv2.dilate(maroon_mask, self.kernal)
        output = cv2.bitwise_and(image, image, mask=maroon_mask)
        # Tracking the maroon Color
        im2, contours, hierarchy = cv2.findContours(maroon_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        for pic, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            if (area > 300):
                x, y, w, h = cv2.boundingRect(contour)
                image = cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
                cv2.putText(image, "MAROON", (x, y), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 0, 255))

        return image

    def olive_detection(self, image, hsv, image_det=True):

        if image_det == False:
            olive_mask = cv2.inRange(hsv, self.olive_lower, self.olive_upper)
        else:
            olive_mask = cv2.inRange(image, self.olive_lower, self.olive_upper)

        olive_mask = cv2.dilate(olive_mask, self.kernal)
        output = cv2.bitwise_and(image, image, mask=olive_mask)
        # Tracking the olive Color
        im2, contours, hierarchy = cv2.findContours(olive_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        for pic, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            if (area > 300):
                x, y, w, h = cv2.boundingRect(contour)
                image = cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
                cv2.putText(image, "OLIVE", (x, y), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 0, 255))

        return image

    def green_detection(self, image, hsv, image_det=True):

        if image_det == False:
            green_mask = cv2.inRange(hsv, self.green_lower, self.green_upper)
        else:
            green_mask = cv2.inRange(image, self.green_lower, self.green_upper)

        green_mask = cv2.dilate(green_mask, self.kernal)
        output = cv2.bitwise_and(image, image, mask=green_mask)
        # Tracking the green Color
        im2, contours, hierarchy = cv2.findContours(green_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        for pic, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            if (area > 300):
                x, y, w, h = cv2.boundingRect(contour)
                image = cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
                cv2.putText(image, "GREEN", (x, y), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 0, 255))

        return image

    def purple_detection(self, image, hsv, image_det=True):

        if image_det == False:
            purple_mask = cv2.inRange(hsv, self.purple_lower, self.purple_upper)
        else:
            purple_mask = cv2.inRange(image, self.purple_lower, self.purple_upper)

        purple_mask = cv2.dilate(purple_mask, self.kernal)
        output = cv2.bitwise_and(image, image, mask=purple_mask)
        # Tracking the purpleColor
        im2, contours, hierarchy = cv2.findContours(purple_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        for pic, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            if (area > 300):
                x, y, w, h = cv2.boundingRect(contour)
                image = cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
                cv2.putText(image, "PURPLE", (x, y), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 0, 255))

        return image

    def teal_detection(self, image, hsv, image_det=True):

        if image_det == False:
            teal_mask = cv2.inRange(hsv, self.teal_lower, self.teal_upper)
        else:
            teal_mask = cv2.inRange(image, self.teal_lower, self.teal_upper)

        teal_mask = cv2.dilate(teal_mask, self.kernal)
        output = cv2.bitwise_and(image, image, mask=teal_mask)
        # Tracking the tealColor
        im2, contours, hierarchy = cv2.findContours(teal_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        for pic, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            if (area > 300):
                x, y, w, h = cv2.boundingRect(contour)
                image = cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
                cv2.putText(image, "TEAL", (x, y), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 0, 255))

        return image

    def navy_detection(self, image, hsv, image_det=True):

        if image_det == False:
            navy_mask = cv2.inRange(hsv, self.navy_lower, self.navy_upper)
        else:
            navy_mask = cv2.inRange(image, self.navy_lower, self.navy_upper)

        navy_mask = cv2.dilate(navy_mask, self.kernal)
        output = cv2.bitwise_and(image, image, mask=navy_mask)
        # Tracking the navyColor
        im2, contours, hierarchy = cv2.findContours(navy_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        for pic, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            if (area > 300):
                x, y, w, h = cv2.boundingRect(contour)
                image = cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
                cv2.putText(image, "NAVY", (x, y), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 0, 255))

        return image

    def skyblue_detection(self, image, hsv, image_det=True):

        if image_det == False:
            skyblue_mask = cv2.inRange(hsv, self.skyblue_lower, self.skyblue_upper)
        else:
            skyblue_mask = cv2.inRange(image, self.skyblue_lower, self.skyblue_upper)

        skyblue_mask = cv2.dilate(skyblue_mask, self.kernal)
        output = cv2.bitwise_and(image, image, mask=skyblue_mask)
        # Tracking the skyblueColor
        im2, contours, hierarchy = cv2.findContours(skyblue_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        for pic, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            if (area > 300):
                x, y, w, h = cv2.boundingRect(contour)
                image = cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
                cv2.putText(image, "skyblue", (x, y), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 0, 255))

        return image

    def ImageDetection(self, imagePath, Black=False, White=False, Red=False, Lime=False, Blue=False,
                       Yellow=False, Cyan=False, Magenda=False, Silver=False, Gray=False, Maroon=False,
                       Olive=False, Green=False, Purple=False, Teal=False, Navy=False, Skyblue=False, All=False):
        image = cv2.imread(imagePath)
        img_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        if Black == True or All == True:
            image = self.black_detection(image, img_hsv, image_det=True)
        if White == True or All == True:
            image = self.white_detection(image, img_hsv, image_det=True)
        if Red == True or All == True:
            image = self.red_detection(image, img_hsv, image_det=True)
        if Lime == True or All == True:
            image = self.lime_detection(image, img_hsv, image_det=True)
        if Blue == True or All == True:
            image = self.blue_detection(image, img_hsv, image_det=True)
        if Yellow == True or All == True:
            image = self.yellow_detection(image, img_hsv, image_det=True)
        if Cyan == True or All == True:
            image = self.cyan_detection(image, img_hsv, image_det=True)
        if Magenda == True or All == True:
            image = self.magenda_detection(image, img_hsv, image_det=True)
        if Silver == True or All == True:
            image = self.silver_detection(image, img_hsv, image_det=True)
        if Gray == True or All == True:
            image = self.gray_detection(image, img_hsv, image_det=True)
        if Maroon == True or All == True:
            image = self.maroon_detection(image, img_hsv, image_det=True)
        if Olive == True or All == True:
            image = self.olive_detection(image, img_hsv, image_det=True)
        if Green == True or All == True:
            image = self.green_detection(image, img_hsv, image_det=True)
        if Purple == True or All == True:
            image = self.purple_detection(image, img_hsv, image_det=True)
        if Teal == True or All == True:
            image = self.teal_detection(image, img_hsv, image_det=True)
        if Navy == True or All == True:
            image = self.navy_detection(image, img_hsv, image_det=True)
        if Skyblue == True or All == True:
            image = self.skyblue_detection(image, img_hsv, image_det=True)

        # show the images
        cv2.imshow("images", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def LifeVideoDetection(self, Black=False, White=False, Red=False, Lime=False, Blue=False,
                           Yellow=False, Cyan=False, Magenda=False, Silver=False, Gray=False, Maroon=False,
                           Olive=False, Green=False, Purple=False, Teal=False, Navy=False, Skyblue=False, All=False):
        cap = cv2.VideoCapture(0)
        while (True):
            _, image = cap.read()
            img_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

            if Black == True or All == True:
                image = self.black_detection(image, img_hsv, image_det=False)
            if White == True or All == True:
                image = self.white_detection(image, img_hsv, image_det=False)
            if Red == True or All == True:
                image = self.red_detection(image, img_hsv, image_det=False)
            if Lime == True or All == True:
                image = self.lime_detection(image, img_hsv, image_det=False)
            if Blue == True or All == True:
                image = self.blue_detection(image, img_hsv, image_det=False)
            if Yellow == True or All == True:
                image = self.yellow_detection(image, img_hsv, image_det=False)
            if Cyan == True or All == True:
                image = self.cyan_detection(image, img_hsv, image_det=False)
            if Magenda == True or All == True:
                image = self.magenda_detection(image, img_hsv, image_det=False)
            if Silver == True or All == True:
                image = self.silver_detection(image, img_hsv, image_det=False)
            if Gray == True or All == True:
                image = self.gray_detection(image, img_hsv, image_det=False)
            if Maroon == True or All == True:
                image = self.maroon_detection(image, img_hsv, image_det=False)
            if Olive == True or All == True:
                image = self.olive_detection(image, img_hsv, image_det=False)
            if Green == True or All == True:
                image = self.green_detection(image, img_hsv, image_det=False)
            if Purple == True or All == True:
                image = self.purple_detection(image, img_hsv, image_det=False)
            if Teal == True or All == True:
                image = self.teal_detection(image, img_hsv, image_det=False)
            if Navy == True or All == True:
                image = self.navy_detection(image, img_hsv, image_det=False)
            if Skyblue == True or All == True:
                image = self.skyblue_detection(image, img_hsv, image_det=False)

            cv2.imshow("Color Tracking", image)
            if cv2.waitKey(10) & 0xFF == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                break

