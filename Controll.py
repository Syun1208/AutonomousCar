import numpy as np
import time
import cv2

# ------------Initialize Variable------------#
sign_check = np.zeros(2)
pre_time = time.time()
tim_str = time.time()
error_arr = np.zeros(5)
pre_t = time.time()


def make_coordinates(image_option, line_parameters):
    slope, intercept = line_parameters
    y1 = image_option.shape[0]
    y2 = int(y1 * (3 / 5))
    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)
    return np.array([x1, y2, x2, y2])


# def calculateDistance(option):
#     out_distance = 65
#     if len(option) != 0:
#         avg = np.sum(option) / len(option)
#         d = 80 - avg
#         if d > 70:
#             return 200
#         if d - out_distance < 3:
#             out_distance = d
#             return d
#         else:
#             return 200
#     else:
#         out_distance = 65
#         return 200


# def displayLines(image_option, line_option):
#     line_image = np.zeros_like(image_option)
#     if line_option is not None:
#         for line in line_option:
#             x1, y1, x2, y2 = line_option.reshape(4)
#             cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 10)


class findingLine:
    def __init__(self, images=None, blur_times=0):
        self.images = images
        self.blur_times = blur_times

    def canny(self):
        # convert image to gray scale
        # gray = cv2.cvtColor(self.images, cv2.COLOR_BGR2GRAY)
        # reduce noise using gaussian blur
        blur = cv2.GaussianBlur(self.images, (5, 5), 0)
        for i in range(self.blur_times):
            blur = cv2.GaussianBlur(blur, (5, 5), 0)
        # canny
        canny = cv2.Canny(blur, 150, 200)
        return canny

    def ROI(self):
        height = self.canny().shape[0]
        triangle = np.array([
            [(200, height), (1100, height), (550, 250)]
        ])
        mask = np.zeros_like(self.canny())
        cv2.findContours(mask, triangle, 255)
        # Bitwise_and
        masked_image = cv2.bitwise_and(self.canny(), mask)
        return masked_image

    def findDistance(self):
        lane_image = np.copy(self.images)
        canny = self.canny()
        # cropped_image = self.ROI()
        lines = cv2.HoughLinesP(canny, 2, np.pi / 4, 100, np.array([]), minLineLength=10, maxLineGap=10)
        return lines
        # left_lines, right_lines = average_slope_intercept(lane_image, lines)
        # output_distances = calculateDistance(left_lines)
        # line_image = displayLines(lane_image, averaged_lines)
        # combo_image = cv2.addWeighted(lane_image, 0.8, 1, 1)
        # return output_distances


class Controller(findingLine):
    def __init__(self, images, sign, current_speed):
        super().__init__(images)
        self.image = images
        self.sign = sign
        self.current_speed = current_speed
        self.width = np.zeros(10)
        self.min1 = 30
        self.min2 = 60
        self.max1 = 100
        self.max2 = 120
        self.sendBack_speed = 0
        self.MAX_SPEED = 50
        self.center = 0
        self.error = 0

    def straight(self):
        width_road = 0
        Min, Max = self.line()
        if 100 <= Max <= 150 and 2 <= Min <= 70:
            self.width[1:] = self.width[0:-1]
            if Max - Min > 60:
                self.width[0] = Max - Min
            width_road = np.average(self.width)
        self.center = int((Min + Max) / 2)
        if Max < self.max1 and self.min1 <= Min <= self.min2 and not Min == Max == 91 or Max >= self.max2 \
                and self.min1 <= Min <= self.min2 and not Min == Max == 91:
            self.center = Min + int(width_road / 2)
        elif Min >= self.min2 and self.max1 <= Max <= self.max2 and not Min == Max == 91 or Min < self.min1 \
                and self.max1 <= Max <= self.max2 and not Min == Max == 91:
            self.center = Max - int(width_road / 2)
        if float(self.current_speed) <= 7:
            self.sendBack_speed = self.MAX_SPEED
        elif float(self.current_speed) >= self.MAX_SPEED:
            self.sendBack_speed = 10
        return self.sendBack_speed, self.center

    def turnLeft(self):
        distances = self.average_slope_intercept()
        if distances <= 10:
            self.center = 5
            self.sendBack_speed = 10
        else:
            self.sendBack_speed, self.center = self.straight()
        return self.sendBack_speed, self.center

    def turnRight(self):
        Min, Max = self.line()
        distances_right = self.average_slope_intercept()
        if self.image.shape[1] - (Max - Min) - distances_right <= 10:
            self.center = 150
            self.sendBack_speed = 10
        else:
            self.sendBack_speed, self.center = self.straight()
        return self.sendBack_speed, self.center

    def line(self):
        arr = []
        height = 18
        lineRow = self.image[height, :]
        for x, y in enumerate(lineRow):
            if y == 255:
                arr = np.append(arr, x)
        Min = min(arr)
        Max = max(arr)
        return Min, Max

    def average_slope_intercept(self, show=True):
        output_intercept = 0
        left_fit = []
        right_fit = []
        h = 0
        line_option = super().findDistance()
        if line_option is not None:
            for line in line_option:
                x1, y1, x2, y2 = line_option.reshape(4)
                parameters = np.poly((x1, x2), (y1, y2), 1)
                slope = parameters[0]
                intercept = parameters[1]
                if slope == 0 and intercept > 5:
                    avg_intercept = sum(intercept) / len(intercept)
                    return avg_intercept

    def controller(self):
        Min, Max = self.line()
        if self.sign == "unknown":
            self.sendBack_speed = 0
        elif self.sign == "straight":
            self.sendBack_speed, self.center = self.straight()
        elif self.sign == "nostraight":
            self.sendBack_speed = 10
            if Min <= 5:
                self.sendBack_speed, self.center = self.turnLeft()
            elif Max >= 150:
                self.sendBack_speed, self.center = self.turnRight()
        elif self.sign == "turnright":
            self.sendBack_speed, self.center = self.turnRight()
        elif self.sign == "turnleft":
            self.sendBack_speed, self.center = self.turnLeft()
        elif self.sign == "noright":
            self.sendBack_speed = 10
            if Min <= 5:
                self.sendBack_speed, self.center = self.turnLeft()
            elif Max >= 139:
                self.sendBack_speed, self.center = self.straight()
        elif self.sign == "noleft":
            self.sendBack_speed = 10
            if Min <= 5:
                self.sendBack_speed, self.center = self.straight()
            elif Max >= 139:
                self.sendBack_speed, self.center = self.turnRight()
        self.error = int(self.image.shape[1] / 2) - self.center
        return self.sendBack_speed, self.error
