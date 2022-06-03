# Import socket module
import socket
import cv2
import numpy as np
import torch
import time
from model import *
import torch.optim as optim
from utils import load_checkpoint
from Controll import *
from CNN import Network
import pygame

global sendBack_angle, sendBack_Speed, current_speed, current_angle
# ------------Initialize variable-------------#
sendBack_angle = 0
sendBack_Speed = 0
current_speed = 0
current_angle = 0
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
error_arr = np.zeros(5)
error_sp = np.zeros(5)
signArray = np.zeros(15)
noneArray = np.zeros(50)
fpsArray = np.zeros(50)
# error_arr = torch.zeros(5)
pre_t = time.time()
dif = time.time()
MAX_SPEED = 50
SPEED_BRAKE = 45.0
SAFE_SPEED = 25.0
Ratio = 0.1
LEARNING_RATE = 1e-4
reset_seconds = 1
fps = 20
carFlag = 0
image_size = (64, 64)
# ----------------Load Checkpoint-------------------#
""" Load the segmentation checkpoint """
print("==> Loading checkpoint")
checkpoint_path = "UNET.pth"
model = build_unet()
model = model.to(device)
model.load_state_dict(torch.load(checkpoint_path, map_location=device))
model.eval()
""" Load the object detection checkpoint """
model_yolov5 = torch.hub.load('ultralytics/yolov5', 'custom', path='best_yolov5.pt')
model_yolov5.to(device)
model_yolov5.eval()
""" Load the object classification checkpoint """
classes = ['noleft', 'noright', 'nostraight', 'straight', 'turnleft', 'turnright', 'unknown']
model_classify = Network()
pre_trained = torch.load('CNN.pth')
model_classify.load_state_dict(pre_trained)
model_classify = model_classify.to(device)
model_classify.eval()
# ----------------------------------------------------------#
# Create a socket object
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
# Define the port on which you want to connect
PORT = 54321
# connect to the server on local computer
# s.connect(('host.docker.internal', PORT))
s.connect(('127.0.0.1', PORT))


def Control(angle, speed):
    global sendBack_angle, sendBack_Speed
    sendBack_angle = angle
    sendBack_Speed = speed
    return sendBack_angle, sendBack_Speed


# -----------------Edit-------------------#
def PID_angle(error, p, i, d):  # 0.43,0,0.02
    global pre_t
    error_arr[1:] = error_arr[0:-1]
    error_arr[0] = error
    P = error * p
    delta_t = time.time() - pre_t
    pre_t = time.time()
    D = (error - error_arr[1]) / delta_t * d
    I = np.sum(error_arr) * delta_t * i
    angle = P + I + D
    if abs(angle) > 5:
        angle = np.sign(angle) * 40
    return int(angle)


def PID_line(lineRow):
    arr = []
    for x, y in enumerate(lineRow):
        if y == 255:
            arr = np.append(arr, x)
    Min = min(arr)
    Max = max(arr)
    center = int((Min + Max) / 2)
    error = int(pred_y.shape[1] / 2) - center
    return error


def Scale_Angle(x):
    return 3 / 13 * x


def predict(image):
    img_rgb = cv2.resize(image, image_size)
    img_rgb = img_rgb / 255
    img_rgb = img_rgb.astype('float32')
    img_rgb = img_rgb.transpose(2, 0, 1)

    img_rgb = torch.from_numpy(img_rgb).unsqueeze(0)

    with torch.no_grad():
        img_rgb = img_rgb.to(device)
        y_pred = model_classify(img_rgb)
        _, predicted = torch.max(y_pred, 1)
        predicted = predicted.data.cpu().numpy()
        # return name of classes
        class_pred = classes[predicted[0]]
    return class_pred


def check_sign(signName, num_minSign):
    for i in range(7):
        if signName == classes[i]:
            new_cls_id = i + 1
    # new_cls_id = box[6] + 1
    signArray[1:] = signArray[0:-1]
    signArray[0] = new_cls_id
    num_cls_id = np.zeros(7)
    for i in range(7):
        num_cls_id[i] = np.count_nonzero(signArray == (i + 1))

    max_num = num_cls_id[0]
    pos_max = 0
    for i in range(7):
        if max_num < num_cls_id[i]:
            max_num = num_cls_id[i]
            pos_max = i

    if max_num >= num_minSign:
        signName = classes[pos_max]
    return signName


def remove_small_contours(image):
    image_binary = np.zeros((image.shape[0], image.shape[1]), np.uint8)
    contours = cv2.findContours(image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[0]
    mask = cv2.drawContours(image_binary, [max(contours, key=cv2.contourArea)], -1, (255, 255, 255), -1)
    image_remove = cv2.bitwise_and(image, image, mask=mask)
    return image_remove


def Detect_YOLOv5(image):
    imageYOLO = cv2.resize(image, (640, 640))
    results = model_yolov5(image)
    sign_YOLOv5 = results.pandas().xyxy[0]
    if (len(sign_YOLOv5) != 0):
        if float(sign_YOLOv5.confidence[0]) >= 0.5:
            x1 = int(results.xyxy[0][0][0])
            y1 = int(results.xyxy[0][0][1])
            x2 = int(results.xyxy[0][0][2])
            y2 = int(results.xyxy[0][0][3])
            img_classifier = imageYOLO[y1:y2, x1:x2]
            sign_cnn = predict(img_classifier)
            if float(sign_YOLOv5.confidence[0]) > 0.7:
                if sign_YOLOv5.name[0] != 'unknown' or sign_cnn != "unknown":
                    sign_checked = check_sign(sign_cnn, 2)
                    return sign_checked


def Detect_UNET(image):
    x = torch.from_numpy(image)
    x = x.to(device)
    x = x.transpose(1, 2).transpose(0, 1)
    x = x / 255.0
    x = x.unsqueeze(0).float()
    with torch.no_grad():
        pred_y = model(x)
        pred_y = torch.sigmoid(pred_y)
        pred_y = pred_y[0]
        pred_y = pred_y.squeeze()
        pred_y = pred_y > 0.5
        pred_y = pred_y.cpu().numpy()
        pred_y = np.array(pred_y, dtype=np.uint8)
        pred_y = pred_y * 255
    return pred_y


if __name__ == "__main__":
    frame = 0
    count = 0
    out_sign = "straight"
    flag_timer = 0
    try:
        while True:
            """
            - Chương trình đưa cho bạn 1 giá trị đầu vào:
                * image: hình ảnh trả về từ xe
                * current_speed: vận tốc hiện tại của xe
                * current_angle: góc bẻ lái hiện tại của xe
            - Bạn phải dựa vào giá trị đầu vào này để tính toán và
            gán lại góc lái và tốc độ xe vào 2 biến:
                * Biến điều khiển: sendBack_angle, sendBack_Speed
                Trong đó:
                    + sendBack_angle (góc điều khiển): [-25, 25]
                        NOTE: ( âm là góc trái, dương là góc phải)
                    + sendBack_Speed (tốc độ điều khiển): [-150, 150]
                        NOTE: (âm là lùi, dương là tiến)
            """

            message_getState = bytes("0", "utf-8")
            s.sendall(message_getState)
            state_date = s.recv(100)

            try:
                current_speed, current_angle = state_date.decode(
                    "utf-8"
                ).split(' ')
            except Exception as er:
                print(er)
                pass

            message = bytes(f"1 {sendBack_angle} {sendBack_Speed}", "utf-8")
            s.sendall(message)
            data = s.recv(100000)

            try:
                image = cv2.imdecode(
                    np.frombuffer(
                        data,
                        np.uint8
                    ), -1
                )

                # print(current_speed, current_angle)
                # print(image.shape)
                # -------------------------------------------Workspace---------------------------------- #

                start = time.time()
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image_crop = image[125:, :]
                imageSeg = cv2.resize(image_crop, (160, 80))

                """Detect lane"""
                pred_y = Detect_UNET(imageSeg)
                pred_y = remove_small_contours(pred_y)
                output_sign = Detect_YOLOv5(image)
                print("Traffic signs: ", output_sign)
                # ------------------- Check none array --------------- #
                CC = Controller(pred_y, output_sign, current_speed)
                # sendBack_Speed, error = CC.controller()
                if output_sign == 'noleft' or out_sign == 'noright' or output_sign == 'nostraight' or output_sign == \
                        'straight' or output_sign == 'turnleft' or output_sign == 'turnright' or output_sign == 'unknown':
                    sendBack_Speed, error = CC.controller()
                    # pygame.init()
                    # pygame.mixer.music.load("floretino.mp3")
                    # pygame.mixer.music.play(0)
                    #
                    # clock = pygame.time.Clock()
                    # clock.tick(10)
                    # while pygame.mixer.music.get_busy():
                    #     pygame.event.poll()
                    #     clock.tick(10)
                else:
                    # Min, Max = CC.line()
                    arr_normal = []
                    height = 18  # 18  Try 20
                    lineRow = pred_y[height, :]
                    for x, y in enumerate(lineRow):
                        if y == 255:
                            arr_normal.append(x)
                    Min = min(arr_normal)
                    Max = max(arr_normal)
                    center = int((Min + Max) / 2)
                    error = int(pred_y.shape[1] / 2) - center
                    # pygame.init()
                    # pygame.mixer.music.load("floretino.mp3")
                    # pygame.mixer.music.play(0)
                    #
                    # clock = pygame.time.Clock()
                    # clock.tick(10)
                    # while pygame.mixer.music.get_busy():
                    #     pygame.event.poll()
                    #     clock.tick(10)
                """PID controller"""
                sendBack_angle = -PID_angle(error, 0.35, 0, 0.01)
                sendBack_angle = Scale_Angle(sendBack_angle)
                # ------------------TEST----------------#
                if sendBack_angle <= -9 or sendBack_angle >= 9:
                    sendBack_Speed = 10
                elif sendBack_angle >= -1 and sendBack_angle <= 1:
                    sendBack_Speed = 30
                if sendBack_Speed > MAX_SPEED:
                    sendBack_Speed = MAX_SPEED - 10
                elif sendBack_Speed <= 0:
                    sendBack_Speed = 35
                elif sendBack_Speed <= 14 and sendBack_Speed >= 10:
                    sendBack_Speed = 25
                # sendBack_Speed = PID_speed(error, 0.55, 0.2, 0.01)
                result_angle, result_speed = Control(sendBack_angle, sendBack_Speed)
                print("===============================VÀ ĐÂY LÀ PHOLOTINO !====================================")
                print("|       GÓC LỤM HOA CỦA PHOLOTINO: {}".format(result_angle) + \
                      " TỐC ĐỘ MÚA CỦA PHOLOTINO: {}    |".format(result_speed))
                print("-----------------------------------QUẢ GHẾ GÔM!-----------------------------------------")
                end = time.time()
                fps = 1 / (end - start)
                print("FPS: {}".format(fps))
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            except Exception as er:
                print(er)
                pass
    finally:
        print('----------------------------------------CAY VCL!-------------------------------------------------------')
        s.close()
