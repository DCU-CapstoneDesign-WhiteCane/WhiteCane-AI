# -*- coding: utf-8 -*-
import logging

import cv2
from keras.models import load_model
import numpy as np
import time


## 이미지 전처리
def preprocessing(frame):
    # 사이즈 조정
    size = (224, 224)
    frame_resized = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)

    # 이미지 정규화
    frame_normalized = (frame_resized.astype(np.float32) / 127.0) - 1

    # 이미지 차원 재조정 - 예측을 위해 reshape 해줍니다.
    frame_reshaped = frame_normalized.reshape((1, 224, 224, 3))

    return frame_reshaped


## 학습된 모델 불러오기
model = load_model("converted_keras\keras_model.h5", compile=False)

# 카메라 캡쳐 객체, 0 = 내장 카메라
capture = cv2.VideoCapture(0)

# 캡쳐 프레임 사이즈 조절
capture.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

sleep_cnt = 0

product_check = "Null" # 객체 인식하는 상품 이름 저장할 변수
while True:
    ret, frame = capture.read()
    if ret == True:
        print("read success!")

    # 이미지 뒤집기
    #frame_fliped = cv2.flip(frame, 1)

    # 이미지 출력
    cv2.imshow("VideoFrame", frame)

    # 1초마다 검사하며, videoframe 창으로 아무 키나 누르게 되면 종료
    if cv2.waitKey(200) > 0:
        break

    # 데이터 전처리
    preprocessed = preprocessing(frame)

    # 예측
    prediction = model.predict(preprocessed)

    # print(prediction) # [[0.00533728 0.99466264]]

    """
    print(prediction)
    print(prediction[0, 0]) #None1
    print(prediction[0, 1]) #None2
    print(prediction[0, 3]) #Coca
    print(prediction[0, 4]) #Sprite
    """

    # 1초 정지
    time.sleep(1)
    # 형태 인식
    if prediction[0, 0] > 0.94:
        print('제품이 잘 보이지 않습니다.')
        product_check = "Null"
        sleep_cnt = 0

    elif prediction[0, 1] > 0.94:
        print('제품이 잘 보이지 않습니다.')
        product_check = "Null"
        sleep_cnt = 0

    # 코카콜라 PET(으)로 판별
    elif prediction[0, 2] > 0.94:
        product_name = "코카콜라 PET"
        print('(' + product_name + ')인식중')
        sleep_cnt += 1
        print(prediction[0, 2])
        if product_name != product_check:
            product_check = product_name
            sleep_cnt = 0

        if sleep_cnt == 3:
            break

    # 코카콜라 CAN(으)로 판별
    elif prediction[0, 3] > 0.94:
        product_name = "코카콜라 캔"
        print('(' + product_name + ')인식중')
        sleep_cnt += 1
        print(prediction[0, 3])
        if product_name != product_check:
            product_check = product_name
            sleep_cnt = 0

        if sleep_cnt == 3:
            break

    # 펩시 PET(으)로 판별
    elif prediction[0, 4] > 0.94:
        product_name = "펩시 PET"
        print('(' + product_name + ')인식중')
        sleep_cnt += 1
        print(prediction[0, 4])
        if product_name != product_check:
            product_check = product_name
            sleep_cnt = 0

        if sleep_cnt == 3:
            break

    # 펩시 CAN(으)로 판별
    elif prediction[0, 5] > 0.94:
        product_name = "펩시 CAN"
        print('(' + product_name + ')인식중')
        sleep_cnt += 1
        print(prediction[0, 5])
        if product_name != product_check:
            product_check = product_name
            sleep_cnt = 0

        if sleep_cnt == 3:
            break

    else:
        print('제품을 인식할 수 없습니다.')
        sleep_cnt = 0

print(product_name + "으로 인식")
# 카메라 객체 반환
capture.release()
# 화면에 나타난 윈도우들을 종료
cv2.destroyAllWindows()