# -*- coding: utf-8 -*-
import logging

import cv2
import tensorflow.python.keras
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


## 학습된 FORM 모델 불러오기
form_path = r'C:\Users\kgd39\PycharmProjects\AI-Test\converted_keras\converted_keras(FORM)\keras_model.h5'
form_model_filename = form_path
form_model = tensorflow.keras.models.load_model(form_model_filename)

## 학습된 CAN 모델 불러오기
can_path = r'C:\Users\kgd39\PycharmProjects\AI-Test\converted_keras\converted_keras(CAN)\keras_model.h5'
can_model_filename = can_path
can_model = tensorflow.keras.models.load_model(can_model_filename)

## 학습된 PET 모델 불러오기
pet_path = r'C:\Users\kgd39\PycharmProjects\AI-Test\converted_keras\converted_keras(PET)\keras_model.h5'
pet_model_filename = pet_path
pet_model = tensorflow.keras.models.load_model(pet_model_filename)

# 카메라 캡쳐 객체, 0 = 내장 카메라
capture = cv2.VideoCapture(0)

# 캡쳐 프레임 사이즈 조절
capture.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

form_cnt = 0  # 3초간 제품을 인식할 변수
sleep_cnt = 0

product_check = "Null" # 객체 인식하는 상품 이름 저장할 변수
form_check = "Null"
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
    prediction_form = form_model.predict(preprocessed)
    prediction_can = can_model.predict(preprocessed)
    prediction_pet = pet_model.predict(preprocessed)
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
    if prediction_form[0, 0] > 0.94:
        print('제품이 잘 보이지 않습니다.')
        product_check = "Null"
        form_cnt = 0
        print(form_cnt)

    elif prediction_form[0, 1] > 0.94:
        print('제품이 잘 보이지 않습니다.')
        product_check = "Null"
        form_cnt = 0
        print(form_cnt)

    # 캔으로 판별
    elif prediction_form[0, 2] > 0.94:
        form_name = "캔음류"
        print('(' + form_name + ')인식중')
        form_cnt += 1
        print(prediction_form[0, 2])
        if form_name != form_check:
            form_check = form_name
            form_cnt = 0

        if form_cnt >= 3:
            print(form_name + "으로 인식")
            form_cnt = 1

            if prediction_can[0, 0] > 0.94:
                print('제품이 잘 보이지 않습니다.')
                product_check = "Null"
                sleep_cnt = 0

            elif prediction_can[0, 1] > 0.94:
                print('제품이 잘 보이지 않습니다.')
                product_check = "Null"
                sleep_cnt = 0

            elif prediction_can[0, 2] > 0.94:
                product_name = "스프라이트"
                print('(' + product_name + ') 캔 인식중')
                sleep_cnt += 1
                print(prediction_can[0, 2])

                if sleep_cnt == 4:
                    # sleep_cnt = 1
                    print('스프라이트 캔')
                    break

            elif prediction_can[0, 3] > 0.94:
                product_name = "코카콜라"
                print('(' + product_name + ') 캔 인식중')
                sleep_cnt += 1
                print(prediction_can[0, 3])

                if sleep_cnt == 4:
                    # sleep_cnt = 1
                    print('코카콜라 캔')
                    break

            elif prediction_can[0, 4] > 0.94:
                product_name = "펩시"
                print('(' + product_name + ') 캔 인식중')
                sleep_cnt += 1
                print(prediction_can[0, 4])


                if sleep_cnt == 4:
                    # sleep_cnt = 1
                    print('펩시 캔')
                    break

            else:
                print('제품을 인식할 수 없습니다.')
                sleep_cnt = 1

    # 페트병으로 판별
    elif prediction_form[0, 3] > 0.94:
        form_name = "페트병음류"
        print('(' + form_name + ')인식중')
        form_cnt += 1
        print(prediction_form[0, 3])
        print(form_cnt)
        if form_name != form_check:
            form_check = form_name
            form_cnt = 0

        if form_cnt >= 3:
            print(form_name + "으로 인식")
            form_cnt = 1

            if prediction_pet[0, 0] > 0.94:
                print('제품이 잘 보이지 않습니다.')
                product_check = "Null"
                sleep_cnt = 0

            elif prediction_pet[0, 1] > 0.94:
                print('제품이 잘 보이지 않습니다.')
                product_check = "Null"
                sleep_cnt = 0

            elif prediction_pet[0, 2] > 0.94:
                product_name = "스프라이트"
                print('(' + product_name + ') 페트병 인식중')
                sleep_cnt += 1
                print(prediction_can[0, 2])

                if product_name != product_check:
                    product_check = product_name
                    sleep_cnt = 0

                if sleep_cnt == 4:
                    # sleep_cnt = 1
                    print('스프라이트 페트병')
                    break

            elif prediction_pet[0, 3] > 0.94:
                product_name = "코카콜라"
                print('(' + product_name + ') 페트병 인식중')
                sleep_cnt += 1
                print(prediction_can[0, 3])

                if product_name != product_check:
                    product_check = product_name
                    sleep_cnt = 0

                if sleep_cnt == 4:
                    # sleep_cnt = 1
                    print('코카콜라 페트병')
                    break

            elif prediction_pet[0, 4] > 0.94:
                product_name = "펩시"
                print('(' + product_name + ') 페트병 인식중')
                sleep_cnt += 1
                print(prediction_can[0, 4])

                if product_name != product_check:
                    product_check = product_name
                    sleep_cnt = 0

                if sleep_cnt == 4:
                    # sleep_cnt = 1
                    print('펩시 페트병')
                    break

            else:
                print('제품을 인식할 수 없습니다.')
                sleep_cnt = 1

    else:
        print('제품을 인식할 수 없습니다.')
        form_cnt = 0


# 카메라 객체 반환
capture.release()
# 화면에 나타난 윈도우들을 종료
cv2.destroyAllWindows()