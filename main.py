# -*- coding: utf-8 -*-
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


## 학습된 모델 불러오기
path = r'C:\Users\kgd39\PycharmProjects\AI-Test\converted_keras\keras_model.h5'
model_filename = path
model = tensorflow.keras.models.load_model(model_filename)

# 카메라 캡쳐 객체, 0=내장 카메라
capture = cv2.VideoCapture(0)

# 캡쳐 프레임 사이즈 조절
capture.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

sleep_cnt = 1  # 3초간 제품을 인식할 변수
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
    print(prediction)
    print(prediction[0, 0])
    print(prediction[0, 1])
    print(prediction[0, 2])
    print(prediction[0, 3])
    print(prediction[0, 4])

    time.sleep(1)

    if prediction[0, 0] > 0.94:
        print('(코카콜라)인식중')
        sleep_cnt += 1
        print(prediction[0, 0])

        # 졸린 상태가 30초간 지속되면 소리 & 카카오톡 보내기
        if sleep_cnt / 3 == 0:
            sleep_cnt = 1
            print('코카콜라')
            break  ## 1번만 알람이 오면 프로그램을 정지 시킴 (반복을 원한다면, 주석으로 막기!)
    elif prediction[0, 1] > 0.94:
        print('제품이 잘 보이지 않습니다.')
        sleep_cnt = 1

    elif prediction[0, 2] > 0.94:
        print('(팹시)인식중')
        sleep_cnt += 1
        print(prediction[0, 2])

        # 졸린 상태가 30초간 지속되면 소리 & 카카오톡 보내기
        if sleep_cnt / 3 == 0:
            sleep_cnt = 1
            print('펩시')
            break  ## 1번만 알람이 오면 프로그램을 정지 시킴 (반복을 원한다면, 주석으로 막기!)
    elif prediction[0, 3] > 0.94:
        print('(스프라이트)인식중')
        sleep_cnt += 1

        # 졸린 상태가 30초간 지속되면 소리 & 카카오톡 보내기
        if sleep_cnt / 3 == 0:
            sleep_cnt = 1
            print('스프라이트')
            break  ## 1번만 알람이 오면 프로그램을 정지 시킴 (반복을 원한다면, 주석으로 막기!)
    elif prediction[0, 4] > 0.94:
        print('(베리베리스트로베리)인식중')
        sleep_cnt += 1

        # 졸린 상태가 30초간 지속되면 소리 & 카카오톡 보내기
        if sleep_cnt / 3 == 0:
            sleep_cnt = 1
            print('베리베리스트로베리')
            break  ## 1번만 알람이 오면 프로그램을 정지 시킴 (반복을 원한다면, 주석으로 막기!)
    else:
        print('제품을 인식할 수 없습니다.')
        sleep_cnt = 1

# 카메라 객체 반환
capture.release()
# 화면에 나타난 윈도우들을 종료
cv2.destroyAllWindows()