from asyncio.windows_events import NULL
from flask import Flask, render_template, request, Response
import os
import pickle
import numpy as np
from sqlalchemy import true
from tensorflow.keras.models import load_model
from PIL import Image
import cv2
import glob

Result = ''
Result2 = ''

AGE_MODEL = 'weights/deploy_age.prototxt'
AGE_PROTO = 'weights/age_net.caffemodel'
MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
AGE_INTERVALS = ['(00, 02)', '(04, 06)', '(08, 12)', '(15, 20)',
                 '(25, 32)', '(38, 43)', '(48, 53)', '(60, + )']
FACE_PROTO = "weights/deploy.prototxt.txt"
FACE_MODEL = "weights/res10_300x300_ssd_iter_140000_fp16.caffemodel"

frame_width = 1280
frame_height = 720

face_net = cv2.dnn.readNetFromCaffe(FACE_PROTO, FACE_MODEL)
age_net = cv2.dnn.readNetFromCaffe(AGE_MODEL, AGE_PROTO)

app = Flask(__name__, template_folder='view', static_folder="upload")
app.config['DEBUG'] = True

# 캐시 자동 삭제
if app.config['DEBUG']:
    app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
 
@app.route('/', methods = ['GET','POST'])

def opencv_video_result():

    if 'upload' in request.files :
        file_data = request.files['upload']
        file_data.save(f'upload/yolo_video.mp4')

    list_of_files = glob.glob('upload/result/*.jpg')

    if list_of_files != []:
        latest_file = max(list_of_files, key=os.path.getctime)
        path = latest_file

        html = render_template('yolo_video_result.html')
        return html
    else:
        html = render_template('video.html')
        return html

@app.route('/result_page', methods=['GET'])
def result_page():

    global Result

    list_of_files = glob.glob('upload/result/*.jpg')

    if list_of_files != []:
        latest_file = max(list_of_files, key=os.path.getctime)
        path = latest_file.split('/')
        path = path[1]
        print(path)
        print(type(path))
        print(Result)
        html = render_template('result_page.html', path = '/result/predicted_age.jpg',value = Result, value2 = Result2)

        return html

@app.route('/yolo_streamming')
def yolo_streamming():
    m1 = 'multipart/x-mixed-replace; boundary=frame'
    frame = yolo_video_detecting()
    r1 = Response(frame, mimetype = m1)

    return r1

def yolo_video_detecting():

    global Res

    cap = cv2.VideoCapture(0)
    sample_num = 0    
    captured_num = 0   
    sec = 0
    while cap.isOpened():

        status, frame = cap.read()
        sample_num = sample_num + 1
        sec += 1
        if not status:
            break
        print(f'{sec} frame')
        # 60 frame마다 한 장씩
        if sample_num == 60:
            captured_num = captured_num + 1
            cv2.imwrite('./captures/img'+str(captured_num)+'.jpg', frame)
            sample_num = 0

        ret, img = cap.read()

        if img is None :
            break

        a1 = cv2.resize(img,(416,416))
        
        blob = cv2.dnn.blobFromImage(a1, 0.00392, (416, 416), (0, 0, 0))

        ret, buffer = cv2.imencode('.jpg', img)
        frame = buffer.tobytes()
        
        if sec == 120:
            break

        yield(b'--frame\r\n'
                b'Contgent-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    
    gender()
    captioning()
    print(f'result {Result}')
    cap.release()


def captioning():

    global Result

    list_of_files = glob.glob('captures/*.jpg')
    if list_of_files != []:
        latest_file = max(list_of_files, key=os.path.getctime)
        path = latest_file
        img = cv2.imread(path)
        frame = img.copy()
        if frame.shape[1] > frame_width:
            frame = image_resize(frame, width=frame_width)
        faces = get_faces(frame)
        for i, (start_x, start_y, end_x, end_y) in enumerate(faces):
            face_img = frame[start_y: end_y, start_x: end_x]
            blob = cv2.dnn.blobFromImage(
                image=face_img, scalefactor=1.0, size=(227, 227), 
                mean=MODEL_MEAN_VALUES, swapRB=False
            )
            age_net.setInput(blob)
            age_preds = age_net.forward()
            print("="*30, f"Face {i+1} Prediction Probabilities", "="*30)
            for i in range(age_preds[0].shape[0]):
                print(f"{AGE_INTERVALS[i]}: {age_preds[0, i]*100:.2f}%")
            i = age_preds[0].argmax()
            age = AGE_INTERVALS[i]
            age_confidence_score = age_preds[0][i]
            # Draw the box
            label = f"Age:{age} - {age_confidence_score*100:.2f}%"
            print(label)
            Result = label
            yPos = start_y - 15
            while yPos < 15:
                yPos += 15
            cv2.putText(frame, label, (start_x, yPos),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), thickness=2)
            cv2.rectangle(frame, (start_x, start_y), (end_x, end_y), color=(255, 0, 0), thickness=2)
        if frame is not NULL:
            cv2.imwrite("upload/result/predicted_age.jpg", frame)

def gender():

    global Result2

    list_of_files = glob.glob('captures/*.jpg')
    if list_of_files != []:
        latest_file = max(list_of_files, key=os.path.getctime)
        path = latest_file

        gender_model = load_model('model/GenderCNN.h5')

        with open('data/Gender_classes.dat', 'rb') as fp :
            gender_categories = pickle.load(fp)

        image_w = 64
        image_h = 64

        # 이미지  변형작업
        # 이미지 데이터들을 담을 리스트
        X = []

        img2 = Image.open(path)
        img2 = img2.convert('RGB')
        img2 = img2.resize((image_h, image_w))
        data = np.array(img2)
        X.append(data)
 
        X = np.array(X)

        gender_pred = gender_model.predict(X)
        # 각 예측 결과 중 가장 큰 값을 가지고 있는 곳의 인덱스를 가져온다.
        result = np.argmax(gender_pred, axis=1)
        # print('----------------------------------'+ categories[result[0]])
        result2 = gender_categories[result[0]]

        Result2 = result2

        print(Result2)

def get_faces(frame, confidence_threshold=0.5):
    """Returns the box coordinates of all detected faces"""
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104, 177.0, 123.0))
    face_net.setInput(blob)
    output = np.squeeze(face_net.forward())
    faces = []
    for i in range(output.shape[0]):
        confidence = output[i, 2]
        if confidence > confidence_threshold:
            box = output[i, 3:7] * np.array([frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]])
            start_x, start_y, end_x, end_y = box.astype(np.int)
            start_x, start_y, end_x, end_y = start_x - \
                10, start_y - 10, end_x + 10, end_y + 10
            start_x = 0 if start_x < 0 else start_x
            start_y = 0 if start_y < 0 else start_y
            end_x = 0 if end_x < 0 else end_x
            end_y = 0 if end_y < 0 else end_y
            faces.append((start_x, start_y, end_x, end_y))
    return faces

def display_img(title, img):
    cv2.imshow(title, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def get_optimal_font_scale(text, width):
    """Determine the optimal font scale based on the hosting frame width"""
    for scale in reversed(range(0, 60, 1)):
        textSize = cv2.getTextSize(text, fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=scale/10, thickness=1)
        new_width = textSize[0][0]
        if (new_width <= width):
            return scale/10
    return 1

def image_resize(image, width = None, height = None, inter = cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]
    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))
    return cv2.resize(image, dim, interpolation = inter)

app.run(debug=True, host='0.0.0.0')