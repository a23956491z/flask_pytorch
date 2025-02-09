# -*- coding: utf-8 -*-
# @Time    : 2018/4/24 10:04
# @Author  : zhoujun
import flask
from flask_uploads import UploadSet, IMAGES, configure_uploads, ALL
from flask import request, Flask, redirect, url_for, render_template
import time
import cv2
import numpy as np
import config
from werkzeug.utils import secure_filename

import os
from model import Pytorch_model

app = Flask(__name__)
app.config.from_object(config)
photos = UploadSet('PHOTO')
configure_uploads(app, photos)

def open_resize(img_path):

    store_size = 512
    img_file = os.path.join('uploads', img_path)

    img = cv2.imread(img_file)

    biggest = np.max(img.shape)
    scale_ratio = biggest / store_size
    print()
    resized = cv2.resize(img, (int(img.shape[1]//scale_ratio), int(img.shape[0]//scale_ratio)))

    cv2.imwrite(img_file, resized)
    return img_path
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/demo', methods=['POST', 'GET'])
def upload():

    if request.method == 'POST':
        img = request.files['img'].filename
        topk = request.form['topk']
        img = secure_filename(img)

        if 'JPEG' in img:
            img = img.replace("JPEG", 'jpg')
        print(img)
        new_name = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime()) + '_' + img

        filename = photos.save(request.files['img'], name=new_name)

        open_resize(filename)

        data = predict_img(photos.path(filename), is_numpy=False,topk=int(topk))
        img_path = photos.url(filename)

        return flask.jsonify({"result":data,"img_path":img_path})
    else:
        img_path = None
        result = []
    return render_template('upload.html', img_path=img_path, result=result)


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    data = {'state': False}
    if request.method == 'POST':
        img = request.files['image'].read()
        try:
            topk = request.form['topk']
        except:
            topk = 1
        img = np.fromstring(img, np.uint8)
        img = cv2.imdecode(img, flags=1)
        data = predict_img(img, is_numpy=True, topk=topk)
    return flask.jsonify(data)


def predict_img(img, is_numpy=False, topk=1):
    data = dict()
    start = time.time()
    result = model.predict(img, is_numpy=is_numpy, topk=int(topk))
    cost_time = time.time() - start
    data['predictions'] = list()
    for label, prob in result:
        m_predict = {'label': label, 'probability': ("%.4f" % prob)}
        data['predictions'].append(m_predict)
    data['state'] = True
    data['time'] = cost_time
    return data


if __name__ == '__main__':
    print("Loading PyTorch model and Flask starting server ...")
    print("Please wait until server has fully started")

    model_path = 'feed.pt'
    gpu_id = 0
    model = Pytorch_model(model_path=model_path, img_shape=[
        224, 224], img_channel=3, gpu_id=gpu_id, classes_txt='labels')
    app.run(host="0.0.0.0", port="3000")
