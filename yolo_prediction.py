from darkflow.net.build import TFNet
import cv2
import os
import time
import argparse

options = {'model': 'cfg/yolov2-tiny-voc-1c.cfg', 'load':4000, 'threshold':0.2,
            'labels':'labels.txt'}

network = TFNet(options)

def make_prediction(img):
    result = network.return_predict(img)

    result_ = []

    for item in result:
        confidence = item['confidence']
        topleft_x, topleft_y = item['topleft']['x'], item['topleft']['y']
        bottomright_x, bottomright_y = item['bottomright']['x'], item['bottomright']['y']
        label = item['label']

        rect2 = [topleft_x, topleft_y, bottomright_x, bottomright_y]

        result_.append([label] + rect2 + [confidence])
    font = cv2.FONT_HERSHEY_SIMPLEX
    if not result_ == []:
        for bbox in result_:
            label = bbox[0]
            color = [255, 0, 0]

            img = cv2.rectangle(img, (bbox[1], bbox[2]),
                    (bbox[3], bbox[4]), color, 1)

            cv2.putText(img, '{}:{}'.format(label, bbox[-1]),
                    (bbox[1], bbox[2]), font, 0.6, color, 2)

    return img, result_
