import os
import time
from absl import app, flags, logging
from absl.flags import FLAGS
import cv2
import numpy as np
import tensorflow as tf
from .yolov3_tf2.models import (
    YoloV3
)
from .yolov3_tf2.dataset import transform_images, load_tfrecord_dataset
from .yolov3_tf2.utils import draw_outputs


def main(_):

    yolo = YoloV3(classes = 80)

    yolo.load_weights(FLAGS.weights).expect_partial()
    logging.info('weights loaded')

    img_raw = tf.image.decode_image(
            open(FLAGS.image, 'rb').read(), channels=3)

    img = tf.expand_dims(img_raw, 0)
    img = transform_images(img, 416)

    t1 = time.time()
    boxes, scores, classes, nums = yolo(img)
    t2 = time.time()
    logging.info('time: {}'.format(t2 - t1))

    logging.info('detections:')
    for i in range(nums[0]):
        logging.info('\t{}, {}'.format(np.array(scores[0][i]),
                                        np.array(boxes[0][i])))

    img = cv2.cvtColor(img_raw.numpy(), cv2.COLOR_RGB2BGR)
    img, cropped_images = draw_outputs(img, (boxes, scores, classes, nums), ['face'])
    cv2.imwrite(os.path.join(FLAGS.output, 'output.jpg'), img)
    for i,cropped_image in enumerate(cropped_images):
        cv2.imwrite(os.path.join(FLAGS.cropped_output, 'cropped_output.jpg'), cropped_image)
    return {'msg':'done'}

def predict(image):

    yolo = YoloV3(classes = 80)

    yolo.load_weights('./face_detection/checkpoints/yolov3_train_16_00.tf').expect_partial()
    logging.info('weights loaded')

    img = tf.expand_dims(image, 0)
    img = transform_images(img, 416)

    t1 = time.time()
    boxes, scores, classes, nums = yolo(img)
    t2 = time.time()
    logging.info('time: {}'.format(t2 - t1))

    logging.info('detections:')
    for i in range(nums[0]):
        logging.info('\t{}, {}'.format(np.array(scores[0][i]),
                                        np.array(boxes[0][i])))

    img = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    img, cropped_images = draw_outputs(img, (boxes, scores, classes, nums), ['face'])
    return cropped_images

if __name__=='__main__':
    FLAGS=flags.FLAGS
    flags.DEFINE_string('weights','./checkpoints/yolov3_train_16_00.tf','path to weights')
    flags.DEFINE_string('image','','path to the image')
    flags.DEFINE_string('output','./output/plain/','path to all image output')
    flags.DEFINE_string('cropped_output','./output/cropped/','path to the cropped output')
    app.run(main)
