import os
import tensorflow as tf
import yolov3_tf2.dataset as dataset
from yolov3_tf2.models import (
    YoloV3, YoloV3Tiny, YoloLoss,
    yolo_anchors, yolo_anchor_masks,
    yolo_tiny_anchors, yolo_tiny_anchor_masks
)
from absl import flags, logging, app
from yolov3_tf2.utils import freeze_all
import numpy as np

def main(_argv):
    model = YoloV3(FLAGS.size, training=True, classes=80)
    anchors = yolo_anchors
    anchor_masks = yolo_anchor_masks
    train_dataset = dataset.load_tfrecord_dataset("./data/tfrecord/train.tfrecord", './data/faces.names')
    train_dataset = train_dataset.shuffle(buffer_size=512)
    train_dataset = train_dataset.batch(FLAGS.batch_size)
    train_dataset = train_dataset.map(lambda x, y: (
        dataset.transform_images(x, 416),
        dataset.transform_targets(y, anchors, anchor_masks, 416)))
    train_dataset = train_dataset.prefetch(
        buffer_size=tf.data.experimental.AUTOTUNE)
    val_dataset = dataset.load_tfrecord_dataset("./data/tfrecord/val.tfrecord", './data/faces.names')
    val_dataset = val_dataset.batch(FLAGS.batch_size)
    val_dataset = val_dataset.map(lambda x, y: (
        dataset.transform_images(x, 416),
        dataset.transform_targets(y, anchors, anchor_masks, 416)))

    # test_dataset = dataset.load(FLAGS.batch_size, split='test')
    # test_dataset = train_dataset.shuffle(buffer_size=512)
    # test_dataset = train_dataset.batch(FLAGS.batch_size)
    # test_dataset = test_dataset.map(lambda x, y: (
    #     dataset.transform_images(x, FLAGS.size),
    #     dataset.transform_targets(y, anchors, anchor_masks, FLAGS.size)))

    model.load_weights(FLAGS.weights)
    darknet = model.get_layer('yolo_darknet')
    freeze_all(darknet)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     

    optimizer = tf.keras.optimizers.Adam(lr=FLAGS.learning_rate)

    avg_loss = tf.keras.metrics.Mean('loss', dtype=tf.float32)
    avg_val_loss = tf.keras.metrics.Mean('val_loss', dtype=tf.float32)
    loss = [YoloLoss(anchors[mask], classes=80)
            for mask in anchor_masks]

    for epoch in range(1, FLAGS.epochs + 1):
        for batch, (images, labels) in train_dataset.enumerate():
            with tf.GradientTape() as tape:
                outputs = model(images, training=True)
                regularization_loss = tf.reduce_sum(model.losses)
                pred_loss = []
                for output, label, loss_fn in zip(outputs, labels, loss):
                    pred_loss.append(loss_fn(label, output))
                total_loss = tf.reduce_sum(pred_loss) + regularization_loss

            grads = tape.gradient(total_loss, model.trainable_variables)
            optimizer.apply_gradients(
                zip(grads, model.trainable_variables))

            logging.info("{}_train_{}, {}, {}".format(
                epoch, batch, total_loss.numpy(),
                list(map(lambda x: np.sum(x.numpy()), pred_loss))))
            avg_loss.update_state(total_loss)

        for batch, (images, labels) in enumerate(val_dataset):
            outputs = model(images)
            regularization_loss = tf.reduce_sum(model.losses)
            pred_loss = []
            for output, label, loss_fn in zip(outputs, labels, loss):
                pred_loss.append(loss_fn(label, output))
            total_loss = tf.reduce_sum(pred_loss) + regularization_loss

            logging.info("{}_val_{}, {}, {}".format(
                epoch, batch, total_loss.numpy(),
                list(map(lambda x: np.sum(x.numpy()), pred_loss))))
            avg_val_loss.update_state(total_loss)

        logging.info("{}, train: {}, val: {}".format(
            epoch,
            avg_loss.result().numpy(),
            avg_val_loss.result().numpy()))

        avg_loss.reset_states()
        avg_val_loss.reset_states()
        model.save_weights(
            'checkpoints/yolov3_train_{}.tf'.format(epoch))

if __name__ == "__main__":
    FLAGS = flags.FLAGS
    flags.DEFINE_integer('size', 416, 'image size')
    flags.DEFINE_integer('epochs', 8, 'number of epochs')
    flags.DEFINE_integer('batch_size', 3, 'batch size')
    flags.DEFINE_float('learning_rate', 1e-3, 'learning rate')
    flags.DEFINE_string('weights', './checkpoints/yolov3.tf',
                    'path to weights file')
    flags.DEFINE_string('classes', './data/faces.names', 'path to classes file')

    app.run(main)