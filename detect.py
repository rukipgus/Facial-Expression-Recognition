import time
from absl import app, flags, logging
from absl.flags import FLAGS
import cv2
import numpy as np
import tensorflow as tf
from fer.models import (
    YoloV3, YoloV3Tiny,
    yolo_boxes, YoloLoss,
    yolo_nms,
)
from fer.dataset import transform_images, load_tfrecord_dataset
from fer.utils import draw_outputs


flags.DEFINE_string('classes', './data/nota.names', 'path to classes file')
flags.DEFINE_string('weights', './checkpoints/yolov3.tf',
                    'path to weights file')
flags.DEFINE_boolean('tiny', False, 'yolov3 or yolov3-tiny')
flags.DEFINE_integer('size', 416, 'resize images to')
flags.DEFINE_string('image', './data/family', 'path to input image (without extension of image)')
flags.DEFINE_string('tfrecord', None, 'tfrecord instead of image')
flags.DEFINE_string('output', './output.jpg', 'path to output image')
flags.DEFINE_integer('num_classes', 80, 'number of classes in the model')
flags.DEFINE_integer('channels', 3, 'number of channels in the imagess')
flags.DEFINE_string('model_path', './checkpoints/yolov3_train_tiny.h5', 'path to model')
flags.DEFINE_float('learning_rate', 5e-4, 'learning rate')


def main(_argv):
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    for physical_device in physical_devices:
        tf.config.experimental.set_memory_growth(physical_device, True)

    yolo_tiny_anchors = np.array([(10, 14), (23, 27), (37, 58), (81, 82), (135, 169),  (344, 319)], np.float32) / 416
    yolo_tiny_anchor_masks = np.array([[3, 4, 5], [0, 1, 2]])
    anchor_masks = yolo_tiny_anchor_masks
    anchors = yolo_tiny_anchors
    #model_best = tf.keras.models.load_model('./yolov3-tf2-master/checkpoints/yolov3_train_tiny.h5', compile=False)#custom_object={"loss": YoloLoss})
    model_best = tf.keras.models.load_model(FLAGS.model_path, compile=False)
    logging.info('model loaded')

    optimizer = tf.keras.optimizers.Adam(lr=FLAGS.learning_rate)
    loss = [YoloLoss(anchors[mask], classes=FLAGS.num_classes) for mask in anchor_masks]
    
    model_best.compile(optimizer = optimizer, loss = loss, metrics = ['accuracy'])
    masks=yolo_tiny_anchor_masks
    x = inputs = tf.keras.layers.Input([FLAGS.size, FLAGS.size, FLAGS.channels], name='input_yolo')
    output_0, output_1 = model_best(x)

    class_names = [c.strip() for c in open(FLAGS.classes).readlines()]
    logging.info('classes loaded')

    boxes_0 = tf.keras.layers.Lambda(lambda x: yolo_boxes(x, anchors[masks[0]], FLAGS.num_classes), name='yolo_boxes_0')(output_0)
    boxes_1 = tf.keras.layers.Lambda(lambda x: yolo_boxes(x, anchors[masks[1]], FLAGS.num_classes), name='yolo_boxes_1')(output_1)
    outputs = tf.keras.layers.Lambda(lambda x: yolo_nms(x, anchors, masks, FLAGS.num_classes), name='yolo_nms')((boxes_0[:3], boxes_1[:3]))
    model_best = tf.keras.Model(inputs = [inputs], outputs = outputs, name='yolov3_tiny')
    img_raw = tf.image.decode_image(open(FLAGS.image + '.jpg', 'rb').read(), channels =3)
    img = tf.expand_dims(img_raw, 0)
    img = transform_images(img, FLAGS.size)
    t1 = time.time()
    print(np.shape(model_best(img)))
    boxes, scores, classes, nums = model_best(img)

    t2 = time.time()
    logging.info('time: {}'.format(t2 - t1))
    logging.info('detections:')
    for i in range(nums[0]):
        logging.info('\t{}, {}. {}'.format(class_names[int(classes[0][i])], np.array(scores[0][i]), np.array(boxes[0][i])))

    img = cv2.cvtColor(img_raw.numpy(), cv2.COLOR_RGB2BGR)
    img = draw_outputs(img, (boxes, scores, classes, nums), class_names)
    cv2.imwrite(FLAGS.image +'_out' +'.jpg', img)
    logging.info('output save to {}'.format(FLAGS.image +'_out' + '.jpg'))

"""
    if FLAGS.tiny:
        yolo = YoloV3Tiny(classes=FLAGS.num_classes)
    else:
        yolo = YoloV3(classes=FLAGS.num_classes)

    yolo.load_weights(FLAGS.weights).expect_partial()
    logging.info('weights loaded')

    class_names = [c.strip() for c in open(FLAGS.classes).readlines()]
    logging.info('classes loaded')

    if FLAGS.tfrecord:
        dataset = load_tfrecord_dataset(
            FLAGS.tfrecord, FLAGS.classes, FLAGS.size)
        dataset = dataset.shuffle(512)
        img_raw, _label = next(iter(dataset.take(1)))
    else:
        img_raw = tf.image.decode_image(
            open(FLAGS.image, 'rb').read(), channels=3)

    img = tf.expand_dims(img_raw, 0)
    img = transform_images(img, FLAGS.size)

    t1 = time.time()
    boxes, scores, classes, nums = yolo(img)
    t2 = time.time()
    logging.info('time: {}'.format(t2 - t1))

    logging.info('detections:')
    for i in range(nums[0]):
        logging.info('\t{}, {}, {}'.format(class_names[int(classes[0][i])],
                                           np.array(scores[0][i]),
                                           np.array(boxes[0][i])))

    img = cv2.cvtColor(img_raw.numpy(), cv2.COLOR_RGB2BGR)
    img = draw_outputs(img, (boxes, scores, classes, nums), class_names)
    cv2.imwrite(FLAGS.output, img)
    logging.info('output saved to: {}'.format(FLAGS.output))
"""

if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
