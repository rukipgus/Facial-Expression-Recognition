from absl import flags
from absl.flags import FLAGS
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import (
    Add,
    Concatenate,
    Conv2D,
    Input,
    Lambda,
    LeakyReLU,
    MaxPool2D,
    UpSampling2D,
    ZeroPadding2D,
    BatchNormalization,
)
from tensorflow.keras.regularizers import l2
from tensorflow.keras.losses import (
    binary_crossentropy,
    sparse_categorical_crossentropy
)
from .utils import broadcast_iou

flags.DEFINE_integer('yolo_max_boxes', 100,
                     'maximum number of boxes per image')
flags.DEFINE_float('yolo_iou_threshold', 0.4, 'iou threshold')
flags.DEFINE_float('yolo_score_threshold', 0.4, 'score threshold')

yolo_anchors = np.array([(10, 13), (16, 30), (33, 23), (30, 61), (62, 45),
                         (59, 119), (116, 90), (156, 198), (373, 326)],
                        np.float32) / 416
yolo_anchor_masks = np.array([[6, 7, 8], [3, 4, 5], [0, 1, 2]])

yolo_tiny_anchors = np.array([(10, 14), (23, 27), (37, 58),
                              (81, 82), (135, 169),  (344, 319)],
                             np.float32) / 416
yolo_tiny_anchor_masks = np.array([[3, 4, 5], [0, 1, 2]])

class Mish(tf.keras.layers.Layer):
    """
    ..math::
        Mish(x) = x*tanh(softplus(x)) = x * tanh(ln(1 + e^{x}))
    
    Example:
        X_input = Input(input_shape)
        X = Mish()(X_input)
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def call(self, inputs):
        return inputs * tf.math.tanh(tf.math.softplus(inputs))
    
    def get_config(self):
        base_config = super().get_config()
        return {**base_config}

    def compute_output_shape(self, input_shape):
        return input_shape

def DarknetConv(x, filters, size, strides=1, activation="LeakyReLU", batch_norm=True):
    if strides == 1:
        padding = 'same'
    else:
        x = ZeroPadding2D(((1, 0), (1, 0)))(x)  # top left half-padding
        padding = 'valid'
    x = Conv2D(filters=filters, kernel_size=size,
               strides=strides, padding=padding,
               use_bias=not batch_norm, kernel_regularizer=l2(0.0005))(x)
    
    if batch_norm:
        x = BatchNormalization()(x)

    if activation == "Mish":
        x = Mish()(x)
    elif activation == "LeakyReLU":
        x = LeakyReLU(alpha=0.1)(x)
    
    elif activation == "Linear":
        x = x
    else:
        print("Check the DarknetConv's batch_norm")
        exit()
    return x


def DarknetResidual(x, filters, activation = "LeakyReLU", batch_norm = True):
    prev = x
    x = DarknetConv(x, filters // 2, 1, activation = activation, batch_norm = batch_norm)
    x = DarknetConv(x, filters, 3, activation = activation, batch_norm = batch_norm)
    x = Add()([prev, x])
    return x


def DarknetBlock(x, filters, blocks, activation = "LeakyReLU", batch_norm = True):
    x = DarknetConv(x, filters, 3, strides=2, activation = activation, batch_norm = batch_norm)
    for _ in range(blocks):
        x = DarknetResidual(x, filters, activation = activation, batch_norm = batch_norm)
    return x

def csdarknet_residual_block(input, filters, blocks, activation = "LeakyReLU", batch_norm = True):
    x = DarknetConv(input, filters, 1, activation = activation, batch_norm = batch_norm)
    for _ in repeat(None, blocks):
        x = csdarknet_residual(x, filters, activation = activation, batch_norm = batch_norm)
    return x

def csdarknet_block(input, filters, blocks, activation = "LeakyReLU", batch_norm = True):
    x = DarknetConv(input, 2*filters, 3, strides = 2, activation = activation, batch_norm = batch_norm)
    previous_1 = x
    x = DarknetConv(x, filters, 1, strides = 1, activation = activation, batch_norm = batch_norm)
    previous_2 = x

    x = csdarknet_residual_block(previous_1, filters, blocks, activation = activation, batch_norm = batch_norm)
    
    x = DarknetConv(x, filters, 1, strides = 1, activation = activation, batch_norm = batch_norm)
    x = Concatenate()([previous_2, x])
    y = x
    x = DarknetConv(x, filters * 2, 1, strides = 1, activation = activation, batch_norm = batch_norm)
    return x, y

def spp(input):
    x = MaxPool2D(pool_size = (5,5))(input)
    previous_1 = x
    x = MaxPool2D(pool_size = (9,9))(input)
    previous_2 = x
    x = MaxPool2D(pool_size = (13, 13))(input)
    previous_3 = x
    x = Concatenate()([previous_3, previous_2, previous_1, input])
    return x


def csdarknet53_spp(name = None, activation = "LeakyReLU", batch_norm = True):
    x = inputs = Input([None, None, 3])
    x = DarknetConv(x, 32, 3, activation = activation, batch_norm = batch_norm)
    x = DarknetConv(x, 64, 3, strides = 2, activation = activation, batch_norm = batch_norm)
    previous_1 = x
    x = DarknetConv(x, 64, 1)
    previous_2 = x
    x = DarknetConv(previous_1, 64, 1, activation = activation, batch_norm = batch_norm)
    previous_3 = x
    x = DarknetConv(x, 32, 1, activation = activation, batch_norm = batch_norm)
    x = DarknetConv(x, 64, 1, activation = activation, batch_norm = batch_norm)
    x = Add()([previous_3, x])
    x = DarknetConv(x, 64, 1, activation = activation, batch_norm = batch_norm)
    x = Concatenate()([previous_2, x])
    x = DarknetConv(x, 64, 1, activation = activation, batch_norm = batch_norm)

    x, _ = csdarknet_block(x, 64, 2, activation = activation, batch_norm = batch_norm)
    x, x_54 = csdarknet_block(x, 128, 8, activation = activation, batch_norm = batch_norm)
    x, x_85 = csdarknet_block(x, 256, 8, activation = activation, batch_norm = batch_norm)
    x, _ = csdarknet_block(x, 512, 4, activation = activation, batch_norm = batch_norm)
    x = DarknetConv(x, 512, 1, activation= "LeakyReLU" )
    x = DarknetConv(x, 1024, 3, activation= "LeakyReLU")
    x = DarknetConv(x, 512, 1, activation= "LeakyReLU")
    x = spp(x)

    return Model(inputs, (x, x_54, x_85), name = name)


def panet_block(filters):
    def panet_bloc(input, subinput):
        x = DarknetConv(input, filters, 1, activation= "LeakyReLU")
        x = DarknetConv(x, filters * 2, 3, activation= "LeakyReLU")
        x = DarknetConv(x, filters, 1, activation= "LeakyReLU")
        output_1 = x
        x = DarknetConv(x, filters // 2, 1, activation= "LeakyReLU")
        x = UpSampling2D(2)(x)
        previous = x
        x = DarknetConv(subinput, filters // 2, 1, activation= "LeakyReLU")
        x = Concatenate()([x, previous])
        x = DarknetConv(x, filters // 2, 1, activation= "LeakyReLU")
        x = DarknetConv(x, filters, 3, activation= "LeakyReLU")
        return x, output_1
    return panet_bloc

def panet(input, subinput_1, subinput_2):
    x, output_1 = panet_block(512)(input, subinput_1)
    x, output_2 = panet_block(256)(x, subinput_2)

    return x, output_1, output_2

def yoloConv_v4(filters, name = None):
    def yolo_conv_v4(x_in):
        if isinstance(x_in, tuple):
            inputs = Input(x_in[0].shape[1:]), Input(x_in[1].shape[1:])
            x, x_sub = inputs

            x = DarknetConv(x, filters, 3, strides = 2, batch_norm= "LeakyReLU")
            x = Concatenate()([x, x_sub])
            x = DarknetConv(x, filters, 3, batch_norm= "LeakyReLU")
            x = DarknetConv(x, filters * 2, 3, batch_norm= "LeakyReLU")
        else:
            x = inputs = Input(x_in.shape[1:])

        x = DarknetConv(x, filters, 1, batch_norm= "LeakyReLU")
        x = DarknetConv(x, filters * 2, 3, batch_norm= "LeakyReLU")
        x = DarknetConv(x, filters, 1, batch_norm= "LeakyReLU")
        #nex = x 
        return Model(inputs, x, name = name)(x_in)

    return yolo_conv_v4

def yolo_v4(size = None, channels = 3, anchors = yolo_anchors, masks = yolo_anchor_masks, classes = 80, training = False):
    x = inputs = Input([size, size, channels])
    x, x_54, x_85 = csdarknet53_spp(name = 'csdarknet53')(x)
    x, output_1, output_2 = panet(x, x_54, x_85)

    x = yoloConv_v4(128, name = 'yolo_conv_v4_0')(x)
    output_v4_0 = Yolov4Output(128, len(masks[2]), classes, name = 'yolo_output_v4_0')(x)
    x = yoloConv_v4(256, name = 'yolo_conv_v4_1')((x, output_2))
    output_v4_1 = Yolov4Output(256, len(masks[1]), classes, name = 'yolo_output_v4_1')(x)
    x = yoloConv_v4(512, name = 'yolo_conv_v4_2')((x, output_1))
    output_v4_2 = Yolov4Output(512, len(masks[0]), classes, name = 'yolo_output_v4_2')(x)

    if training:
        return Model(inputs, (output_v4_0, output_v4_1, output_v4_2), name = 'yolov4')

    boxes_v4_0 = Lambda(lambda x: yolo_boxes(x, anchors[masks[2]], classes), name = 'yolo_boxes_v4_0')(output_v4_0)
    boxes_v4_1 = Lambda(lambda x: yolo_boxes(x, anchors[masks[1]], classes), name = 'yolo_boxes_v4_1')(output_v4_1)
    boxes_v4_2 = Lambda(lambda x: yolo_boxes(x, anchors[masks[0]], classes), name = 'yolo_boxes_v4_2')(output_v4_2)
    outputs_v4 = Lambda(lambda x: yolo_nms(x, anchors, masks, classes), name = 'yolo_nms')((boxes_v4_0[:3], boxes_v4_1[:3], boxes_v4_2[:3]))

    return Model(inputs, outputs_v4, name = 'yolov4')

def Darknet(name=None):
    x = inputs = Input([None, None, 3])
    x = DarknetConv(x, 32, 3)
    x = DarknetBlock(x, 64, 1)
    x = DarknetBlock(x, 128, 2)  # skip connection
    x = x_36 = DarknetBlock(x, 256, 8)  # skip connection
    x = x_61 = DarknetBlock(x, 512, 8)
    x = DarknetBlock(x, 1024, 4)
    return tf.keras.Model(inputs, (x_36, x_61, x), name=name)


def DarknetTiny(name=None):
    x = inputs = Input([None, None, 3])
    x = DarknetConv(x, 16, 3)
    x = MaxPool2D(2, 2, 'same')(x)
    x = DarknetConv(x, 32, 3)
    x = MaxPool2D(2, 2, 'same')(x)
    x = DarknetConv(x, 64, 3)
    x = MaxPool2D(2, 2, 'same')(x)
    x = DarknetConv(x, 128, 3)
    x = MaxPool2D(2, 2, 'same')(x)
    x = x_8 = DarknetConv(x, 256, 3)  # skip connection
    x = MaxPool2D(2, 2, 'same')(x)
    x = DarknetConv(x, 512, 3)
    x = MaxPool2D(2, 1, 'same')(x)
    x = DarknetConv(x, 1024, 3)
    return tf.keras.Model(inputs, (x_8, x), name=name)


def YoloConv(filters, name=None):
    def yolo_conv(x_in):
        if isinstance(x_in, tuple):
            inputs = Input(x_in[0].shape[1:]), Input(x_in[1].shape[1:])
            x, x_skip = inputs

            # concat with skip connection
            x = DarknetConv(x, filters, 1)
            x = UpSampling2D(2)(x)
            x = Concatenate()([x, x_skip])
        else:
            x = inputs = Input(x_in.shape[1:])

        x = DarknetConv(x, filters, 1)
        x = DarknetConv(x, filters * 2, 3)
        x = DarknetConv(x, filters, 1)
        x = DarknetConv(x, filters * 2, 3)
        x = DarknetConv(x, filters, 1)
        return Model(inputs, x, name=name)(x_in)
    return yolo_conv


def YoloConvTiny(filters, name=None):
    def yolo_conv(x_in):
        if isinstance(x_in, tuple):
            inputs = Input(x_in[0].shape[1:]), Input(x_in[1].shape[1:])
            x, x_skip = inputs

            # concat with skip connection
            x = DarknetConv(x, filters, 1)
            x = UpSampling2D(2)(x)
            x = Concatenate()([x, x_skip])
        else:
            x = inputs = Input(x_in.shape[1:])
            x = DarknetConv(x, filters, 1)

        return Model(inputs, x, name=name)(x_in)
    return yolo_conv


def YoloOutput(filters, anchors, classes, name=None):
    def yolo_output(x_in):
        x = inputs = Input(x_in.shape[1:])
        x = DarknetConv(x, filters * 2, 3)
        x = DarknetConv(x, anchors * (classes + 5), 1, batch_norm=False)
        x = Lambda(lambda x: tf.reshape(x, (-1, tf.shape(x)[1], tf.shape(x)[2], anchors, classes + 5)))(x)
        return tf.keras.Model(inputs, x, name=name)(x_in)
    return yolo_output

def YoloOutputv4(filters, anchors, classes, name=None): 
    def yolo_output(x_in):
        x = inputs = Input(x_in.shape[1:])
        x = DarknetConv(x, filters * 2, 3, batch_norm="LeakyReLU")
        x = DarknetConv(x, anchors * (classes + 5), 1, activation="Linear", batch_nom=False)
        x = Lambda(lambda x: tf.reshape(x, (-1, tf.shape(x)[1], tf.shape(x)[2], anchors, classes + 5)))(x)
        return tf.keras.Model(inputs, x, name=name)(x_in)
    return yolo_output
#x = DarknetConv(x, anchors * (classes + 5), 1, batch_norm= "Linear")
        #x = Lambda(lambda x: tf.reshape(x, (-1, tf.shape(x)[1], tf.shape(x)[2], anchors, classes + 5)))(x)


def yolo_boxes(pred, anchors, classes):
    # pred: (batch_size, grid, grid, anchors, (x, y, w, h, obj, ...classes))
    grid_size = tf.shape(pred)[1:3]
    box_xy, box_wh, objectness, class_probs = tf.split(
        pred, (2, 2, 1, classes), axis=-1)

    box_xy = tf.sigmoid(box_xy)
    objectness = tf.sigmoid(objectness)
    class_probs = tf.sigmoid(class_probs)
    pred_box = tf.concat((box_xy, box_wh), axis=-1)  # original xywh for loss

    # !!! grid[x][y] == (y, x)
    grid = tf.meshgrid(tf.range(grid_size[1]), tf.range(grid_size[0]))
    grid = tf.expand_dims(tf.stack(grid, axis=-1), axis=2)  # [gx, gy, 1, 2]

    box_xy = (box_xy + tf.cast(grid, tf.float32)) / \
        tf.cast(grid_size, tf.float32)
    box_wh = tf.exp(box_wh) * anchors

    box_x1y1 = box_xy - box_wh / 2
    box_x2y2 = box_xy + box_wh / 2
    bbox = tf.concat([box_x1y1, box_x2y2], axis=-1)

    return bbox, objectness, class_probs, pred_box


def yolo_nms(outputs, anchors, masks, classes):
    # boxes, conf, type
    b, c, t = [], [], []

    for o in outputs:
        b.append(tf.reshape(o[0], (tf.shape(o[0])[0], -1, tf.shape(o[0])[-1])))
        c.append(tf.reshape(o[1], (tf.shape(o[1])[0], -1, tf.shape(o[1])[-1])))
        t.append(tf.reshape(o[2], (tf.shape(o[2])[0], -1, tf.shape(o[2])[-1])))

    bbox = tf.concat(b, axis=1)
    confidence = tf.concat(c, axis=1)
    class_probs = tf.concat(t, axis=1)

    scores = confidence * class_probs
    boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
        boxes=tf.reshape(bbox, (tf.shape(bbox)[0], -1, 1, 4)),
        scores=tf.reshape(
            scores, (tf.shape(scores)[0], -1, tf.shape(scores)[-1])),
        max_output_size_per_class=FLAGS.yolo_max_boxes,
        max_total_size=FLAGS.yolo_max_boxes,
        iou_threshold=FLAGS.yolo_iou_threshold,
        score_threshold=FLAGS.yolo_score_threshold
    )

    return boxes, scores, classes, valid_detections


def YoloV3(size=None, channels=3, anchors=yolo_anchors,
           masks=yolo_anchor_masks, classes=80, training=False):
    x = inputs = Input([size, size, channels], name='input')

    x_36, x_61, x = Darknet(name='yolo_darknet')(x)

    x = YoloConv(512, name='yolo_conv_0')(x)
    output_0 = YoloOutput(512, len(masks[0]), classes, name='yolo_output_0')(x)

    x = YoloConv(256, name='yolo_conv_1')((x, x_61))
    output_1 = YoloOutput(256, len(masks[1]), classes, name='yolo_output_1')(x)

    x = YoloConv(128, name='yolo_conv_2')((x, x_36))
    output_2 = YoloOutput(128, len(masks[2]), classes, name='yolo_output_2')(x)

    if training:
        return Model(inputs, (output_0, output_1, output_2), name='yolov3')

    boxes_0 = Lambda(lambda x: yolo_boxes(x, anchors[masks[0]], classes),
                     name='yolo_boxes_0')(output_0)
    boxes_1 = Lambda(lambda x: yolo_boxes(x, anchors[masks[1]], classes),
                     name='yolo_boxes_1')(output_1)
    boxes_2 = Lambda(lambda x: yolo_boxes(x, anchors[masks[2]], classes),
                     name='yolo_boxes_2')(output_2)

    outputs = Lambda(lambda x: yolo_nms(x, anchors, masks, classes),
                     name='yolo_nms')((boxes_0[:3], boxes_1[:3], boxes_2[:3]))

    return Model(inputs, outputs, name='yolov3')


def YoloV3Tiny(size=None, channels=3, anchors=yolo_tiny_anchors,
               masks=yolo_tiny_anchor_masks, classes=80, training=False):
    x = inputs = Input([size, size, channels], name='input')

    x_8, x = DarknetTiny(name='yolo_darknet')(x)

    x = YoloConvTiny(256, name='yolo_conv_0')(x)
    output_0 = YoloOutput(256, len(masks[0]), classes, name='yolo_output_0')(x)

    x = YoloConvTiny(128, name='yolo_conv_1')((x, x_8))
    output_1 = YoloOutput(128, len(masks[1]), classes, name='yolo_output_1')(x)

    if training:
        return Model(inputs, (output_0, output_1), name='yolov3')

    boxes_0 = Lambda(lambda x: yolo_boxes(x, anchors[masks[0]], classes),
                     name='yolo_boxes_0')(output_0)
    boxes_1 = Lambda(lambda x: yolo_boxes(x, anchors[masks[1]], classes),
                     name='yolo_boxes_1')(output_1)
    outputs = Lambda(lambda x: yolo_nms(x, anchors, masks, classes),
                     name='yolo_nms')((boxes_0[:3], boxes_1[:3]))
    return Model(inputs, outputs, name='yolov3_tiny')


def YoloLoss(anchors, classes=80, ignore_thresh=0.5):
    def yolo_loss(y_true, y_pred):
        # 1. transform all pred outputs
        # y_pred: (batch_size, grid, grid, anchors, (x, y, w, h, obj, ...cls))
        pred_box, pred_obj, pred_class, pred_xywh = yolo_boxes(
            y_pred, anchors, classes)
        pred_xy = pred_xywh[..., 0:2]
        pred_wh = pred_xywh[..., 2:4]

        # 2. transform all true outputs
        # y_true: (batch_size, grid, grid, anchors, (x1, y1, x2, y2, obj, cls))
        true_box, true_obj, true_class_idx = tf.split(
            y_true, (4, 1, 1), axis=-1)
        true_xy = (true_box[..., 0:2] + true_box[..., 2:4]) / 2
        true_wh = true_box[..., 2:4] - true_box[..., 0:2]

        # give higher weights to small boxes
        box_loss_scale = 2 - true_wh[..., 0] * true_wh[..., 1]

        # 3. inverting the pred box equations
        grid_size = tf.shape(y_true)[1]
        grid = tf.meshgrid(tf.range(grid_size), tf.range(grid_size))
        grid = tf.expand_dims(tf.stack(grid, axis=-1), axis=2)
        true_xy = true_xy * tf.cast(grid_size, tf.float32) - \
            tf.cast(grid, tf.float32)
        true_wh = tf.math.log(true_wh / anchors)
        true_wh = tf.where(tf.math.is_inf(true_wh),
                           tf.zeros_like(true_wh), true_wh)

        # 4. calculate all masks
        obj_mask = tf.squeeze(true_obj, -1)
        # ignore false positive when iou is over threshold
        best_iou = tf.map_fn(
            lambda x: tf.reduce_max(broadcast_iou(x[0], tf.boolean_mask(
                x[1], tf.cast(x[2], tf.bool))), axis=-1),
            (pred_box, true_box, obj_mask),
            tf.float32)
        ignore_mask = tf.cast(best_iou < ignore_thresh, tf.float32)

        # 5. calculate all losses
        xy_loss = obj_mask * box_loss_scale * \
            tf.reduce_sum(tf.square(true_xy - pred_xy), axis=-1)
        wh_loss = obj_mask * box_loss_scale * \
            tf.reduce_sum(tf.square(true_wh - pred_wh), axis=-1)
        obj_loss = binary_crossentropy(true_obj, pred_obj)
        obj_loss = obj_mask * obj_loss + \
            (1 - obj_mask) * ignore_mask * obj_loss
        # TODO: use binary_crossentropy instead
        class_loss = obj_mask * sparse_categorical_crossentropy(
            true_class_idx, pred_class)

        # 6. sum over (batch, gridx, gridy, anchors) => (batch, 1)
        xy_loss = tf.reduce_sum(xy_loss, axis=(1, 2, 3))
        wh_loss = tf.reduce_sum(wh_loss, axis=(1, 2, 3))
        obj_loss = tf.reduce_sum(obj_loss, axis=(1, 2, 3))
        class_loss = tf.reduce_sum(class_loss, axis=(1, 2, 3))

        return xy_loss + wh_loss + obj_loss + class_loss
    return yolo_loss
