
import time
from PIL import Image
from PIL import ImageDraw
import tflite_runtime.interpreter as tflite
import platform
import collections
import numpy as np
from tensorflow.keras.losses import MSE
import tensorflow as tf

import matplotlib as mpl
import matplotlib.pyplot as plt



EDGETPU_SHARED_LIB = {
  'Linux': 'libedgetpu.so.1',
  'Darwin': 'libedgetpu.1.dylib',
  'Windows': 'edgetpu.dll'
}[platform.system()]

Object = collections.namedtuple('Object', ['id', 'score', 'bbox'])

class BBox(collections.namedtuple('BBox', ['xmin', 'ymin', 'xmax', 'ymax'])):
  """Bounding box.

  Represents a rectangle which sides are either vertical or horizontal, parallel
  to the x or y axis.
  """
  __slots__ = ()

  @property
  def width(self):
    """Returns bounding box width."""
    return self.xmax - self.xmin

  @property
  def height(self):
    """Returns bounding box height."""
    return self.ymax - self.ymin

  @property
  def area(self):
    """Returns bound box area."""
    return self.width * self.height

  @property
  def valid(self):
    """Returns whether bounding box is valid or not.

    Valid bounding box has xmin <= xmax and ymin <= ymax which is equivalent to
    width >= 0 and height >= 0.
    """
    return self.width >= 0 and self.height >= 0

  def scale(self, sx, sy):
    """Returns scaled bounding box."""
    return BBox(xmin=sx * self.xmin,
                ymin=sy * self.ymin,
                xmax=sx * self.xmax,
                ymax=sy * self.ymax)

  def translate(self, dx, dy):
    """Returns translated bounding box."""
    return BBox(xmin=dx + self.xmin,
                ymin=dy + self.ymin,
                xmax=dx + self.xmax,
                ymax=dy + self.ymax)

  def map(self, f):
    """Returns bounding box modified by applying f for each coordinate."""
    return BBox(xmin=f(self.xmin),
                ymin=f(self.ymin),
                xmax=f(self.xmax),
                ymax=f(self.ymax))

  @staticmethod
  def intersect(a, b):
    """Returns the intersection of two bounding boxes (may be invalid)."""
    return BBox(xmin=max(a.xmin, b.xmin),
                ymin=max(a.ymin, b.ymin),
                xmax=min(a.xmax, b.xmax),
                ymax=min(a.ymax, b.ymax))

  @staticmethod
  def union(a, b):
    """Returns the union of two bounding boxes (always valid)."""
    return BBox(xmin=min(a.xmin, b.xmin),
                ymin=min(a.ymin, b.ymin),
                xmax=max(a.xmax, b.xmax),
                ymax=max(a.ymax, b.ymax))

  @staticmethod
  def iou(a, b):
    """Returns intersection-over-union value."""
    intersection = BBox.intersect(a, b)
    if not intersection.valid:
      return 0.0
    area = intersection.area
    return area / (a.area + b.area - area)

def input_size(interpreter):
  """Returns input image size as (width, height) tuple."""
  _, height, width, _ = interpreter.get_input_details()[0]['shape']
  return width, height

def input_tensor(interpreter):
  """Returns input tensor view as numpy array of shape (height, width, 3)."""
  tensor_index = interpreter.get_input_details()[0]['index']
  return interpreter.tensor(tensor_index)()[0]

def set_input(interpreter, size, resize):
  """Copies a resized and properly zero-padded image to the input tensor.

  Args:
    interpreter: Interpreter object.
    size: original image size as (width, height) tuple.
    resize: a function that takes a (width, height) tuple, and returns an RGB
      image resized to those dimensions.
  Returns:
    Actual resize ratio, which should be passed to `get_output` function.
  """
  width, height = input_size(interpreter)
  w, h = size
  scale = min(width / w, height / h)
  w, h = int(w * scale), int(h * scale)
  tensor = input_tensor(interpreter)
  tensor.fill(0)  # padding
  _, _, channel = tensor.shape
  tensor[:h, :w] = np.reshape(resize((w, h)), (h, w, channel))
  return scale, scale

def output_tensor(interpreter, i):
  """Returns output tensor view."""
  tensor = interpreter.tensor(interpreter.get_output_details()[i]['index'])()
  return np.squeeze(tensor)

def get_output(interpreter, score_threshold, image_scale=(1.0, 1.0)):
  """Returns list of detected objects."""
  boxes = output_tensor(interpreter, 0)
  class_ids = output_tensor(interpreter, 1)
  scores = output_tensor(interpreter, 2)
  count = int(output_tensor(interpreter, 3))

  width, height = input_size(interpreter)
  image_scale_x, image_scale_y = image_scale
  sx, sy = width / image_scale_x, height / image_scale_y

  def make(i):
    ymin, xmin, ymax, xmax = boxes[i]
    return Object(
        id=int(class_ids[i]),
        score=float(scores[i]),
        bbox=BBox(xmin=xmin,
                  ymin=ymin,
                  xmax=xmax,
                  ymax=ymax).scale(sx, sy).map(int))

  return [make(i) for i in range(count) if scores[i] >= score_threshold]


def load_labels(path, encoding='utf-8'):
    """Loads labels from file (with or without index numbers).
    Args:
        path: path to label file.
        encoding: label file encoding.
    Returns:
        Dictionary mapping indices to labels.
    """
    with open(path, 'r', encoding=encoding) as f:
        lines = f.readlines()
        if not lines:
            return {}

        if lines[0].split(' ', maxsplit=1)[0].isdigit():
            pairs = [line.split(' ', maxsplit=1) for line in lines]
            return {int(index): label.strip() for index, label in pairs}
        else:
            return {index: line.strip() for index, line in enumerate(lines)}

def make_interpreter(model_file):
    model_file, *device = model_file.split('@')
    return tflite.Interpreter(
        model_path=model_file,
        experimental_delegates=[
            tflite.load_delegate(EDGETPU_SHARED_LIB,
                                {'device': device[0]} if device else {})
        ])

def draw_objects(draw, objs, labels):
    """Draws the bounding box and label for each object."""
    for obj in objs:
        bbox = obj.bbox
        draw.rectangle([(bbox.xmin, bbox.ymin), (bbox.xmax, bbox.ymax)],
                    outline='red')
        draw.text((bbox.xmin + 10, bbox.ymin + 10),
            '%s\n%.2f' % (labels.get(obj.id, obj.id), obj.score),
            fill='red')

def generate_image_adversary(model, image, loss, label, eps=2 / 255.0):
	# cast the image
    image = tf.cast(image, tf.float32)
      
    # record our gradients
    with tf.GradientTape() as tape:
      # explicitly indicate that our image should be tacked for
      # gradient updates

      x = tf.constant(4.0)

      tape.watch(image)

      # tape.watch(x)

      # y = x * x * x

      # use our model to make predictions on the input image and
      # then compute the loss
      # pred = model(image)
      # loss = MSE(label, pred)
      


      # calculate the gradients of loss with respect to the image, then
      # compute the sign of the gradient
      gradient = tape.gradient(loss, image)


      # gradient = tape.gradient(y, x)
      signedGrad = tf.sign(gradient)

      # construct the image adversary
      adversary = (image + (signedGrad * eps)).numpy()

    # return the image adversary to the calling function
    return adversary

class StopSignDetector(object):
    '''
    Requires an EdgeTPU for this part to work

    This part will run a EdgeTPU optimized model to run object detection to detect a stop sign.
    We are just using a pre-trained model (MobileNet V2 SSD) provided by Google.
    '''

    def __init__(self, min_score, show_bounding_box, debug=False):
        self.STOP_SIGN_CLASS_ID = 12
        self.TRAFFIC_LIGHT_CLASS_ID = 9
        self.min_score = min_score
        self.show_bounding_box = show_bounding_box
        self.debug = debug
        self.labels = load_labels("coco_labels.txt") if "coco_labels.txt" else {}
        self.interpreter = make_interpreter("ssd_mobilenet_v2_coco_quant_postprocess_edgetpu.tflite")

        self.pretrained_model = tf.keras.applications.MobileNetV2(include_top=True,
                                                     weights='imagenet')
        self.pretrained_model.trainable = False
        # ImageNet labels
        self.decode_predictions = tf.keras.applications.mobilenet_v2.decode_predictions
                


    def convertImageArrayToPILImage(self, img_arr):
        img = Image.fromarray(img_arr.astype('uint8'), 'RGB')

        return img


    # Helper function to preprocess the image so that it can be inputted in MobileNetV2
    def preprocess(image):
      image = tf.cast(image, tf.float32)
      image = tf.image.resize(image, (224, 224))
      image = tf.keras.applications.mobilenet_v2.preprocess_input(image)
      image = image[None, ...]
      return image

    # Helper function to extract labels from probability vector
    def get_imagenet_label(probs):
      return decode_predictions(probs, top=1)[0][0]

    '''
    Return an object if there is a traffic light in the frame
    '''
    def detect_stop_sign (self, img_arr):
        
        self.interpreter.allocate_tensors()
        image = self.convertImageArrayToPILImage(img_arr)
        scale = set_input( self.interpreter, image.size,
                                lambda size: image.resize(size, Image.ANTIALIAS))

        # Start adversarial attack here
        # function to generate FGSM Fast Gradient Signed Method
        # adversarial example using the Fast Gradient Signed Method 
        # (FGSM) attack as described in Explaining and Harnessing Adversarial 
        # Examples by Goodfellow et al. This was one of the first and most 
        # popular attacks to fool a neural network.
        # Tutorial in https://www.tensorflow.org/tutorials/generative/adversarial_fgsm

        mpl.rcParams['figure.figsize'] = (8, 8)
        mpl.rcParams['axes.grid'] = False
        image = tf.cast(image, tf.float32)
        image = tf.image.resize(image, (224, 224))
        image = tf.keras.applications.mobilenet_v2.preprocess_input(image)
        image = image[None, ...]
        image_probs = self.pretrained_model.predict(image)
        # plt.figure()
        # plt.imshow(image[0] * 0.5 + 0.5)  # To change [-1, 1] to [0,1]
        _, image_class, class_confidence = self.decode_predictions(image_probs, top=1)[0][0]
        print('{} : {:.2f}% Confidence'.format(image_class, class_confidence*100))
        # plt.show()
        traffic_light = 9
        label = tf.one_hot(traffic_light, image_probs.shape[-1])
        label = tf.reshape(label, (1, image_probs.shape[-1]))
        loss_object = tf.keras.losses.CategoricalCrossentropy()
        with tf.GradientTape() as tape:  
          tape.watch(image)
          prediction = self.pretrained_model(image)
          loss = loss_object(label, prediction)
        # Get the gradients of the loss w.r.t to the input image.
        gradient = tape.gradient(loss, image)
        # Get the sign of the gradients to create the perturbation
        signed_grad = tf.sign(gradient)
        epsilons = [0, 0.01, 0.1, 0.15]
        descriptions = [('Epsilon = {:0.3f}'.format(eps) if eps else 'Input') for eps in epsilons]
        for i, eps in enumerate(epsilons):
          adv_x = image + eps*signed_grad
          adv_x = tf.clip_by_value(adv_x, -1, 1)
          _, label, confidence = self.decode_predictions(self.pretrained_model.predict(adv_x), top=1)[0][0]
          # plt.figure()
          # plt.imshow(image[0]*0.5+0.5)
          print('{} \n {} : {:.2f}% Confidence'.format(descriptions[i], label, confidence*100))
          # plt.show()
        # END adversarial attack




        print('----INFERENCE TIME----')
        print('Note: The first inference is slow because it includes',
                'loading the model into Edge TPU memory.')
        start = time.perf_counter()
        self.interpreter.invoke()
        inference_time = time.perf_counter() - start
        objs = get_output(self.interpreter, 0.2, scale)
        print('%.2f ms' % (inference_time * 1000))
        print('-------RESULTS--------')
        if not objs:
            print('No objects detected')

        max_score = 0
        traffic_light_obj = None
        for obj in objs:
            print( self.labels.get(obj.id, obj.id))

            print('  id:    ', obj.id)
            print('  score: ', obj.score)
            print('  bbox:  ', obj.bbox)
            if (obj.id == self.STOP_SIGN_CLASS_ID or obj.id == self.TRAFFIC_LIGHT_CLASS_ID):
                if self.debug:
                    print("stop sign detected, score = {}".format(obj.score))
                if (obj.score > max_score):
                    print(obj.bbox)
                    traffic_light_obj = obj
                    max_score = obj.score

        return traffic_light_obj

    def draw_bounding_box(self, traffic_light_obj, img_arr):
        bbox = traffic_light_obj.bbox
        image = self.convertImageArrayToPILImage(img_arr)
        labels = load_labels("coco_labels.txt") if "coco_labels.txt" else {}
        draw_objects(ImageDraw.Draw(image), traffic_light_obj, labels)
        ImageDraw.Draw(image).rectangle([(bbox.xmin, bbox.ymin), (bbox.xmax, bbox.ymax)],
                    outline='red')

    def run(self, img_arr, throttle, debug=False):
        if img_arr is None:
            return throttle, img_arr

        # Detect traffic light object
        traffic_light_obj = self.detect_stop_sign(img_arr)

        if traffic_light_obj:
            # if self.show_bounding_box:
            #     self.draw_bounding_box(traffic_light_obj, img_arr)
            return 0, img_arr
        else:
            return throttle, img_arr
