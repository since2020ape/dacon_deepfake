from tensorflow.keras.utils import Sequence
import tensorflow as tf
# import os
import numpy as np
from imgaug import augmenters as iaa
import random as rnd


import dlib
from imutils import face_utils
import random as rnd
import cv2


args = {'shape_predictor': './resource/shape_predictor_68_face_landmarks.dat', 'picamera': -1}
detector = dlib.get_frontal_face_detector ()
predictor = dlib.shape_predictor (args["shape_predictor"])


def blind_face(img):
    #     img_ = np.array(img, np.uint8)
    seed = rnd.random ()
    probability = 0.5

    if (seed < probability):
        seed = rnd.random ()
        case = int (seed * 7)
        gray = cv2.cvtColor (img, cv2.COLOR_RGB2GRAY)

        rects = detector (gray, 0)
        if len(rects) > 0 :
            shape = predictor (gray, rects[0])
            shape = face_utils.shape_to_np (shape)
            pts_ = get_poly (shape, case)
            cv2.fillConvexPoly (img, pts_, (0, 0, 0))
            del shape, pts_

        del gray

def get_poly(shape_, case_):
    input_shape = (380, 380)

    pts = np.array ([shape_[36], shape_[20], shape_[23], shape_[45]], np.int32)

    if case_ == 0:  # both eye
        pt2 = np.array ([shape_[20][0], shape_[36][1] + shape_[36][1] - shape_[20][1]])
        pt1 = np.array ([shape_[23][0], shape_[45][1] + shape_[45][1] - shape_[23][1]])
        pts = np.array ([shape_[36], shape_[20], shape_[23], shape_[45], pt1, pt2], np.int32)
    if case_ == 1:  # upper
        pts = np.array([shape_[0], shape_[17], shape_[26], shape_[16], shape_[14], shape_[34], shape_[32], shape_[2]],
                       np.int32)
    # elif case_ == 1:  # mouth + chic
    #     pts = np.array ([shape_[1], shape_[15], shape_[10], shape_[8], shape_[5]], np.int32)
    elif case_ == 2:  # left
        pts = np.array ([shape_[0], shape_[17], shape_[21], shape_[27], shape_[8], shape_[5], shape_[3]], np.int32)
    elif case_ == 3:  # right
        pts = np.array ([shape_[16], shape_[26], shape_[22], shape_[27], shape_[8], shape_[11], shape_[14]], np.int32)
    elif case_ == 4:  # nose
        #         pt0 = np.array( [ shape_[27][0], shape_[21][1] ] )
        pt1 = np.array ([shape_[31][0], shape_[28][1]])
        pt2 = np.array ([shape_[35][0], shape_[28][1]])
        pts = np.array ([shape_[27], pt1, shape_[31], shape_[51], shape_[35], pt2], np.int32)
    elif case_ == 5:  # upper
        #         pts = np.array([shape[0], shape_[17], shape_[26], shape_[16], shape_[14], shape_[52], shape_[50], shape_[2]], np.int32)
        pts = np.array ([shape_[0], shape_[17], shape_[26], shape_[16], shape_[14], shape_[34], shape_[32], shape_[2]],
                        np.int32)
    elif case_ == 6:  # forehead
        # pt1 = np.array ([250, 250])
        pt1 = np.array ([0, shape_[17][1]])
        pt2 = np.array ([input_shape[0] - 1, shape_[26][1]])
        pt3 = np.array ([input_shape[0] - 1, 0])
        pt4 = np.array ([0, 0])
        pts = np.array ([pt1, pt2, pt3, pt4], np.int32)

    return pts

def get_label(path):
    spl = tf.strings.split (path, "\\")
    spl = tf.strings.split (spl, "/")
    label = tf.cast (spl[0] == "fake", tf.uint8)
    label = tf.reduce_sum (label)

    #     class_num = np.argmax (onehot)
    class_num = int (label.numpy ())
    # if(class_num==1):
    #     class_num = np.array([0, 1])
    # else:
    #     class_num = np.array ([1, 0])

    onehot = tf.cast(class_num, tf.uint8)
    return onehot


def load_image_label(path):
    # readimage

    gfile = tf.io.read_file (path)
    image = tf.image.decode_png (gfile)

    del gfile
    # norm
    # image = tf.image.resize (image, (380, 380))
    # image = tf.cast (image, tf.uint8)

    # label
    label = get_label (path)

    return image, label


class DataGenerator (Sequence):

    def __init__(self, X, y, batch_size, step_per_epoch, input_shape, shuffle=True, augment = True):
        self.X = X
        self.y = y if y is not None else y
        self.batch_size = batch_size
        self.step_per_epoch = step_per_epoch
        self.input_shape = input_shape
        self.shuffle = shuffle
        self.augment = augment
        self.augment_size = int(self.batch_size*0.3)
        self.init_generator ()
        self.on_epoch_end ()

    def _shuffle_sample(self):
        if (self.shuffle == True):
            np.random.shuffle (self.indexes)

    def init_generator(self):
        self.size_of_sample = len (self.X)
        self.indexes = np.zeros (self.size_of_sample, dtype=np.int)

        for i in range (self.size_of_sample):
            self.indexes[i] = int (i)

        self._shuffle_sample ()

        self.size_of_mini_epoch = self.batch_size * self.step_per_epoch
        self.cnt_mini_epoch = self.size_of_sample // self.size_of_mini_epoch
        self.itor_mini_epoch = 0

    def on_epoch_end(self):

        if (self.itor_mini_epoch < self.cnt_mini_epoch):
            self.mini_indexes = self.indexes[self.itor_mini_epoch * self.size_of_mini_epoch: (self.itor_mini_epoch + 1) * self.size_of_mini_epoch]
            self.itor_mini_epoch = self.itor_mini_epoch + 1
        else:
            self.itor_mini_epoch = 0
            self._shuffle_sample ()
            self.on_epoch_end ()

    def __len__(self):
        return int (self.step_per_epoch)

    #         return int(np.floor(len(self.X) / self.batch_size))

    def __data_generation(self, X_list, y_list):

        # X = np.empty ((self.batch_size, self.input_shape[0], self.input_shape[1], self.input_shape[2]))
        # y = np.empty ((self.batch_size, ), dtype=np.uint8)
        X = np.zeros ((self.batch_size, self.input_shape[0], self.input_shape[1], self.input_shape[2]), dtype=np.float32)
        y = np.zeros ((self.batch_size, ), dtype=int)
        # y = tf.cast(y, tf.uint8)

        if y is not None:
            for i, (path, label) in enumerate(zip(X_list, y_list)):
                img, label_ = load_image_label(path)
                if self.augment and i < self.augment_size:
                    img = augmentor(img)
                # X[i] = tf.cast(img, tf.float32) / 255.0
                X[i] = tf.cast(img, tf.float32)
                y[i] = label_


            return X, y
        else:
            for i, img in enumerate (X_list):
                X[i] = img

            return X

    def __getitem__(self, index):
        indexes = self.mini_indexes[index * self.batch_size: (index + 1) * self.batch_size]
        X_list = [self.X[k] for k in indexes]

        if self.y is not None:
            y_list = [self.y[k] for k in indexes]
            X, y = self.__data_generation (X_list, y_list)
            return X, y
        else:
            y_list = None
            X = self.__data_generation (X_list, y_list)
            return X



def image_compression(img, quality):
    encoded_img = tf.image.encode_jpeg(img, format='rgb', quality=quality)
    img = tf.image.decode_jpeg(encoded_img)
    del encoded_img
    return img



def pattern_augment(img):
    sometimes = lambda aug: iaa.Sometimes(0.5, aug)
    seq = iaa.Sequential(
        [
            # apply the following augmenters to most images
            iaa.Fliplr(0.5),  # horizontally flip 50% of all images
            iaa.Flipud(0.2),  # vertically flip 20% of all images
           iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5)),  # sharpen images
           iaa.Emboss(alpha=(0, 1.0), strength=(0, 2.0)),  # emboss images
           ],
           random_order=True
           )
    return seq.augment_image(img)

noise_pattern_list = ["./noise_pattern.png","./noise_pattern2.png","./noise_pattern3.png"]
def patternBlending(img):
    noise_pattern = cv2.imread(noise_pattern_list[rnd.randint(0,2)])
    noise_pattern = pattern_augment(noise_pattern)
    noise_pattern = cv2.resize(noise_pattern,(img.shape[1],img.shape[0]))
    w,h = 0,0
    rows, cols, channels = noise_pattern.shape
    m_rows= min(rows+h,img.shape[0]-1)
    m_cols = min(cols + w,img.shape[1]-1)
    roi = img[h:m_rows, w:m_cols,:]
    b_th = 0.2
    img[h:m_rows, w:m_cols,:] = cv2.addWeighted(roi, 1-b_th, noise_pattern[:roi.shape[0],:roi.shape[1]] ,b_th, 0)
    return img

def augmentor(image):
    '''
    image = np.array (image)
    blind_face(image)
    return image
    '''
    # global origin_image,compression_image
    if rnd.random() <= 0.5:
        image = image_compression(image, quality=rnd.randint(10, 30))
    image = np.array(image)
    'Apply data augmentation'
    #if rnd.random() <= 0.3:
        #image = patternBlending(image)
    blind_face (image)
    sometimes = lambda aug: iaa.Sometimes(0.5, aug)
    seq = iaa.Sequential(
        [
            # apply the following augmenters to most images
            iaa.Fliplr(0.5),  # horizontally flip 50% of all images
            sometimes(iaa.AdditiveGaussianNoise(scale=(0.05 * 255,0.1 * 255), per_channel=True)),
            sometimes(iaa.GaussianBlur( sigma=(0.0, 1.0) ))
            # sometimes(iaa.UniformColorQuantization(n_colors=32))
        ],
        random_order=True
    )
    return seq.augment_image(image)






