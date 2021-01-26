import tensorflow as tf
import cv2
import numpy as np

from glob import glob
import os
import sys
import argparse
import shutil

def mtcnn_fun(img, min_size, factor, thresholds):
    with open('./resource/mtcnn.pb', 'rb') as f:
        graph_def = tf.compat.v1.GraphDef.FromString(f.read())

    with tf.device('/cpu:0'):
        prob, landmarks, box = tf.compat.v1.import_graph_def(graph_def,
            input_map={
                'input:0': img,
                'min_size:0': min_size,
                'thresholds:0': thresholds,
                'factor:0': factor
            },
            return_elements=[
                'prob:0',
                'landmarks:0',
                'box:0']
            , name='')
    print(box, prob, landmarks)
    return box, prob, landmarks

# wrap graph function as a callable function
mtcnn_fun = tf.compat.v1.wrap_function(mtcnn_fun, [
    tf.TensorSpec(shape=[None, None, 3], dtype=tf.float32),
    tf.TensorSpec(shape=[], dtype=tf.float32),
    tf.TensorSpec(shape=[], dtype=tf.float32),
    tf.TensorSpec(shape=[3], dtype=tf.float32)
])

# 원본 데이터 폴더와 같은 데이터 폴더를 생성
def cp_dir(origin_dir,copy_path):
    folder_name_1 = origin_dir.split('\\')[-1]
    mk_1 = copy_path +"/"+folder_name_1 + "/"
    if not os.path.exists(mk_1):
        os.makedirs(mk_1)
    return mk_1


# 이미지에서 face box 좌표에 마진을 주고 자른 얼굴이미지를 리턴
def margine_face(img,box):
    w = box[3] - box[1] # 27
    h = box[2] - box[0] # 20
    w_m = int(h * 0.135)
    h_m = int(w * 0.10)

    x_min = max(box[1] - w_m,0)
    y_min = max(box[0] - h_m,0)
    x_max = min(box[3]+w_m,img.shape[1])
    y_max = min(box[2]+h_m,img.shape[0])

    face_img = img[y_min:y_max,x_min:x_max].copy()

    return face_img

# 여러 박스 좌표 중 가장 큰 박스 좌표의 인덱스 리턴
def get_max_size_box(bbox):
    max_idx = 0
    max_size = 0
    i = 0
    for box in bbox:
        w = box[3] - box[1]
        h = box[2] - box[0]
        box_size = w * h
        if max_size < box_size:
            max_size = box_size
            max_idx = i
        i += 1
    return max_idx

suc_cnt = 0
err_path=[]
def write_face(image_dirs,write_path):
    global suc_cnt, err_path
    for file_path in image_dirs:
        img_file_name = file_path.split('\\')[-1]
        mk_file = write_path + img_file_name

        # read data
        img = cv2.imread(file_path)

        # detect Face
        bbox, scores, landmarks = mtcnn_fun(img, 160, 0.3, [0.98, 0.98, 0.98])  # bbox : face box position
        bbox, scores, landmarks = bbox.numpy(), scores.numpy(), landmarks.numpy()

        total_box = len(bbox)

        if (total_box < 1):
            img = cv2.rotate(img, cv2.ROTATE_180)
            bbox, scores, landmarks = mtcnn_fun(img, 160, 0.3, [0.98, 0.98, 0.98])
            bbox, scores, landmarks = bbox.numpy(), scores.numpy(), landmarks.numpy()
            total_box = len(bbox)

        for box, pts in zip(bbox, landmarks):
            box = box.astype('int32')
            pts = pts.astype('int32')
            f_img = margine_face(img,box)
            try:
                f_img = cv2.resize(f_img, (380, 380)) # mean face shape : (155,205,3)
            except:
                continue

            cv2.imwrite(mk_file, f_img)

        if (total_box > 1):
            print("##### total box over ###### ", file_path)
            err_path.append(file_path)

        if (total_box < 1):
            print("##### total box under ###### ",file_path)
            err_path.append(file_path)
        suc_cnt = suc_cnt + total_box

def mv_dataset(paths,move_paths,val_str):
    jpg_datas = glob(paths + "*.jpg")
    while (len(jpg_datas) < 1):
        paths += "*/"
        jpg_datas = glob(paths + "*.jpg")
    data_paths = glob(paths + "*.jpg")
    data_paths = np.array(data_paths)
    np.random.shuffle(data_paths)
    i = 1
    for data in data_paths[:1000]:
        printProgress(i, 1000, 'create validation ' + val_str)
        shutil.move(data, move_paths + "image_" + str(i).zfill(5) + ".jpg")
        i += 1

# Create validation dataset
def create_validation_dataset(mv_path):
    fake_paths = mv_path+"/fake/"
    real_paths = mv_path+"/real/"
    move_fake_paths = mv_path+"/validation/fake/samples/"
    if not os.path.exists(move_fake_paths):
        os.makedirs(move_fake_paths)
    move_real_paths = mv_path + "/validation/real/samples/"
    if not os.path.exists(move_real_paths):
        os.makedirs(move_real_paths)

    mv_dataset(fake_paths, move_fake_paths,'fake') # move fake 1000 data
    mv_dataset(real_paths, move_real_paths,'real') # move real 1000 data



total_sample =0
def crop_dataset(origin_paths,mk_paths):
    '''
    원본 데이터 경로 : ./data/CW/20201117/image.jpg
    origin_paths : ./data
    mk_paths : ./make_dataset
    result : ./make_dataset/CW/20201117/image.jpg (얼굴 데이터 )
    '''
    global total_sample

    jpg_datas = glob(origin_paths+"/*.jpg")
    print("total_sample:",total_sample,"face_cnt :",suc_cnt)
    if len(jpg_datas)>0:
        write_face(jpg_datas, mk_paths)
        total_sample += len(jpg_datas)
    else:
        origin_paths += "/*"
        for path in glob(origin_paths):
            mk_dir = cp_dir(path, mk_paths)
            crop_dataset(path+"/",mk_dir)


# 데이터에서 얼굴을 찾아 자른 얼굴 이미지 리턴
def get_faces(img_paths, input_shape):
    '''
    img_paths : 원본 데이터 경로
    input_shape : 얼굴 이미지의 Shape
    '''
    input_shape = input_shape
    faces = np.zeros((len(img_paths),input_shape[0],input_shape[1],input_shape[2]),dtype=np.float32)
    tmp_dir = "./temp/"
    if not os.path.exists(tmp_dir):
        os.makedirs(tmp_dir)

    i=0
    for path in img_paths:
        img = cv2.imread(path)
        bbox, scores, landmarks = mtcnn_fun(img, 100, 0.7, [0.6, 0.7, 0.8])
        bbox, scores, landmarks = bbox.numpy(), scores.numpy(), landmarks.numpy()
        total_box = len(bbox)

        # not found face box
        if (total_box < 1):
            # 180 rotate image
            img = cv2.rotate(img, cv2.ROTATE_180)
            bbox, scores, landmarks = mtcnn_fun(img, 100, 0.7, [0.6, 0.7, 0.8])
            bbox, scores, landmarks = bbox.numpy(), scores.numpy(), landmarks.numpy()
            total_box = len(bbox)

        max_idx = 0
        # 얼굴 박스가 2개 이상일 경우 크게 잡은 얼굴 기준
        if total_box > 1: # face box
            max_idx = get_max_size_box(bbox)
        bbox = [bbox[max_idx]]
        landmarks = [landmarks[max_idx]]

        for box, pts in zip(bbox, landmarks):
            box = box.astype('int32')
            pts = pts.astype('int32')
            f_img = margine_face(img, box)
            f_img = cv2.resize(f_img, (input_shape[0], input_shape[1]))
            cv2.imwrite(tmp_dir+"__temp__.jpg",f_img)
            gfile = tf.io.read_file(tmp_dir+"__temp__.jpg")
            f_img = tf.image.decode_png(gfile)
            os.remove(tmp_dir+"__temp__.jpg")

        faces[i] = f_img
        i+=1
    faces = tf.cast(faces,tf.float32)
    return faces # batchsizextensor


# Loading Bar
def printProgress(iteration, total, prefix = '', suffix = '', decimals = 1, barLength = 50):
    formatStr = "{0:." + str(decimals) + "f}"
    percent = formatStr.format(100 * (iteration / float(total)))
    filledLength = int(round(barLength * iteration / float(total)))
    bar = '#' * filledLength + '-' * (barLength - filledLength)
    sys.stdout.write('\r%s |%s| %s%s %s' % (prefix, bar, percent, '%', suffix)),
    if iteration == total:
    	sys.stdout.write('\n')
    sys.stdout.flush()

def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('data_path', type=str,
                        help='Path to the data directory  ex) /home/dacon/deepfake_1st')
    parser.add_argument('create_data_path', type=str,
                        help='Path to the create crop face data directory  ex) /home/dacon/cropface')
    return parser.parse_args(argv)

if __name__ == '__main__':
    '''
    학습용 얼굴 데이터셋 생성
    입력:
        데이터셋 경로  ex) /home/dacon/deepfake_1st
        생성할 데이터셋 폴더 경로  ex) /home/dacon/cropface
    
    결과:
        얼굴 데이터셋 생성 (TRAIN 데이터 폴더: fake,real  TEST 데이터 폴더: validation)
    '''
    argv = parse_arguments(sys.argv[1:])
    crop_dataset (argv.data_path,argv.create_data_path)
    create_validation_dataset(argv.create_data_path)
