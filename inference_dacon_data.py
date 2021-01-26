from glob import glob
import tensorflow as tf


import data_generator_dacon as data_generator
import triplet_loss_dacon as triplet_loss
import crop_face_dacon as crop_face

import numpy as np

from sklearn.svm import SVC
import csv
import pickle
import sys
import argparse



def dacon_leaderboard(model,dacon_data,csv_path, svm_path):
    batch_size = dacon_data.batch_size

    # load svm
    with open (svm_path, 'rb') as infile:
        (svc_model) = pickle.load (infile)

    # detect face & model predict
    answers = np.zeros ((dacon_data.size_of_sample), np.uint8)
    for it in range (dacon_data.step_per_epoch):
        x_test = crop_face.get_faces(dacon_data.X[it * batch_size:(it + 1) * batch_size], dacon_data.input_shape)
        crop_face.printProgress(it,dacon_data.step_per_epoch,'Read Data') #print Loarding Bar
        val_embeddings = model.predict (x_test)
        predictions = svc_model.predict_proba (val_embeddings)
        answers[it * batch_size: (it + 1) * batch_size] = np.argmax(predictions, axis=1)

    # write submission csv
    p_names = []
    for it in range (dacon_data.size_of_sample):
        p_ = tf.strings.split (dacon_data.X[it], "\\")[-1]
        p_names.append (p_)

    f = open (csv_path, 'r')
    rdr = csv.reader (f)
    lines = []
    idx = 0
    for line in  (rdr):
        if idx is not 0:
            fname_ = line[0]
            fname_ = tf.strings.split (fname_, "/")[-1]

            a = tf.equal (fname_, p_names)
            b = np.argmax (a)
            line[1] = answers[b]

        idx = idx + 1
        lines.append (line)


    f.close()
    csv_split = csv_path.split('.')
    csv_out_path = csv_split[-2] + '_out.csv'
    write_f = open (csv_out_path, 'w+', newline='')
    wr = csv.writer (write_f)
    wr.writerows (lines)
    write_f.close ()


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('data_path', type=str,
                        help='Path of inference data directory  ex) /home/dacon/data')
    parser.add_argument('csv_path', type=str,
                        help='sample_submission.csv  ex)./home/dacon/sample_submission.csv')
    parser.add_argument ('network_model', type=str,
                         help='network model hdf5 file ex)./home/dacon/resource/triplet_effB4_ep06_BS28.hdf5')
    parser.add_argument ('svm_model', type=str,
                         help='classifier pickle file ex) /home/dacon/resource/triplet_effB4_ep06_BS28.pkl')
    return parser.parse_args(argv)

def inference_dacon(args) :
    input_shape = (380, 380, 3)
    batch_size = 10

    # load_model_path = 'resource/triplet_effB4_ep06_BS28.hdf5'
    load_model_path = args.network_model
    model = tf.keras.models.load_model (load_model_path, custom_objects={'triplet_loss_adapted_from_tf':triplet_loss.triplet_loss_adapted_from_tf} )

    dacon_paths = glob (args.data_path+'/*.jpg')
    dacon_steps = len (dacon_paths) // batch_size
    dacon_data = data_generator.DataGenerator (dacon_paths, dacon_paths, batch_size, dacon_steps, input_shape, False, False)
    dacon_leaderboard(model, dacon_data, args.csv_path, args.svm_model)



if __name__ == '__main__':
    print(sys.argv[1:])
    inference_dacon (parse_arguments(sys.argv[1:]))

