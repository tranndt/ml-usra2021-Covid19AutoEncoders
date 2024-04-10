# importing libraries
import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector
import os
import argparse
import sys

sys.path.insert(0, "./")

from src.make_directory import make_directory
from src.data_preprocessing import dataset_dict


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_type', required=True, help="automatically looks into data directory"
                                                              "example would be breast-cancer-wisconsin.data")
    # if using covid-19 dataset, should add this argument
    parser.add_argument('--covid_type', choices=['covid_result', 'intensive_result'])
    arg = parser.parse_args()
    arg.random_seed = 42  # for placeholder to prevent error
    arg.num_folds = 5  # for placeholder to prevent error

    dataset_filename = os.path.join('data', arg.dataset_type)
    dataset = dataset_dict[arg.dataset_type](dataset_filename, arg)
    dataset.process_dataset()  # get X and Y value

    # save directory
    save_dir = os.path.join('visualization', arg.dataset_type)
    metadata_path = os.path.join(save_dir, 'metadata.tsv')
    make_directory(save_dir)  # make directory for save_dir if not exist

    # process the data into embedding projector in tensorflow
    data_tensor = tf.Variable(dataset.X, name='data_information')
    # process the y value into metadata to visualize
    with open(metadata_path, 'w') as f:
        for value in dataset.Y:
            f.write(f'{value}\n')

    # save the input data value
    sess = tf.Session()
    saver = tf.train.Saver([data_tensor])
    sess.run(data_tensor.initializer)
    saver.save(sess, os.path.join(save_dir, "data_information.ckpt"))

    # save the input data y value so that we can see which embedding correspond to the appropriate target value
    config = projector.ProjectorConfig()
    embedding = config.embeddings.add()
    embedding.tensor_name = data_tensor.name
    embedding.metadata_path = 'metadata.tsv'
    projector.visualize_embeddings(tf.summary.FileWriter(save_dir), config)
