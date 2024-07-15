# python test_prob_cancer_tissue.py metrics/test_dataset_32.csv metrics/confusion_matrix_validation.txt metrics/probabilities.csv

from __future__ import division, print_function
import os, shutil, time, random
import numpy as np
import pandas as pd
import cv2
import sys
import torch
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms


NETWORK = 'squeezenet1_1'

NUM_CLASSES = 2

INITIAL_MODEL = '/your_path/2024-1-P2-Estudo-de-Arquiteturas-de-Redes-Neurais-Profundas-para-Analise-de-Imagens-Medicas/src/squeezeNet/runs_cancer_tissue/squeezenet1_1/02_57344_864955357/models/squeezenet1_1_32_2.pth'
#'/your_path/2024-1-P2-Estudo-de-Arquiteturas-de-Redes-Neurais-Profundas-para-Analise-de-Imagens-Medicas/src/squeezeNet/runs/squeezenet1_1/05_57344_853097098/models/squeezenet1_1_43_4.pth'

INITIAL_MODEL_TEST = True

TEST = (
        # '/your_path/2024-1-P2-Estudo-de-Arquiteturas-de-Redes-Neurais-Profundas-para-Analise-de-Imagens-Medicas/dataset/cancer_tissue_dataset/crop_mammo_test_32.txt',
        '/your_path/2024-1-P2-Estudo-de-Arquiteturas-de-Redes-Neurais-Profundas-para-Analise-de-Imagens-Medicas/dataset/cancer_tissue_dataset/crop_mammo_test_64.txt',
)
TEST_DIR = (
        # '/your_path/2024-1-P2-Estudo-de-Arquiteturas-de-Redes-Neurais-Profundas-para-Analise-de-Imagens-Medicas/dataset',
        # '/your_path/2024-1-P2-Estudo-de-Arquiteturas-de-Redes-Neurais-Profundas-para-Analise-de-Imagens-Medicas/src/mammo_viewer/crop_mammo_test_set',
        '',
)

TRANSFORMS = transforms.Normalize([0.4818, 0.4818, 0.4818], [0.1752, 0.1752, 0.1752])

BATCH_SIZE = 1

NUM_WORKERS = 4


def load_matching_name_and_shape_layers(net, new_model_name, new_state_dict):
    # print('\n' + new_model_name + ':')
    state_dict = net.state_dict()
    for key in state_dict:
        if key in new_state_dict and new_state_dict[key].shape == state_dict[key].shape:
            state_dict[key] = new_state_dict[key]
            # print('\t' + key + ' loaded.')
    net.load_state_dict(state_dict)

def Net():
    model = getattr(models, NETWORK)
    net = model(num_classes=NUM_CLASSES)
    # load_matching_name_and_shape_layers(net, 'Torchvision pretrained model', model(pretrained=True).state_dict())
    return net

class DatasetFromCSV(Dataset):
    def __init__(self, csv_files, root_dirs, label=None, shuffle=False, transforms=None, dataset_file=None):
        data = []
        for csv_file, root_dir in zip(csv_files, root_dirs):
            d = pd.read_csv(csv_file, header=None, names=['images', 'labels'], delim_whitespace=True)
            if label != None:
                d = d.loc[d['labels'] == label].reset_index().drop('index', 1)
            if root_dir != '':
                d['images'] = root_dir + ('' if root_dir.endswith('/') else '/') + d['images']
            data.append(d)
        if len(data) == 1:
            data = data[0]
        else:
            data = pd.concat(data).reset_index().drop('index', 1)
        self.data_len = len(data)


        self.images = np.asarray(data.iloc[:, 0])
        self.labels = np.asarray(data.iloc[:, 1])

        self.transforms = transforms

        if dataset_file != None:
            with open(dataset_file, 'a') as dataset:
                for image, label in zip(self.images, self.labels):
                    dataset.write(image + ',' + str(label) + '\n')
                    # print(image + ',' + str(label) + '\n')


    def __len__(self):
        return self.data_len

    def __getitem__(self, i):
        image = cv2.imread(self.images[i], 3)
        # print(image.shape)
        image = np.transpose(image, [2, 0, 1])[[2, 1, 0]]
        image = image/255
        image = torch.from_numpy(image.astype(np.float32))
        if self.transforms != None:
            image = self.transforms(image)
        return (image, self.labels[i])


def test(net, dataset_name, datasets_per_label, dataloaders_per_label, results_file, probabilidade_file):
    net.eval()
    str_buf = '\n\t' + dataset_name + ':\n\n\t\tConfusion Matrix\tClass Accuracy\n'
    print(str_buf)

    if results_file != None:
        with open(results_file, 'a') as results:
            results.write(str_buf + '\n')
    
    average_class_accuracy = 0.0
    valid_classes = 0


    for i in range(NUM_CLASSES):
        
        dataset, dataloader = datasets_per_label[i], dataloaders_per_label[i]
        
        line = np.zeros(NUM_CLASSES, dtype=int)
        class_accuracy = 0.0
        if dataset.data_len > 0:
            valid_classes += 1
            with torch.no_grad():
                for batch in dataloader:
                    classification = net(batch[0].to('cuda:0'))
                    m = nn.Softmax(dim=1)
                    batch_s = m(classification)
                    batch_s = batch_s.tolist()
                    
                    for s in batch_s:
                        with open(probabilidade_file, 'a') as ptest:
                            ptest.write(str(s[0]) + ',' + str(s[1]) + '\n')
                        # print(s)
                    # exit()
                    c = torch.max(classification, 1)[1].tolist()

                    for j in range(NUM_CLASSES):
                        line[j] += c.count(j)
            class_accuracy = float(line[i])/dataset.data_len
            average_class_accuracy += class_accuracy
        str_buf = '\t'
        for j in range(NUM_CLASSES):
            str_buf += '\t' + str(line[j])
        str_buf += '\t{:.9f}'.format(class_accuracy)
        print(str_buf)
        if results_file != None:
            with open(results_file, 'a') as results:
                results.write(str_buf + '\n')
    average_class_accuracy /= valid_classes
    str_buf = '\n\t\tAverage Class Accuracy: {:.9f}'.format(average_class_accuracy)
    print(str_buf)
    if results_file != None:
        with open(results_file, 'a') as results:
            results.write(str_buf + '\n')



def main(args):
    torch.multiprocessing.set_start_method('spawn', force=True)

    dataset_file = args[1]

    results_file = args[2]

    probabilidade_file = args[3]

    ##ajustar o código para criar a pasta e os arquivos

    net = Net().to('cuda:0')
    if INITIAL_MODEL != None:
        load_matching_name_and_shape_layers(net, INITIAL_MODEL, torch.load(INITIAL_MODEL))


    if TEST != None:
        if INITIAL_MODEL_TEST:
            print('\n' + (INITIAL_MODEL if INITIAL_MODEL != None else 'Initial model') + ' tests:')
        tests = []
        for csv_file, root_dir in zip(TEST, TEST_DIR):
            datasets_per_label = [DatasetFromCSV((csv_file,), (root_dir,), label=i, transforms=TRANSFORMS, dataset_file=dataset_file) for i in range(NUM_CLASSES)]
            dataloaders_per_label = [DataLoader(dataset, BATCH_SIZE, num_workers=NUM_WORKERS) for dataset in datasets_per_label]
            if INITIAL_MODEL_TEST:
                test(net, csv_file, datasets_per_label, dataloaders_per_label, results_file, probabilidade_file)



if __name__ == '__main__':
    sys.exit(main(sys.argv)) 
