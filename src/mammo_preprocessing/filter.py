import cv2
import os

for root, dirs, files in os.walk('/your_path/2024-1-P2-Estudo-de-Arquiteturas-de-Redes-Neurais-Profundas-para-Analise-de-Imagens-Medicas/dataset/Calc-CC_flipped_dataset/crops/BENIGN'):
    for i, f in enumerate(files):
        img = cv2.imread(root + '/' + f)
        imgAux = img[:, :img.shape[1], :]
        if ((imgAux == 0).sum() >= 50160):
            cv2.imwrite(root + '/trash_files/' + f, imgAux); os.remove(root + '/' + f)
        del img
