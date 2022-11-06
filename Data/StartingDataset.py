!pip -q install pyngrok
!pip -q install streamlit
!pip -q install patool

import cv2
import gdown
import glob
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import patoolib

from joblib import dump
from pyngrok import ngrok
from tqdm import tqdm

from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

DATA_ROOT = '/content/data'
os.makedirs(DATA_ROOT, exist_ok=True)
max_samples = 3000

blood_slide_url = 'https://drive.google.com/uc?id=1lffxAG8gykh1dh1pCP34uRkH3XMwuNt-'
blood_slide_path = os.path.join(DATA_ROOT, 'blood_slide.jpg')
gdown.download(blood_slide_url, blood_slide_path, True)

malaria_imgs_url = 'https://drive.google.com/uc?id=1s2_zVe0JYKvHR5j8w1LEqRzAx19dUejC'
malaria_imgs_path = os.path.join(DATA_ROOT, 'malaria_imgs.zip')
gdown.download(malaria_imgs_url, malaria_imgs_path, True)

if os.path.exists(os.path.join(DATA_ROOT, 'malaria_images')) == False:
  patoolib.extract_archive(os.path.join(DATA_ROOT, 'malaria_imgs.zip'), outdir=DATA_ROOT)

print("Downloaded Data")

u_malaria_img_paths = glob.glob('/content/data/malaria_images/Uninfected/*png')
p_malaria_img_paths = glob.glob('/content/data/malaria_images/Parasitized/*png')

NUM_SAMPLES = len(u_malaria_img_paths) + len(p_malaria_img_paths)

X = []
y = []

X_g = []

for i in tqdm(range(max_samples)):
  img = cv2.imread(u_malaria_img_paths[i])
  X.append(cv2.resize(img,(50,50)))

  gray_img = cv2.imread(u_malaria_img_paths[i],0)
  X_g.append(cv2.resize(gray_img,(50,50)))

  y.append(0)

for i in tqdm(range(max_samples)):
  img = cv2.imread(p_malaria_img_paths[i])
  X.append(cv2.resize(img,(50,50)))

  gray_img = cv2.imread(p_malaria_img_paths[i],0)
  X_g.append(cv2.resize(gray_img,(50,50)))

  y.append(1)

X = np.stack(X)
X_g = np.stack(X_g)
X_reshaped = np.reshape(X_g,(X_g.shape[0],2500))

y = np.array(y)

blood_samples_dir = 'blood_samples'
if (os.path.exists(blood_samples_dir) == False):
  os.mkdir(blood_samples_dir)

for i, img in enumerate(X[2995:3005]):
  plt.imsave('test_img_{}.jpg'.format(i), img)
  
print("Created our X and y variables")
