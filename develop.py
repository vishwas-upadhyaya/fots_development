import streamlit as st
import matplotlib.pyplot as plt

from ml_models import inferencePipeline_lite
import cv2

import numpy as np

st.title('TEXT DETECTION')

st.header('Input Image')

img=st.file_uploader('upload a image')

#st.text(img.shape)
if img:

    file_bytes = np.asarray(bytearray(img.read()), dtype=np.uint8)
    opencv_image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    # bgr_image = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)

    # Convert BGR to RGB
    # print(opencv_image)
    # opencv_image = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB)
    # plt.imsave('xxx.png',opencv_image)

    #img = cv2.imread('xxx.png', cv2.IMREAD_UNCHANGED)
    #print(img.shape)
    im=inferencePipeline_lite(opencv_image)

    im=cv2.resize(im,dsize=(1200,720))
    st.text('Input Image')
    st.image(img)
    st.text('Output Image')
    st.image(im)