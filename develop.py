import streamlit as st

from ml_models import inferencePipeline_lite
import cv2

import numpy as np

st.title('TEXT DETECTION')

st.header('Input Image')

img=st.file_uploader('upload a image')

#st.text(img.shape)
if img:
    st.image(img)
    file_bytes = np.asarray(bytearray(img.read()), dtype=np.uint8)
    opencv_image = cv2.imdecode(file_bytes, 1)
    #plt.imsave('xxx.png',opencv_image)

    #img = cv2.imread('xxx.png', cv2.IMREAD_UNCHANGED)
    #print(img.shape)
    im=inferencePipeline_lite(opencv_image)

    im=cv2.resize(im,dsize=(1200,720))
    st.image(im)