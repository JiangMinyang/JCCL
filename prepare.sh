#!/bin/bash
gdown -O datasets/Demo/sample_input.zip https://drive.google.com/uc?id=1BTlNDgYUz4G5ynsSGLk991IK2-_iowv1
gdown -O exp/JHU_CKPT.zip https://drive.google.com/uc?id=12dBO9cLbQrwYGsevpDosXiNVotB7aKhG

cd datasets/Demo/
unzip sample_input.zip
rm sample_input.zip
cd ../../exp
unzip JHU_CKPT.zip
rm JHU_CKPT.zip
cd ..