## Setup
Requirement: Cuda 10.0
Install dependencies: `pip install -r requirements.txt`
Install DCNv2: `cd models/DCNv2 & ./make.sh`

You can download sample demo inputs and checkpoint using `./prepare.sh`
Or
they can be download from: `https://drive.google.com/drive/folders/1q-qfp_PnnJUcd9sTU-dWGJOEsWt8ioFy?usp=sharing`

unzip checkpoint JHU_CKPT.zip into `./exp` directory
unzip demo_input.zip into directory `./datasets/Demo`
The folder structure should be as follow:
```
datasets
    |-----Demo
            |------img
                    |-----Demo1.jpg
                    ...
                    |-----Demo99.jpg
            |------label
                    |-----Demo1.txt
                    ...
                    |-----Demo99.txt
            |------dataset_config.py
            |------demo_dataloader.py
            |------loading_data.py
    |-----exp
            |------JHU_CKPT
                    |--------jhu_ckpt.pth
    ...
```

## Run
Demo can be ran using `python3 demo.py`
The localization output will be placed under `output/JHU_CKPT` directory

## More sample outputs
Output samples of Shanghai Tech B test set can also be found under `https://drive.google.com/drive/folders/1q-qfp_PnnJUcd9sTU-dWGJOEsWt8ioFy?usp=sharing`