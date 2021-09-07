## Setup
Requirement: Cuda 10.0 \
Install dependencies: `pip install -r requirements.txt` \
Install DCNv2: `cd models/DCNv2 & ./make.sh`

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
