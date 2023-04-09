# Pose Model

## Usage

```bash
usage: <python code>.py [-h] [--device DEVICE] [--epochs EPOCHS] [--train_ratio TRAIN_RATIO] [--train-data-folder TRAIN_DATA_FOLDER]
                  [--valid-data-folder VALID_DATA_FOLDER] [--test-data-folder TEST_DATA_FOLDER] [--video-dir VIDEO_DIR] [--csv-dir CSV_DIR]

options:
  -h, --help            show this help message and exit
  --device DEVICE       device to train on
  --epochs EPOCHS       number of epochs to train for (train.py)
  --train_ratio TRAIN_RATIO
                        ratio of data to use for training (train.py)
  --train-data-folder TRAIN_DATA_FOLDER
                        path to train data folder (train.py)
  --valid-data-folder VALID_DATA_FOLDER
                        path to validation data folder (train.py)
  --test-data-folder TEST_DATA_FOLDER
                        path to test data folder (predict.py)
  --video-dir VIDEO_DIR
                        path to video folder (getdata.py)
  --csv-dir CSV_DIR     path to data folder for extract (getdata.py)
```
