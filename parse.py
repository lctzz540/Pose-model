import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--device", type=str, default="cpu",
                    help="device to train on")
parser.add_argument(
    "--epochs", type=int, default=10, help="number of epochs to train for (train.py)"
)
parser.add_argument(
    "--train_ratio",
    type=float,
    default=0.75,
    help="ratio of data to use for training (train.py)",
)
parser.add_argument(
    "--train-data-folder",
    type=str,
    default="./dataset/train/",
    help="path to train data folder (train.py)",
)
parser.add_argument(
    "--valid-data-folder",
    type=str,
    default="./dataset/valid/",
    help="path to validation data folder (train.py)",
)
parser.add_argument(
    "--test-data-folder",
    type=str,
    default="./dataset/test/",
    help="path to test data folder (predict.py)",
)
parser.add_argument(
    "--video-dir",
    type=str,
    default="./video/test video/",
    help="path to video folder (getdata.py)",
)

parser.add_argument(
    "--csv-dir",
    type=str,
    default="./dataset/test/",
    help="path to data folder for extract (getdata.py)",
)
args = parser.parse_args()
