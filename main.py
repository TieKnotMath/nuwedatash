from train import train
from utils import data_augmentation
from models import predict



def main():
    data_augmentation('train.csv')
    train()
    predict()

if __name__ == '__main__':
    main()




