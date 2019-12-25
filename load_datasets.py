from keras.utils import to_categorical
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from google.colab import drive

def load_kddcup():
    #importing the dataset
    drive.mount('/content/gdrive')
    dataset = pd.read_csv('/content/gdrive/My Drive/kddcup.data_10_percent_corrected.csv')
    
    #change Multi-class to binary-class
    dataset = dataset.replace(['back', 'buffer_overflow', 'ftp_write', 'guess_passwd', 'imap', 'ipsweep', 'land', 'loadmodule', 'multihop', 'neptune', 'nmap', 'perl', 'phf', 'pod', 'portsweep', 'rootkit', 'satan', 'smurf', 'spy', 'teardrop', 'warezclient', 'warezmaster'], 'attack')
    #dataset['normal.'] = dataset['normal.'].replace(['back.', 'buffer_overflow.', 'ftp_write.', 'guess_passwd.', 'imap.', 'ipsweep.', 'land.', 'loadmodule.', 'multihop.', 'neptune.', 'nmap.', 'perl.', 'phf.', 'pod.', 'portsweep.', 'rootkit.', 'satan.', 'smurf.', 'spy.', 'teardrop.', 'warezclient.', 'warezmaster.'], 'attack')
    x = dataset.iloc[:, :-1].values
    y = dataset.iloc[:, 41].values
    
    #encoding categorical data
    from sklearn.preprocessing import LabelEncoder, OneHotEncoder
    labelencoder_x_1 = LabelEncoder()
    labelencoder_x_2 = LabelEncoder()
    labelencoder_x_3 = LabelEncoder()
    x[:, 1] = labelencoder_x_1.fit_transform(x[:, 1])
    x[:, 2] = labelencoder_x_2.fit_transform(x[:, 2])
    x[:, 3] = labelencoder_x_3.fit_transform(x[:, 3])
    onehotencoder_1 = OneHotEncoder(categorical_features = [1])
    x = onehotencoder_1.fit_transform(x).toarray()
    onehotencoder_2 = OneHotEncoder(categorical_features = [4])
    x = onehotencoder_2.fit_transform(x).toarray()
    onehotencoder_3 = OneHotEncoder(categorical_features = [70])
    x = onehotencoder_3.fit_transform(x).toarray()
    labelencoder_y = LabelEncoder()
    y = labelencoder_y.fit_transform(y)
    
    #splitting the dataset into the training set and test set
    #from sklearn.cross_validation import train_test_split
    from sklearn.model_selection import train_test_split
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 0)
    
    #feature scaling
    from sklearn.preprocessing import StandardScaler
    sc_x = StandardScaler()
    x_train = sc_x.fit_transform(x_train)
    x_test = sc_x.transform(x_test)
    #in tike ro az code haye paein bardashti
    """x_train = x_train.reshape(-1, 32, 32, 3).astype('float32') / 255.
    x_test = x_test.reshape(-1, 32, 32, 3).astype('float32') / 255.
    y_train = to_categorical(y_train.astype('float32'))
    y_test = to_categorical(y_test.astype('float32'))"""
    return (x_train, y_train), (x_test, y_test)

def load_cifar10():
    from keras.datasets import cifar10
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    x_train = x_train.reshape(-1, 32, 32, 3).astype('float32') / 255.
    x_test = x_test.reshape(-1, 32, 32, 3).astype('float32') / 255.
    # mean = np.mean(x_train, axis=(0,1,2))
    # x_train -= mean
    # x_test -= mean
    
    y_train = to_categorical(y_train.astype('float32'))
    y_test = to_categorical(y_test.astype('float32'))
    
#     return (x_train[0:100], y_train[0:100]), (x_test[0:100], y_test[0:100])
    return (x_train, y_train), (x_test, y_test)

def load_cifar100():
    from keras.datasets import cifar100
    (x_train, y_train), (x_test, y_test) = cifar100.load_data()

    x_train = x_train.reshape(-1, 32, 32, 3).astype('float32') / 255.
    x_test = x_test.reshape(-1, 32, 32, 3).astype('float32') / 255.
    y_train = to_categorical(y_train.astype('float32'))
    y_test = to_categorical(y_test.astype('float32'))

    return (x_train, y_train), (x_test, y_test)

def load_mnist():
    # the data, shuffled and split between train and test sets
    from keras.datasets import mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255.
    x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255.
    y_train = to_categorical(y_train.astype('float32'))
    y_test = to_categorical(y_test.astype('float32'))
    return (x_train, y_train), (x_test, y_test)


def load_fmnist():
    # the data, shuffled and split between train and test sets
    from keras.datasets import fashion_mnist
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

    x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255.
    x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255.
    y_train = to_categorical(y_train.astype('float32'))
    y_test = to_categorical(y_test.astype('float32'))
    return (x_train, y_train), (x_test, y_test)


def load_svhn():
    from scipy import io as spio
    from keras.utils import to_categorical
    import numpy as np
    svhn = spio.loadmat("train_32x32.mat")
    x_train = np.einsum('ijkl->lijk', svhn["X"]).astype(np.float32) / 255.
    y_train = to_categorical((svhn["y"] - 1).astype('float32'))

    svhn_test = spio.loadmat("test_32x32.mat")
    x_test = np.einsum('ijkl->lijk', svhn_test["X"]).astype(np.float32) / 255.
    y_test = to_categorical((svhn_test["y"] - 1).astype('float32'))

    print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
    return (x_train, y_train), (x_test, y_test)


def get_annotations_map():
    valAnnotationsPath = 'tiny_imagenet/tiny-imagenet-200/val/val_annotations.txt'
    valAnnotationsFile = open(valAnnotationsPath, 'r')
    valAnnotationsContents = valAnnotationsFile.read()
    valAnnotations = {}

    for line in valAnnotationsContents.splitlines():
        pieces = line.strip().split()
        valAnnotations[pieces[0]] = pieces[1]

    return valAnnotations


def load_tiny_imagenet(path,num_classes):
    #Load images
    X_train=np.zeros([num_classes*500,64,64, 3],dtype='uint8')
    y_train=np.zeros([num_classes*500], dtype='uint8')

    trainPath=path+'/train'
    
    i=0
    j=0
    annotations={}
    for sChild in os.listdir(trainPath):
        sChildPath = os.path.join(os.path.join(trainPath,sChild),'images')
        annotations[sChild]=j
        for c in os.listdir(sChildPath):
            X=np.array(Image.open(os.path.join(sChildPath,c)))
            if len(np.shape(X))==2:
                X_train[i,:,:,0]=X/255.
                X_train[i,:,:,1]=X/255.
                X_train[i,:,:,2]=X/255.
            else:
                X_train[i]=X/255.
            y_train[i]=j
            i+=1
        j+=1
        if (j >= num_classes):
            break

    val_annotations_map = get_annotations_map()

    X_test = np.zeros([num_classes*50,64,64, 3],dtype='uint8')
    y_test = np.zeros([num_classes*50], dtype='uint8')

    i = 0
    testPath=path+'/val/images'
    for sChild in os.listdir(testPath):
        if val_annotations_map[sChild] in annotations.keys():
            sChildPath = os.path.join(testPath, sChild)
            X=np.array(Image.open(sChildPath))
            if len(np.shape(X))==2:
                X_train[i,:,:,0]=X/255.
                X_train[i,:,:,1]=X/255.
                X_train[i,:,:,2]=X/255.
            else:
                X_test[i]=X/255.
            y_test[i]=annotations[val_annotations_map[sChild]]
            i+=1
        else:
            pass
    
    y_train = to_categorical(y_train.astype('float32'))
    y_test = to_categorical(y_test.astype('float32'))
    return X_train,y_train,X_test,y_test

def resize(data_set, size):
    X_temp = []
    import scipy
    for i in range(data_set.shape[0]):
        resized = scipy.misc.imresize(data_set[i], (size, size))
        X_temp.append(resized)
    X_temp = np.array(X_temp, dtype=np.float32) / 255.
    return X_temp
