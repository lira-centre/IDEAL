import torch
import os
from torchvision.models.feature_extraction import create_feature_extractor
from torchvision.models.feature_extraction import get_graph_node_names
from torchvision.datasets import CIFAR100
import pickle
import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.spatial.distance import cdist
from sklearn.metrics import confusion_matrix
from scipy.special import softmax
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, cohen_kappa_score, confusion_matrix
# import IDEAL_optimization as IDEAL
from torchvision.models import vit_b_16, ViT_B_16_Weights
import time
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import argparse
import torchvision
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier


def unpickle(file):
    '''Load byte data from file'''
    with open(file, 'rb') as f:
        data = pickle.load(f,encoding='latin1')
    return data



def load_cifar10_data(data_dir):
    '''
    Return train_data, train_labels, test_data, test_labels
    The shape of data returned would be as it is in the data-set N X 3072

    We don't particularly need the metadata - the mapping of label numbers to real labels
    '''
    train_data = None
    train_labels = []

    for i in range(1, 6):
        data_dic = unpickle(data_dir + "/data_batch_"+str(i))
        if i == 1:
            train_data = data_dic['data']
        else:
            train_data = np.append(train_data, data_dic['data'])
        train_labels += data_dic['labels']

    test_data_dic = unpickle(data_dir + "/test_batch")
    test_data = test_data_dic['data']
    test_labels = test_data_dic['labels']
    names=unpickle(data_dir+'/batches.meta')
    
    return train_data, np.array(train_labels), test_data, np.array(test_labels), names['label_names']


# Load the pre-trained model and perform feature extraction

def feature_extraction(img,model):
    
    img = img.to(device)
    
        #Perform transformation

    feature_extractor = create_feature_extractor(
        model, return_nodes=['getitem_5'])

    with torch.no_grad():
        out = feature_extractor(img)
        
    return out['getitem_5']
    
    
    

class IDEALClassifier:
    def train(self, Input, random_state):

        Images = Input['Images']
        Features = Input['Features']
        Labels = Input['Labels']
        CN = max(Labels)
        Prototypes = self.PrototypesIdentification(Images,Features,Labels,CN,random_state)
        Output = {}
        Output['IDEALParms'] = {}
        Output['IDEALParms']['Parameters'] = Prototypes
        MemberLabels = {}
        for i in range(0,CN+1):
            MemberLabels[i]=Input['Labels'][Input['Labels']==i] 
        Output['IDEALParms']['CurrentNumberofClass']=CN+1
        Output['IDEALParms']['OriginalNumberofClass']=CN+1
        Output['IDEALParms']['MemberLabels']=MemberLabels
        return Output
    
    def predict(self,Input):
    
        Params=Input['IDEALParms']
        datates=Input['Features']
        Test_Results = self.DecisionMaking(Params,datates)
        EstimatedLabels = Test_Results
        Output = {}
        Output['EstLabs'] = EstimatedLabels
        Output['ConfMa'] = confusion_matrix(Input['Labels'],Output['EstLabs'])
        Output['ClassAcc'] = np.sum(Output['ConfMa']*np.identity(len(Output['ConfMa'])))/len(Input['Labels'])
        return Output

    def PrototypesIdentification(self, Image,GlobalFeature,LABEL,CL,random_state):
        data = {}
        image = {}
        label = {}
        Prototypes = {}
        for i in range(0,CL+1):
            seq = np.argwhere(LABEL==i)
            data[i]=GlobalFeature[seq]
            image[i] = {}
            for j in range(0, len(seq)):
                image[i][j] = Image[seq[j][0]]
            label[i] = np.ones((len(seq),1))*i
        for i in range(0, CL+1):
            data[i] = np.squeeze(data[i],axis=1)
            print(data[i].shape)
            Prototypes[i] = data[i][np.random.choice(data[i].shape[0], 100, replace=False)]
            print(Prototypes[i].shape)
        
        return Prototypes
        
    

    def DecisionMaking(self, Params,datates,NN=1):
        PARAM=Params['Parameters']
        # convert dictionary to array
        features_prop = np.concatenate([PARAM[i] for i in range(len(PARAM))])
        labels_prop = np.concatenate([np.ones(PARAM[i].shape[0])*i for i in range(len(PARAM))]).reshape(-1,1)
        
        winner_model = KNeighborsClassifier(n_neighbors=NN)
        winner_model.fit(features_prop, labels_prop)
        
        results = winner_model.predict(datates)
        
        return results
        
            
    def save_model(self, model, name='IDEAL_model'):
        with open(name, 'wb') as file:
            pickle.dump(model, file)

    def load_model(self, name='IDEAL_model'):
        with open(name, 'rb') as file:
            pickle_model = pickle.load(file)
        return pickle_model

    def results(self, predicted,y_test_labels):

        accuracy = accuracy_score(y_test_labels , predicted)
        print('Accuracy: %f' % accuracy)
        # precision tp / (tp + fp)
        precision = precision_score(y_test_labels ,predicted, average='weighted')
        print('Precision: %f' % precision)
        # recall: tp / (tp + fn)
        recall = recall_score(y_test_labels , predicted,average='weighted')
        print('Recall: %f' % recall)
        # f1: 2 tp / (2 tp + fp + fn)
        f1 = f1_score(y_test_labels , predicted, average='weighted')
        print('F1 score: %f' % f1)
        # kappa
        kappa = cohen_kappa_score(y_test_labels , predicted)
        print('Cohens kappa: %f' % kappa)
        # confusion matrix
        matrix = confusion_matrix(y_test_labels , predicted)
        print("Confusion Matrix: ",matrix)
        

    


if __name__ == '__main__':
    # Load the cifa-10 dataset
    
    # os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"   
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    train_features, test_features = [], []
    train_labels, test_labels = [], []
    

   

    # Load the pretrained model
    weights = ViT_B_16_Weights.IMAGENET1K_SWAG_E2E_V1
    model = vit_b_16(weights=weights)
    model.eval()
    model = model.to(device)
    
    


    trainset = torchvision.datasets.CIFAR100(
        root='./data', train=True, download=True, transform=weights.transforms())
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=200, shuffle=False, num_workers=4)

    testset = torchvision.datasets.CIFAR100(
        root='./data', train=False, download=True, transform=weights.transforms())
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=200, shuffle=False, num_workers=4)

    classes = ('plane', 'car', 'bird', 'cat', 'deer',
            'dog', 'frog', 'horse', 'ship', 'truck')
    
    
    
    start = time.time()
       
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            inputs = inputs.to(device)
            train_features.append(feature_extraction(inputs,model).detach().cpu().numpy())
            train_labels.append(targets.numpy())
         
        
    end = time.time()

    print ("###################### training feature extraction ####################")
    print("Training feature extraction Time: ",round(end - start,2), "seconds")
    
    
    start = time.time()
        
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs = inputs.to(device)
            test_features.append(feature_extraction(inputs,model).detach().cpu().numpy())
            test_labels.append(targets.numpy())
        
        
    end = time.time()
    print("Test feature extraction Time: ",round(end - start,2), "seconds")   
    
    train_features = np.array(train_features)
    train_features = train_features.reshape((-1,768))
    test_features = np.array(test_features)
    test_features = test_features.reshape((-1,768))
    train_labels = np.array(train_labels)
    test_labels = np.array(test_labels).reshape((-1,1))
    
    
    train_features_labels = np.concatenate((train_features, train_labels.reshape((-1,1))), axis=1)
    
    print(train_features_labels[0,:])
    
    
    acc = []
    
    for each_shuffle in range(5):
        
        # Shuffle the training data
        np.random.seed(each_shuffle)
        np.random.shuffle(train_features_labels)
    
   
        
        print(train_features_labels[0,:])
        
        train_features = train_features_labels[:,:-1]
        
        train_labels = train_features_labels[:,-1].astype(int)
        
        # Data Input (dict)

        Input = {}

        Input['Images'] = train_labels
        Input['Features'] = train_features
        Input['Labels'] = train_labels

        #########################################################################
        # IDEAL Training

        # Model Definition
        model = IDEALClassifier()

        # IDEAL learning
        start = time.time()
        x = model.train(Input,each_shuffle)
        end = time.time()

        print ("###################### Model Trained ####################")

        print("Training Time: ",round(end - start,2), "seconds")

        # IDEAL parameters (optional for investigation)
        print ("###################### Parameters ####################")

        Prototypes =x['IDEALParms']['Parameters']
        total_prototypes = 0
        print('Number of Training Data Samples:', len(train_features))

        for i in range(len(Prototypes)):
            class_prototypes = len(Prototypes[i])
            total_prototypes = total_prototypes + class_prototypes
            print("Class", i+1, ":", class_prototypes)

        print("Total   :", total_prototypes)
        print("Prototypes as % of the Training Data Samples:", total_prototypes/len(train_features)*100, "%")


        print ("###################### Visual Prototypes ####################")

        Prototypes = x['IDEALParms']['Parameters']
        total_prototypes = 0

        for i in range(len(Prototypes)):
            class_prototypes = len(Prototypes[i])
            total_prototypes = total_prototypes + class_prototypes
            print("Number of prototypes Class", i+1, ":", class_prototypes)
            print("Prototypes : ", Prototypes[i])
            print(" ")

        # Save IDEAL model (optional)
        # model.save_model(x,r'E:\Lancaster_PhD\PHD\Data\WORLDFLOODS\Code\NeurIPS_IDEAL\model\IDEAL_resnet50')
        model.save_model(x,r'/mmfs1/scratch/hpc/00/zhangz65/Code/IDEAL/code/update_on_cifar100/models/random_vit_'+str(each_shuffle)+'_model')



        print ("###################### Validation ####################")

        TestData = {}

        TestData ['IDEALParms'] =  x['IDEALParms']
        TestData ['Images'] = test_labels 
        TestData ['Features'] = test_features
        TestData ['Labels'] = test_labels

        start = time.time()
        # IDEAL Predict
        pred= model.predict(TestData)
        end = time.time()
        print("Validation Time: ",round(end - start,2), "seconds")


        # IDEAL Results
        print ("###################### Results ####################")

        model.results(pred['EstLabs'],test_labels)

        acc.append(accuracy_score(test_labels, pred['EstLabs']))
        
    acc = np.array(acc)
    print("Accuracy: ", acc)
    print("Average accuracy: ", np.mean(acc))
    print("Average accuracy std: ", np.std(acc))

        
        


