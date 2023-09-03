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
import time
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import argparse
import torchvision
from torchvision.models import vgg16,VGG16_Weights

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
        model, return_nodes=['classifier.5'])

    with torch.no_grad():
        out = feature_extractor(img)
        
    return out['classifier.5']
    
    

class IDEALClassifier:
    def train(self, Input):

        Images = Input['Images']
        Features = Input['Features']
        Labels = Input['Labels']
        CN = max(Labels)
        Prototypes = self.PrototypesIdentification(Images,Features,Labels,CN)
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
        EstimatedLabels = Test_Results['EstimatedLabels'] 
        Scores = Test_Results['Scores']
        Output = {}
        Output['EstLabs'] = EstimatedLabels
        Output['Scores'] = Scores
        Output['ConfMa'] = confusion_matrix(Input['Labels'],Output['EstLabs'])
        Output['ClassAcc'] = np.sum(Output['ConfMa']*np.identity(len(Output['ConfMa'])))/len(Input['Labels'])
        return Output

    def PrototypesIdentification(self, Image,GlobalFeature,LABEL,CL):
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
            Prototypes[i] = self.IDEALclassifier(data[i],image[i])
        
        return Prototypes
        
    def IDEALclassifier(self, Data,Image):
        L, N, W = np.shape(Data)
        radius =1 - math.cos(math.pi/6)
        Data_2 = Data**2
        Data_2 = Data_2.reshape(-1, 4096)
        Xnorm = np.sqrt(np.sum(Data_2,axis=1))
        
        data = Data.reshape(-1,4096) / (Xnorm.reshape(-1,1))*(np.ones((1,W)))
        Centre = data[0,]
        Centre = Centre.reshape(-1,4096)
        Center_power = np.power(Centre,2)
        X = np.array([np.sum(Center_power)])
        Support =np.array([1])
        Noc = 1
        GMean = Centre.copy()
        Radius = np.array([radius])
        ND = 1
        VisualPrototype = {}
        VisualPrototype[1] = Image[0]
        Global_X = 1
        for i in range(2,L+1):
            GMean = (i-1)/i*GMean+data[i-1,]/i
            GDelta=Global_X-np.sum(GMean**2,axis = 1)
            
            # CDmax=max(CentreDensity)
            # CDmin=min(CentreDensity)
            DataDensity=1/(1+np.sum((data[i-1,] - GMean) ** 2)/GDelta)
            # if i == 2:
            CentreDensity=1/(1+np.sum(((Centre-GMean)**2),axis=1)/GDelta)
            CDmax = CentreDensity.max()
            CDmin = CentreDensity.min()
            distance = cdist(data[i-1,].reshape(1,-1),Centre,'euclidean')[0]
            # else:
            #     distance = cdist(data[i-1,].reshape(1,-1),Centre,'euclidean')[0]
            value,position= distance.min(0),distance.argmin(0)
            if (DataDensity > CDmax or DataDensity < CDmin):
                # if (DataDensity > CDmax):
                #     CDmax = DataDensity
                # elif (DataDensity < CDmin):
                #     CDmin = DataDensity
            # if (DataDensity > CDmax or DataDensity < CDmin) or value >2*Radius[position]:
                Centre=np.vstack((Centre,data[i-1,]))
                Noc=Noc+1
                VisualPrototype[Noc]=Image[i-1]
                X=np.vstack((X,ND))
                Support=np.vstack((Support, 1))
                Radius=np.vstack((Radius, radius))
            else:
                Centre[position,] = Centre[position,]*(Support[position]/(Support[position]+1))+data[i-1]/(Support[position]+1)
                Support[position]=Support[position]+1
                Radius[position]=0.5*Radius[position]+0.5*(X[position,]-sum(Centre[position,]**2))/2  
        dic = {}
        dic['Noc'] =  Noc
        dic['Centre'] =  Centre
        dic['Support'] =  Support
        dic['Radius'] =  Radius
        dic['GMean'] =  GMean
        dic['Prototype'] = VisualPrototype
        dic['L'] =  L
        dic['X'] =  X
        return dic
    

    def DecisionMaking(self, Params,datates,NN=1):
        PARAM=Params['Parameters']
        CurrentNC=Params['CurrentNumberofClass']
        LAB=Params['MemberLabels']
        VV = 1
        LTes=np.shape(datates)[0]
        EstimatedLabels = np.zeros((LTes))
        Scores=np.zeros((LTes,CurrentNC))
        for i in range(1,LTes + 1):
            data = datates[i-1,]
            Data_2 = data**2
            Data_2 = Data_2.reshape(-1, 4096)
            Xnorm = np.sqrt(np.sum(Data_2,axis=1))
            data = data/Xnorm
            R=np.zeros((VV,CurrentNC))
            numPrototypes = 0
            for j in range(CurrentNC):
                numPrototypes = numPrototypes+PARAM[j]['Noc']
            for k in range(0,CurrentNC):
                if k == 0:
                    
                    distance = np.exp(-1*cdist(data.reshape(1, -1),PARAM[k]['Centre'],'euclidean')**2).T
                    label = np.full(distance.shape,k)

                else:
                    distance_new = np.exp(-1*cdist(data.reshape(1, -1),PARAM[k]['Centre'],'euclidean')**2).T
                    label_new = np.full(distance_new.shape,k)

                    distance = np.vstack((distance,distance_new))
                    label = np.vstack((label,label_new))

            distance_label = np.hstack((distance,label))

            distance_label = distance_label[distance_label[:,0].argsort()]

            distance_label = distance_label[-NN:,:]

           
            EstimatedLabels[i-1]=np.argmax(np.bincount(distance_label[:,1].astype(int)))


        LABEL1=np.zeros((CurrentNC,1))
        
        

        for i in range(0,CurrentNC): 
            LABEL1[i] = np.unique(LAB[i])

        EstimatedLabels = EstimatedLabels.astype(int)
        EstimatedLabels = LABEL1[EstimatedLabels]   
        dic = {}
        dic['EstimatedLabels'] = EstimatedLabels
        dic['Scores'] = Scores

        return dic
            
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
    weights = VGG16_Weights.DEFAULT
    model = vgg16(weights=weights)
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
    train_features = train_features.reshape((-1,4096))
    test_features = np.array(test_features)
    test_features = test_features.reshape((-1,4096))
    train_labels = np.array(train_labels)
    test_labels = np.array(test_labels).reshape((-1,1))
    
    
    train_features_labels = np.concatenate((train_features, train_labels.reshape((-1,1))), axis=1)
    
    print(train_features_labels[0,:])
    
    # run 5 times and average the results
    
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
        x = model.train(Input)
        end = time.time()

        print ("###################### Model Trained ####################")

        print("Training Time: ",round(end - start,2), "seconds")

        # IDEAL parameters (optional for investigation)
        print ("###################### Parameters ####################")

        Prototypes =x['IDEALParms']['Parameters']
        total_prototypes = 0
        print('Number of Training Data Samples:', len(train_features))

        for i in range(len(Prototypes)):
            class_prototypes = len(Prototypes[i]['Prototype'])
            total_prototypes = total_prototypes + class_prototypes
            print("Class", i+1, ":", class_prototypes)

        print("Total   :", total_prototypes)
        print("Prototypes as % of the Training Data Samples:", total_prototypes/len(train_features)*100, "%")


        print ("###################### Visual Prototypes ####################")

        Prototypes = x['IDEALParms']['Parameters']
        total_prototypes = 0

        for i in range(len(Prototypes)):
            class_prototypes = len(Prototypes[i]['Prototype'])
            total_prototypes = total_prototypes + class_prototypes
            print("Number of prototypes Class", i+1, ":", class_prototypes)
            print("Prototypes : ", Prototypes[i]['Prototype'])
            print(" ")


        # Save IDEAL model (optional)
        # model.save_model(x,r'E:\Lancaster_PhD\PHD\Data\WORLDFLOODS\Code\NeurIPS_IDEAL\model\IDEAL_resnet50')
        model.save_model(x,r'/mmfs1/scratch/hpc/00/zhangz65/Code/IDEAL/code/update_on_cifar100/models/IDEAL_vgg16_'+str(each_shuffle)+'_model')


        print ("###################### Validation ####################")

        TestData = {}

        TestData ['IDEALParms'] =  x['IDEALParms']
        TestData ['Images'] = test_labels 
        TestData ['Features'] = test_features
        TestData ['Labels'] = test_labels

        start = time.time()
        print(TestData['Features'].shape)
        # IDEAL Predict
        pred= model.predict(TestData)
        end = time.time()
        print("Validation Time: ",round(end - start,2), "seconds")
        print(pred['EstLabs'].shape)


        # IDEAL Results
        print ("###################### Results ####################")

        model.results(pred['EstLabs'],test_labels)
        
        acc.append(accuracy_score(test_labels, pred['EstLabs']))
        
    acc = np.array(acc)
    print("Accuracy: ", acc)
    print("Average accuracy: ", np.mean(acc))
    print("Average accuracy std: ", np.std(acc))
    
    


