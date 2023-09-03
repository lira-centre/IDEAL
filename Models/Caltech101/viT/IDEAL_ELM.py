import torch
import os
from torchvision.models.feature_extraction import create_feature_extractor
from torchvision.models.feature_extraction import get_graph_node_names
import pickle
import numpy as np
from torchvision.models import vit_b_16, ViT_B_16_Weights
from scipy.spatial.distance import cdist
from sklearn.metrics import confusion_matrix
from scipy.special import softmax
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, cohen_kappa_score, confusion_matrix
# import IDEAL_optimization as IDEAL
import time
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd

import os.path
from typing import Any, Callable, List, Optional, Tuple, Union

from PIL import Image

from torchvision.datasets.utils import download_and_extract_archive, verify_str_arg
from torchvision.datasets.vision import VisionDataset


class Caltech101(VisionDataset):
    """`Caltech 101 <https://data.caltech.edu/records/20086>`_ Dataset.

    .. warning::

        This class needs `scipy <https://docs.scipy.org/doc/>`_ to load target files from `.mat` format.

    Args:
        root (string): Root directory of dataset where directory
            ``caltech101`` exists or will be saved to if download is set to True.
        target_type (string or list, optional): Type of target to use, ``category`` or
            ``annotation``. Can also be a list to output a tuple with all specified
            target types.  ``category`` represents the target class, and
            ``annotation`` is a list of points from a hand-generated outline.
            Defaults to ``category``.
        transform (callable, optional): A function/transform that takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
    """

    def __init__(
        self,
        root: str,
        target_type: Union[List[str], str] = "category",
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
    ) -> None:
        super().__init__(os.path.join(root, "caltech101"), transform=transform, target_transform=target_transform)
        os.makedirs(self.root, exist_ok=True)
        if isinstance(target_type, str):
            target_type = [target_type]
        self.target_type = [verify_str_arg(t, "target_type", ("category", "annotation")) for t in target_type]

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError("Dataset not found or corrupted. You can use download=True to download it")

        self.categories = sorted(os.listdir(os.path.join(self.root, "101_ObjectCategories")))
        self.categories.remove("BACKGROUND_Google")  # this is not a real class

        # For some reason, the category names in "101_ObjectCategories" and
        # "Annotations" do not always match. This is a manual map between the
        # two. Defaults to using same name, since most names are fine.
        name_map = {
            "Faces": "Faces_2",
            "Faces_easy": "Faces_3",
            "Motorbikes": "Motorbikes_16",
            "airplanes": "Airplanes_Side_2",
        }
        self.annotation_categories = list(map(lambda x: name_map[x] if x in name_map else x, self.categories))

        self.index: List[int] = []
        self.y = []
        for (i, c) in enumerate(self.categories):
            n = len(os.listdir(os.path.join(self.root, "101_ObjectCategories", c)))
            self.index.extend(range(1, n + 1))
            self.y.extend(n * [i])

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where the type of target specified by target_type.
        """
        import scipy.io

        img = Image.open(
            os.path.join(
                self.root,
                "101_ObjectCategories",
                self.categories[self.y[index]],
                f"image_{self.index[index]:04d}.jpg",
            )
        ).convert("RGB")

        target: Any = []
        for t in self.target_type:
            if t == "category":
                target.append(self.y[index])
            elif t == "annotation":
                data = scipy.io.loadmat(
                    os.path.join(
                        self.root,
                        "Annotations",
                        self.annotation_categories[self.y[index]],
                        f"annotation_{self.index[index]:04d}.mat",
                    )
                )
                target.append(data["obj_contour"])
        target = tuple(target) if len(target) > 1 else target[0]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target


    def _check_integrity(self) -> bool:
        # can be more robust and check hash of files
        return os.path.exists(os.path.join(self.root, "101_ObjectCategories"))

    def __len__(self) -> int:
        return len(self.index)

    def download(self) -> None:
        if self._check_integrity():
            print("Files already downloaded and verified")
            return

        download_and_extract_archive(
            "https://drive.google.com/file/d/137RyRjvTBkBiIfeYBNZBtViDHQ6_Ewsp",
            self.root,
            filename="101_ObjectCategories.tar.gz",
            md5="b224c7392d521a49829488ab0f1120d9",
        )
        download_and_extract_archive(
            "https://drive.google.com/file/d/175kQy3UsZ0wUEHZjqkUDdNVssr7bgh_m",
            self.root,
            filename="Annotations.tar",
            md5="6f83eeb1f24d99cab4eb377263132c91",
        )

    def extra_repr(self) -> str:
        return "Target type: {target_type}".format(**self.__dict__)


def ELMean(dataPts, radius):
# INPUT:
# ------
# dataPts        - input data, (numPts x numDim)
# radius         - is the bandwidth parameter (scalar)

# OUTPUT:
# -------
# clustCent      - is locations of cluster centers (numClust x numDim)
# data2cluster   - for every data point which cluster it belongs to (1 x numPts)
    global c, clustCent, S, variance, sumPts, sumSqPts, data2cluster

    c = 0 #number of clusters
    clustCent = np.empty((0,0)) #cluster centres
    S = np.empty((0,0))  #support of a cluster
    variance = np.empty((0,0)) #variance of a cluster
    data2cluster = np.empty((0,0)) #cluster number of each sample
    sumPts = np.empty((0,0))       #sum of all samples in a cluster
    sumSqPts = np.empty((0,0))     #sum of squares of all samples in a cluster

# --Initialization--

    numPts,numDim = dataPts.shape
    # Read the first sample
    i = 0
    X = dataPts[i,:].reshape(1,-1) #First sample
    
    c = 1 #number of clusters
    clustCent = np.append(clustCent,X).reshape(1,-1) #cluster centres
    S = np.append(S,1)  #support of a cluster
    variance = np.append(variance,np.zeros((1,numDim))).reshape(1,-1)  #variance of a cluster
    sumSqPts = np.append(sumSqPts,X**2) .reshape(1,-1) 
    sumPts = np.append(sumPts,X).reshape(1,-1)
    data2cluster = np.append(data2cluster,c).reshape(1,-1) #cluster number of each sample

# --Repeat the following steps until there are samples--

    for i in range(1,numPts):
        X = dataPts[i,:].reshape(1,numDim) #Read the next sample
        distance = cdist(X,clustCent,'euclidean')
        Index = np.empty((0,0),dtype=int)
        for j in range(0,clustCent.shape[0]):
            # if distance[j] < np.max(np.linalg.norm(variance[j,:])**2,radius**2)+radius**2:
            if (distance[0,j]**2 < radius**2+radius**2).any():
                Index = np.append(Index,j)
        
        if Index.shape[0] != 0:
            minDist,minInd = distance[0,Index].min(),distance[0,Index].argmin()
            inInd = Index[minInd]

        neighbourAviable = 0 #indicate if a neighbouring center is available

        # Consider X belongs to nearest cluster centers and u[date mean
        if Index.size != 0:
            neighbourAviable = 1
            totalCount = S[inInd] + 1
            sumPts[inInd,:] = sumPts[inInd,:] + X
            sumSqPts[inInd,:] = sumSqPts[inInd,:] + X**2

            # Mean of X and nearest Cluster Centre
            
            meanNew = (S[inInd]*clustCent[inInd,:] + X)/totalCount

        else:
            meanNew = X

        # Merge this new mean with the closest centre

        if neighbourAviable == 1:

        # update the variance while merging
            clustCent[inInd,:] = meanNew
            variance[inInd,:] = (sumSqPts[inInd,:] +totalCount*meanNew**2 - 2*meanNew*sumPts[inInd,:])/(totalCount-1)
            S[inInd] = S[inInd] + 1
            data2cluster = np.append(data2cluster,inInd+1) #cluster number of each sample

        # find the distance between the new mean and all other cluster centers
            while True:
                mergeInd = np.empty((0,0),dtype=int)
                distToCent = np.empty((0,0)).reshape(1,-1)

                for j in range(0,c):
                    distToCent = np.append(distToCent,np.linalg.norm(meanNew-clustCent[j,:]))

                # find the centres foe which the distance < sum of variances

                for j in range(0,c):

                    # dist1 = np.linalg.norm(variance[j,:])
                    # dist2 = np.linalg.norm(variance[inInd,:])
                    # dist = max(dist1,dist2)

                    if ((distToCent[j])**2) < (radius**2+radius**2):
                        if j != inInd:
                            mergeInd = np.append(mergeInd,j)
                
                if mergeInd.size != 0:
                    # merge the cluster centres
                    minDist,minInd = distToCent[mergeInd].min(),distToCent[mergeInd].argmin()
                    mInd = mergeInd[minInd]
                    inInd = clusterMerge(mInd,inInd)
                    meanNew = clustCent[inInd,:]
                else:
                    break


        else:
            # add a new cluster centre
            c = c + 1
            clustCent = np.append(clustCent,meanNew,axis=0)
            data2cluster = np.append(data2cluster,c) #cluster number of each sample
            S = np.append(S,1)  #support of a cluster
            variance = np.append(variance,np.zeros((1,numDim)),axis=0)  #variance of a cluster
            sumSqPts = np.append(sumSqPts,X**2,axis=0)
            sumPts = np.append(sumPts,X,axis=0)

    return clustCent,data2cluster,variance
    


def clusterMerge(mInd,inInd):


# merge 'new mean' and nearest cluster centre that is close enough 
# finally after merging the smallest cluster number remains
# for example, if clusters 2 and 4 are merged, the cluster number of final merged cluster will be 2.

    global c, clustCent, S, variance, sumPts, sumSqPts, data2cluster

    minInd = np.min([mInd,inInd])
    mergeCent = clustCent

    totalCount = S[mInd] + S[inInd]
    meanCent = (S[mInd]*clustCent[mInd,:] + S[inInd]*clustCent[inInd,:])/totalCount

    mergeCent[minInd,:] = meanCent

    # update the variance while merging

    mergeVar = variance # save old variance

    variance = np.empty((0,0)) # new variance

    # print((sumSqPts[mInd,:] + sumSqPts[inInd,:]))
    # print(((sumSqPts[mInd,:] + sumSqPts[inInd,:]) - (totalCount*meanCent**2) - 2*meanCent*(sumPts[mInd,:]+ sumPts[inInd,:])))
    # print(totalCount*meanCent**2)
    # print(2*meanCent*sumPts[mInd,:])
    # print(sumPts[inInd,:])


    mergeVar[minInd,:] = ((sumSqPts[mInd,:] + sumSqPts[inInd,:]) - (totalCount*meanCent**2) - 2*meanCent*(sumPts[mInd,:]+ sumPts[inInd,:]))/(totalCount-1)


    # update the support while merging

    mergeCount = S
    S = np.empty((0,0))
    mergeCount[minInd] = totalCount

    # As cluster centres are merged, the number of cluster is updated.
    # The clustCent matrix is modified after merging clusters
    if (minInd == mInd):
        # Set the value of cluster center matrix with high index to NAN
        mergeCent[inInd,:] = np.nan
    else:
        mergeCent[mInd,:] = np.nan

    clustCent = np.empty((0,0))

    

    for i in range(0,mergeCent.shape[0]):
        if np.isnan(mergeCent[i,:]).any() == True:
            clustCent = np.delete(mergeCent,i,0)
            variance = np.delete(mergeVar,i,0)
            S = np.delete(mergeCount,i,0)
        


    # update the data2cluster matrix to reflect the new cluster numbers

    nanCount = 0
    for j in range(0,mergeCent.shape[0]):
        Ind = np.asarray(np.where(data2cluster == j+1))
        if np.isnan(mergeCent[j,:]).any() == True:
            nanCount = nanCount + 1
            for p in range(0,Ind.shape[0]):
                data2cluster[Ind[p]] = minInd+1
        else:
            if nanCount > 0 :
                for p in range(0,Ind.shape[0]):
                    data2cluster[Ind[p]] = j - nanCount

    #update the number of clusters
    c = clustCent.shape[0]
            

    return minInd

def unpickle(file):
    '''Load byte data from file'''
    with open(file, 'rb') as f:
        data = pickle.load(f,encoding='latin1')
    return data




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
    def train(self, Input):

        Images = Input['Images']
        Features = Input['Features']
        Labels = Input['Labels']
        bandwidth = Input['bandwidth']
        CN = max(Labels)
        Prototypes = self.PrototypesIdentification(Images,Features,Labels,CN,bandwidth)
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

    def PrototypesIdentification(self, Image,GlobalFeature,LABEL,CL,bandwidth):
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
            
            Prototypes[i] ,data2cluster,variance = ELMean(data[i],bandwidth)
        
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
    weights = ViT_B_16_Weights.DEFAULT
    model = vit_b_16(weights=weights)
    model.eval()
    train_nodes, eval_nodes = get_graph_node_names(model)
    
    
    model = model.to(device)
    
    
   
    dataset = Caltech101(
        root= './data', download=True, transform=weights.transforms())

    print(len(dataset))
    generator = torch.Generator().manual_seed(42)
    
    lengths = [6941,1736]
    trainset, testset = torch.utils.data.random_split(dataset, lengths, generator=generator)
    
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=200, shuffle=False, num_workers=4)

    testloader = torch.utils.data.DataLoader(
        testset, batch_size=200, shuffle=False, num_workers=4)
    
    
    
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
    
    np_train_features_2 = np.array(train_features[-1]).reshape((-1,768))
    np_train_features_1 = np.array(train_features[:-1]).reshape((-1,768))

    train_features = np.concatenate((np_train_features_1,np_train_features_2),axis=0)
    


    np_test_features_2 = np.array(test_features[-1]).reshape((-1,768))
    np_test_features_1 = np.array(test_features[:-1]).reshape((-1,768))
    
    
    test_features = np.concatenate((np_test_features_1,np_test_features_2),axis=0)
    
    train_labels_2 = np.array(train_labels[-1]).reshape((-1,1))
    train_labels_1 = np.array(train_labels[:-1]).reshape((-1,1))
    
    train_labels = np.concatenate((train_labels_1,train_labels_2),axis=0)
    
    test_labels_2 = np.array(test_labels[-1]).reshape((-1,1))
    test_labels_1 = np.array(test_labels[:-1]).reshape((-1,1))
    
    test_labels = np.concatenate((test_labels_1,test_labels_2),axis=0)

    
    
    train_features_labels = np.concatenate((train_features, train_labels.reshape((-1,1))), axis=1)
    test_features_labels = np.concatenate((test_features, test_labels), axis=1)
    
    # print(train_features_labels[0,:])
    
    pd_train_features_labels = pd.DataFrame(train_features_labels)
    pd_test_features_labels = pd.DataFrame(test_features_labels)
    
    # pd_train_features_labels = pd.DataFrame(train_features_labels)
    # pd_test_features_labels = pd.DataFrame(test_features_labels)
    
    # pd_train_features_labels.to_csv('/mmfs1/scratch/hpc/00/zhangz65/Code/IDEAL/code/cifa10/training_on_cifa10/model/Resnet101_train_features_labels.csv',index=False,header=False)
    # pd_test_features_labels.to_csv('/mmfs1/scratch/hpc/00/zhangz65/Code/IDEAL/code/cifa10/training_on_cifa10/model/Resnet101_test_features_labels.csv',index=False,header=False)
    
    
    
    acc = []
    
    for each_shuffle in range(5):
        
        # Shuffle the training data
        np.random.seed(each_shuffle)
        np.random.shuffle(train_features_labels)
    
   
        

        
        train_features = train_features_labels[:,:-1]
        
        train_labels = train_features_labels[:,-1].astype(int)
        
        # Data Input (dict)a

        Input = {}

        Input['Images'] = train_labels
        Input['Features'] = train_features
        Input['Labels'] = train_labels
        Input['bandwidth'] = 13

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
            class_prototypes = len(Prototypes[i])
            total_prototypes = total_prototypes + class_prototypes
            print("Class", i+1, ":", class_prototypes)

        print("Total   :", total_prototypes)
        print("Prototypes as % of the Training Data Samples:", total_prototypes/len(train_features)*100, "%")


        # print ("###################### Visual Prototypes ####################")

        # Prototypes = x['IDEALParms']['Parameters']
        # total_prototypes = 0

        # for i in range(len(Prototypes)):
        #     class_prototypes = len(Prototypes[i])
        #     total_prototypes = total_prototypes + class_prototypes
        #     print("Number of prototypes Class", i+1, ":", class_prototypes)
        #     print("Prototypes : ", Prototypes[i])
        #     print(" ")

        # Save IDEAL model (optional)
        # model.save_model(x,r'E:\Lancaster_PhD\PHD\Data\WORLDFLOODS\Code\NeurIPS_IDEAL\model\IDEAL_vgg16')
        # model.save_model(x,r'/mmfs1/scratch/hpc/00/zhangz65/Code/IDEAL/code/cifa10/training_on_cifa10/model/kmeans_vgg16_'+str(each_shuffle)+'_model')



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

        
        


