import torch
import os
from torchvision.models.feature_extraction import create_feature_extractor
from torchvision.models.feature_extraction import get_graph_node_names
from torchvision.datasets import CIFAR10
import pickle
import numpy as np
from torchvision.models import vit_l_16, ViT_L_16_Weights
from scipy.spatial.distance import cdist
from sklearn.metrics import confusion_matrix
from scipy.special import softmax
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, cohen_kappa_score, confusion_matrix
import time
from sklearn.cluster import KMeans
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



def unpickle(file):
    '''Load byte data from file'''
    with open(file, 'rb') as f:
        data = pickle.load(f,encoding='latin1')
    return data



def load_CIFAR10_data(data_dir):
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
            num_cluster = int(data[i].shape[0]*0.1)
            kmeans = KMeans(n_clusters=num_cluster, random_state=random_state).fit(data[i])
            Prototypes[i]  = kmeans.cluster_centers_
            Prototypes[i] = self.NearestDataPoint(Prototypes[i],data[i])
        
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
        
    def NearestDataPoint(self,Centre,data):
        
        for i in range(Centre.shape[0]):

            distance = cdist(Centre[i].reshape(1,-1),data,'euclidean')[0]
            value,position= distance.min(0),distance.argmin(0)
            Centre[i] = data[position]

        return Centre
            
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
    tmp_dir = os.getenv('TMPDIR')
    os.environ['TORCH_HOME'] = tmp_dir #setting the environment variable
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    train_features, test_features = [], []
    train_labels, test_labels = [], []
    

   

    # Load the pretrained model
    weights = ViT_L_16_Weights.IMAGENET1K_SWAG_E2E_V1
    model = vit_l_16(weights=weights)
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
        trainset, batch_size=150, shuffle=False, num_workers=4)

    testloader = torch.utils.data.DataLoader(
        testset, batch_size=150, shuffle=False, num_workers=4)
    
    
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
    
    np_train_features_2 = np.array(train_features[-1]).reshape((-1,1024))
    np_train_features_1 = np.array(train_features[:-1]).reshape((-1,1024))

    train_features = np.concatenate((np_train_features_1,np_train_features_2),axis=0)
    
    print(train_features.shape)

    np_test_features_2 = np.array(test_features[-1]).reshape((-1,1024))
    np_test_features_1 = np.array(test_features[:-1]).reshape((-1,1024))
    
    
    test_features = np.concatenate((np_test_features_1,np_test_features_2),axis=0)
    
    train_labels_2 = np.array(train_labels[-1]).reshape((-1,1))
    train_labels_1 = np.array(train_labels[:-1]).reshape((-1,1))
    
    train_labels = np.concatenate((train_labels_1,train_labels_2),axis=0)
    
    test_labels_2 = np.array(test_labels[-1]).reshape((-1,1))
    test_labels_1 = np.array(test_labels[:-1]).reshape((-1,1))
    
    test_labels = np.concatenate((test_labels_1,test_labels_2),axis=0)

    
    
    train_features_labels = np.concatenate((train_features, train_labels.reshape((-1,1))), axis=1)
    test_features_labels = np.concatenate((test_features, test_labels), axis=1)
    
    
    pd_train_features_labels = pd.DataFrame(train_features_labels)
    pd_test_features_labels = pd.DataFrame(test_features_labels)
    
    # pd_train_features_labels.to_csv('/mmfs1/scratch/hpc/00/zhangz65/Code/IDEAL/code/update_on_cifar100/models/vit_L_train_features_labels.csv',index=False,header=False)
    # pd_test_features_labels.to_csv('/mmfs1/scratch/hpc/00/zhangz65/Code/IDEAL/code/update_on_cifar100/models/vit_L_test_features_labels.csv',index=False,header=False)
    
    
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

        # for i in range(len(Prototypes)):
        #     class_prototypes = len(Prototypes[i])
        #     total_prototypes = total_prototypes + class_prototypes
        #     print("Number of prototypes Class", i+1, ":", class_prototypes)
        #     print("Prototypes : ", Prototypes[i])
        #     print(" ")

        # Save IDEAL model (optional)
        # model.save_model(x,r'E:\Lancaster_PhD\PHD\Data\WORLDFLOODS\Code\NeurIPS_IDEAL\model\IDEAL_resnet50')
        model.save_model(x,r'/mmfs1/scratch/hpc/00/zhangz65/Code/IDEAL/code/Caltech101/viT-L/models/kmeans_nearestdata_vit_L_'+str(each_shuffle)+'_model')



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

        
        


