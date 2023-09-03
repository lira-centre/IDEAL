import torch
import os
from torchvision.models.feature_extraction import create_feature_extractor
from torchvision.models.feature_extraction import get_graph_node_names
from torchvision.datasets import CIFAR100
import pickle
import numpy as np
import matplotlib.pyplot as plt
import math
from torchvision.models import resnet50, ResNet50_Weights
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
    

# Load the pre-trained model and perform feature extraction

def feature_extraction(img,model):
    
    img = img.to(device)
    
        #Perform transformation

    feature_extractor = create_feature_extractor(
        model, return_nodes=['flatten'])

    with torch.no_grad():
        out = feature_extractor(img)
        
    return out['flatten']
    
    
    
class IDEALClassifier:
    
    def __init__(self):
        self.trained_model = None
    
    
    def train(self, Input, random_state):

        Images = Input['Images']
        Features = Input['Features']
        Labels = Input['Labels']
        CN = np.unique(Labels)
        Prototypes = self.PrototypesIdentification(Images,Features,Labels,CN,random_state)
        Output = {}
        Output['IDEALParms'] = {}
        Output['IDEALParms']['Parameters'] = Prototypes
        MemberLabels = {}
        for i in CN:
            MemberLabels[i]=Input['Labels'][Input['Labels']==i] 
        Output['IDEALParms']['CurrentNumberofClass']=len(CN)
        Output['IDEALParms']['OriginalNumberofClass']=len(CN)
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
    
    def update_prototypes(self, Input, random_state, n_clusters=100):
        
        Images = Input['Images']
        Features = Input['Features']
        Labels = Input['Labels']
        CN = np.unique(Labels)


        if CN.min() > self.trained_model['IDEALParms']['CurrentNumberofClass'] - 1:
            new_Prototypes = self.PrototypesIdentification(Images, Features, Labels, CN, random_state)
            updated_Prototypes = {}
            for i in range(self.trained_model['IDEALParms']['CurrentNumberofClass']):
                updated_Prototypes[i] = self.trained_model['IDEALParms']['Parameters'][i]

            for i in range(self.trained_model['IDEALParms']['CurrentNumberofClass'], self.trained_model['IDEALParms']['CurrentNumberofClass']+len(CN)):
                updated_Prototypes[i] = new_Prototypes[i]

            self.trained_model['IDEALParms']['Parameters'] = updated_Prototypes
            self.trained_model['IDEALParms']['CurrentNumberofClass'] = self.trained_model['IDEALParms']['CurrentNumberofClass'] + len(CN)
        else:
            print("No new classes detected. Use the train method for updating the existing classes.")
            

    def PrototypesIdentification(self, Image,GlobalFeature,LABEL,CL,random_state):
        data = {}
        image = {}
        label = {}
        Prototypes = {}
        for i in CL:
            seq = np.argwhere(LABEL==i)
            data[i]=GlobalFeature[seq]
            image[i] = {}
            for j in range(0, len(seq)):
                image[i][j] = Image[seq[j][0]]
            label[i] = np.ones((len(seq),1))*i
        for i in CL:
            data[i] = np.squeeze(data[i],axis=1)
            num_cluster = int(data[i].shape[0]*0.1)
            kmeans = KMeans(n_clusters=num_cluster, random_state=random_state).fit(data[i])
            Prototypes[i]  = kmeans.cluster_centers_
        
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

    
    # os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"   
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    train_features, test_features = [], []
    train_labels, test_labels = [], []
    

   
    # tmp_dir = os.getenv('TMPDIR')
    # os.environ['TORCH_HOME'] = tmp_dir #setting the environment variable
    # Load the pretrained model
    weights = ResNet50_Weights.DEFAULT
    model = resnet50(weights=weights)
    model.eval()
    model = model.to(device)
    
    
    
    
   
    dataset = Caltech101(
        root= './data', download=True, transform=weights.transforms())

    print(len(dataset))

    
    lengths = [6941,1736]
    trainset, testset = torch.utils.data.random_split(dataset, lengths, generator=torch.Generator().manual_seed(42))
    
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
    
    np_train_features_2 = np.array(train_features[-1]).reshape((-1,2048))
    np_train_features_1 = np.array(train_features[:-1]).reshape((-1,2048))

    train_features = np.concatenate((np_train_features_1,np_train_features_2),axis=0)
    
    print(train_features.shape)

    np_test_features_2 = np.array(test_features[-1]).reshape((-1,2048))
    np_test_features_1 = np.array(test_features[:-1]).reshape((-1,2048))
    
    
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
    
    # pd_train_features_labels.to_csv('/mmfs1/scratch/hpc/00/zhangz65/Code/IDEAL/code/Caltech101/viT-L/models/vit_L_train_features_labels.csv',index=False,header=False)
    # pd_test_features_labels.to_csv('/mmfs1/scratch/hpc/00/zhangz65/Code/IDEAL/code/Caltech101/viT-L/models/vit_L_test_features_labels.csv',index=False,header=False)
    
    
    acc = []
    
    train_features = train_features_labels[:,:-1]

    train_labels = train_features_labels[:,-1].astype(int)

    test_features = test_features_labels[:,:-1]

    test_labels = test_features_labels[:,-1].astype(int)
    
    total_accuracy = []
    training_times = []
    testing_times = []
    for run in range(10):
        accuracies = []
        # Initialize the IDEALClassifier
        classifier = IDEALClassifier()

        run_training_times = []
        run_testing_times = []


        for i in range(0, len(np.unique(test_labels))-1, 10):
            
            if i != 90:
            
                train_indices = np.where((train_labels.reshape(-1) >= i) & (train_labels.reshape(-1) < i + 10))[0]
                
            elif i == 90:
                
                train_indices = np.where((train_labels.reshape(-1) >= i) & (train_labels.reshape(-1) <= i + 11))[0]
            
            
            train_input = {
                'Images': train_features[train_indices],
                'Features': train_features[train_indices],
                'Labels': train_labels[train_indices].reshape(-1)
            }

            
            print("Train labels: ", train_labels[train_indices].reshape(-1))
            # Train the classifier
            start_train = time.time()
            if i == 0:
                trained_model = classifier.train(train_input, random_state=0 + run)
            else:
                classifier.update_prototypes(train_input, random_state=0 + run)
            end_train = time.time()

            run_training_times.append(end_train - start_train)

            # Store the trained model
            classifier.trained_model = trained_model
            


            # Test the IDEALClassifier on the same classes as the training set
            
            if i == 0:
                test_indices = np.where((test_labels.reshape(-1) >= i) & (test_labels.reshape(-1) < i + 10))[0]
            else:
                if i != 90:
                    test_indices = np.append(test_indices, np.where((test_labels.reshape(-1) >= i) & (test_labels.reshape(-1) < i + 10))[0])
                elif i == 90:
                    
                    test_indices = np.append(test_indices, np.where((test_labels.reshape(-1) >= i) & (test_labels.reshape(-1) < i + 11))[0])
            
            test_input = {
                'IDEALParms': trained_model['IDEALParms'],
                'Features': test_features[test_indices],
                'Labels': test_labels[test_indices].reshape(-1)
            }
            
            print("Test labels: ", test_labels[test_indices].reshape(-1))
            start_test = time.time()
            test_output = classifier.predict(test_input)
            end_test = time.time()

            run_testing_times.append(end_test - start_test)

            # Calculate accuracy
            accuracy = accuracy_score(test_labels[test_indices].reshape(-1), test_output['EstLabs'])
            accuracies.append(accuracy)

            print(f"Accuracy for run {run + 1}, classes {i}-{i + 9}: {accuracy}")
            # print(f"Mean training time for run {run + 1}, classes {i}-{i + 0}: {np.mean(run_training_times[-1]):.2f} seconds")
            # print(f"Mean testing time for run {run + 1}, classes {i}-{i + 0}: {np.mean(run_testing_times[-1]):.2f} seconds"
        total_accuracy.append(accuracies)
        training_times.append(run_training_times)
        testing_times.append(run_testing_times)

    # Calculate mean and standard deviation of accuracies
    total_accuracy = np.array(total_accuracy)
    mean_accuracy = np.mean(total_accuracy,axis=0)
    std_accuracy = np.std(total_accuracy,axis=0)

    # Calculate mean of training and testing times

    training_times = np.array(training_times)
    testing_times = np.array(testing_times)

    mean_training_times = np.mean(training_times,axis=0)
    mean_testing_times = np.mean(testing_times,axis=0)


    print(f"\nMean accuracy: {mean_accuracy}")
    print(f"Standard deviation of accuracy: {std_accuracy}")
    print(f"Mean training time: {mean_training_times}")
    print(f"Mean testing time: {mean_testing_times}")
