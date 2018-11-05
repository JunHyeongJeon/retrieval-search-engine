import os
import h5py
import torch
import numpy as np

#features 경로 
def load_features():
    features_path=os.path.join(os.path.dirname(os.path.abspath(__file__)),'..', '..','..','..', 'features')
    # print(features_path)
    features=dict()
    # print("features_dict : ", features)
    datasets_location = os.listdir(features_path)
    print("datasets_location : ",datasets_location)

    for datasets in datasets_location:
        feature_vect=dict()
        features_location = os.listdir(os.path.join(features_path,datasets))
        print("features_location :", features_location)
        for image_feature in features_location:
            print("image_feature",image_feature)
            if image_feature == "MAC":
                feat = []
                name = []
                feature_files=os.listdir(os.path.join(features_path,datasets,image_feature))
                for f in feature_files:
                    feature_file=h5py.File(os.path.join(os.path.join(features_path,datasets,image_feature,f)))
                    feat.append(feature_file[image_feature][()])
                    name.append(feature_file['names'][()])
                    feature_file.close()
                feat_np=np.concatenate(feat,axis=0)
                name_np=np.concatenate(name,axis=0)
                feature_vect[image_feature]=torch.tensor(feat_np.reshape([feat_np.shape[0],-1])).cuda()
                feature_vect['names']=name_np

        features[datasets]=feature_vect

    # print("myfeatures: ", features.keys())
    # print(features['photo'].keys())
    return features

