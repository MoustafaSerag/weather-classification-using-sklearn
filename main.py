import os 
from img2vec_pytorch import Img2Vec
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

import pickle

img2vec= Img2Vec()

data_dir=("Weather-classification")
train_dir=os.path.join(data_dir,"train")
val_dir=os.path.join(data_dir,"valid")

data={}

for i,dir_ in enumerate ([train_dir,val_dir]):

    features=[]
    labels=[]

    for category in os.listdir(dir_):
        for img_path in os.listdir(os.path.join(dir_,category)):
            img_path_=os.path.join(dir_,category,img_path)
            img=Image.open(img_path_)
            img_features=img2vec.get_vec(img)

            features.append(img_features)
            labels.append(category)


    data[['train_data','valid_data'][i]]=features
    data[['train_labels','valid_labels'][i]]=labels


model= RandomForestClassifier()

model.fit(data['train_data'],data['train_labels'])

y_pred=model.predict(data['valid_data'])

score = accuracy_score(y_pred,data['valid_labels'])

print(score)



with open('./model.p','wb') as f :
    pickle.dump(model,f)
    f.close()


