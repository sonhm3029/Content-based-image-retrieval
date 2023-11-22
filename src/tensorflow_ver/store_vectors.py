
from keras.applications.vgg16 import VGG16,preprocess_input
from keras.preprocessing import image
from keras.models import Model

import numpy as np
import os
import pickle

from tqdm import tqdm


# Build model to vectorized data
vgg16_model = VGG16(weights="imagenet")
vgg16_model.summary()

extract_model = Model(inputs=vgg16_model.inputs, outputs=vgg16_model.get_layer("fc1").output)

# import data
def load_img(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    
    return x
# vectorized data
def vectorized(image,model): 
    result = model.predict(image)[0]   
     
    return result / np.linalg.norm(result)

# euclid distance
def euclid_dis(vector1, vector2):
    return np.sqrt(np.sum((vector1 - vector2)**2))



if __name__ == "__main__":
    try:
        
        vectors = []
        paths = []


        count = 0
        # store data to binary file for next time import
        root = "src/data/linh_dataset/6"

        for img_pth in tqdm(os.listdir(root)):
    
            img_fpath = f"{root}/{img_pth}"
            img_loaded = load_img(img_fpath)
            img_vector = vectorized(img_loaded, extract_model)
    
            vectors.append(img_vector)
            paths.append(img_fpath)
    
        print(f"Xy ly xong: {img_fpath}")
    
        vectors_file = "src/weights/vectors.pkl"
        paths_file = "src/weights/paths.pkl"
        pickle.dump(vectors, open(vectors_file, "wb"))
        pickle.dump(paths, open(paths_file, "wb"))
    
        print(f"Successfully saved to {vectors_file} and {paths_file}")
    except:
        print("Failed to save weights")




