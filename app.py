import os
import torch
import cv2 # from flask_cors import CORS
import core.checkpoint as checkpoint
import core.transforms as transforms
import test.test_loader as loader
import torch.nn.functional as F
import numpy as np
import pickle as pkl

from flask import Flask, render_template, request, jsonify
from model.CVNet_Rerank_model import CVNet_Rerank
from config import cfg
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial import KDTree
from faiss import IndexFlatL2
import faiss
from lshashpy3 import LSHash


_MEAN = [0.406, 0.456, 0.485]
_SD = [0.225, 0.224, 0.229]


# create flask instance
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = "static"


# Load model (checkpoint or else)
print('LOAD MODEL')
model = CVNet_Rerank(cfg.MODEL.DEPTH, cfg.MODEL.HEADS.REDUCTION_DIM, cfg.SupG.relup) # 50, 2048, relup=True 
print("push to gpu (if available)")
model = model.to(device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
print("load checkpoint")
checkpoint.load_checkpoint('./weights/CVPR2022_CVNet_R50.pyth', model) # cfg.TEST.WEIGHTS, model)


@torch.no_grad()
# create FeatureExtractor class (optional) / model xu ly 1 cai 
def extract_feature(model, image, gemp, rgem, sgem, scale_list):
    image = image.astype(np.float32, copy=False)    
    image = image.transpose([2, 0, 1])
    image = image / 255.0
    image = transforms.color_norm(image, _MEAN, _SD) 
     
    image = torch.tensor(image)
    image = image.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    image = image.unsqueeze(dim=0)

    with torch.no_grad():
        desc = model.extract_global_descriptor(image, gemp, rgem, sgem, scale_list)
    if len(desc.shape) == 1:
        desc = desc.unsqueeze_(0)
    desc = F.normalize(desc, p=2, dim=1)
    desc = desc.cpu().numpy()
    
    print(desc.shape)
    
    return desc

# Load database (db.npy)
print("LOAD DATABASE")
db_rparis6k = np.load("./db_rparis6k.npy")
db_roxford6k = np.load("./db_roxford6k.npy")
db = np.concatenate((db_rparis6k, db_roxford6k), axis=0)
with open("./gnd_rparis6k.pkl", 'rb') as file:
    gnd_rparis6k = pkl.load(file)
with open("./gnd_roxford5k.pkl", 'rb') as file:
    gnd_roxford5k = pkl.load(file)
gnd_rparis6k = gnd_rparis6k['imlist']
gnd_roxford5k = gnd_roxford5k['imlist']
gnd = gnd_rparis6k + gnd_roxford5k
def index_construction(db, lsis):
    if lsis == 'kdtree':
        kdtree = KDTree(db)
        return kdtree
    if lsis == 'lsh':
        lsh = LSHash(10, db.shape[1], 3)
        for i in range(len(db)):
            lsh.index(db[i], extra_data=str(i))
        return lsh
    if lsis == 'faiss':
        index_flat = IndexFlatL2(db.shape[1])
        
        if faiss.get_num_gpus() > 0:
            res = faiss.StandardGpuResources()
            index_flat = faiss.index_cpu_to_gpu(res, 0, index_flat)
        index_flat.train(db) 
        index_flat.add(db)
        
        return index_flat

print("INDEX DATABASE")
indexed_db = index_construction(db, cfg.LSIS)


# main route
@app.route('/', methods=['GET', 'POST'])
def main():
    if request.method == 'POST':
        image = request.files['file']
        
        if image:
            # Save image
            path_to_save = os.path.join(app.config['UPLOAD_FOLDER'], "query_image.jpg")
            image.save(path_to_save)
            
            # convert to cv2
            query_image_ = cv2.imread(path_to_save)
            query_image = cv2.imread(path_to_save)
            text = "query_image"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1.5
            font_thickness = 3
            text_color = (167, 199, 231)  
            text_org = (10, query_image_.shape[0] - 10)  # bottom-left corner
            cv2.putText(query_image_, text, text_org, font, font_scale, text_color, font_thickness)
            cv2.imwrite(path_to_save, query_image_)
            
            # Extract feature for query_image
            Q = extract_feature(model, query_image, cfg.SupG.gemp, cfg.SupG.rgem, cfg.SupG.sgem, scale_list=3)# cfg.SupG.gemp, cfg.SupG.rgem, cfg.SupG.sgem, cfg.TEST.SCALE_LIST) # gemp, rgem, sgem, scale_list)

            k = 11
        
            if cfg.LSIS is None:
                dists = np.linalg.norm(indexed_db - Q, axis=1) 
                ids = np.argsort(dists)[:k]
            if cfg.LSIS == 'kdtree':
                dists, ids = indexed_db.query(Q, k)
                ids = ids[0]
            if cfg.LSIS == 'lsh':
                neighbors = indexed_db.query(Q.flatten(), num_results=k, distance_func='euclidean') 
                ids = [int(neighbor[0][1]) for neighbor in neighbors]
                dists = [neighbor[1] for neighbor in neighbors]
                # print(ids)    
            if cfg.LSIS == 'faiss':
                pass
                # dists, ids = indexed_db.search(Q, k) 
                # ids = ids[0]
            
            ranked_list = []
            
            for index in ids:
                ranked_list.append(gnd[index] + '.jpg')
            
        print(ranked_list)
        return render_template("index.html", msg="Retrieve successfully!", ranked_list=ranked_list)
    
    else: 
        return render_template('index.html')


if __name__ == "__main__":
    app.run(host='0.0.0.0', port='2503')

# SAU KHI XONG THI VIET REPORT NHA CHAU ANH 