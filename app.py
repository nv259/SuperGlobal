import os

from flask import Flask, render_template, request, jsonify
# from flask_cors import CORS
import cv2 

# large image search
lsis: kd-tree (sklearn), lsh (lshashpy3), IndexFlatL2 (faiss)
    """DAY LA BUOC OFFLINE: LOAD DATABASE XONG ROI NHO INDEX NHA"""
    # kdtree = KDTree(collection)
    # lsh = LSHash(10, collection.shape[1], 3)
    # for i in range(len(collection)):
    #     lsh.index(collection[i], extra_data=str(i))
    # index_flat = faiss.IndexFlatL2(collection.shape[1])
    # if faiss.get_num_gpus() > 0:
    #     res = faiss.StandardGpuResources()
    #     index_flat = faiss.index_cpu_to_gpu(res, 0, index_flat)
    # index_flat.train(collection) 
    # index_flat.add(collection)
    
    # # find `k` most similar images to query image
    # start_time = time.time()
    
    
    
    """ DAY LA BUOC ONLINE: RETRIEVAL NHA LECHAUANH"""
    # if args.lsis is None:
    #     dists = np.linalg.norm(collection - feature, axis=1) 
    #     ids = np.argsort(dists)[:k]
    # if args.lsis == 'kdtree':
    #     dists, ids = kdtree.query(feature, k=k)
    #     ids = ids[0]
    # if args.lsis == 'lsh':
    #     neighbors = lsh.query(feature.flatten(), num_results=k, distance_func='euclidean') 
    #     ids = [int(neighbor[0][1]) for neighbor in neighbors]
    #     dists = [neighbor[1] for neighbor in neighbors]
    # if args.lsis == 'faiss':
        # dists, ids = index_flat.search(feature, k) 
        # ids = ids[0]



# create flask instance
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = "static"


# Load model (checkpoint or else)
model = ...

# create FeatureExtractor class (optional) / model xu ly 1 cai 
# import feature_extract or else / load query len batch (bs = 1)

# Load database (db.npy)
extracted_features = ...

# main route
@app.route('/', methods=['GET', 'POST'])
def main():
    if request.method == 'POST':
        image = request.files['file']
        
        if image:
            # Save image
            path_to_save = os.path.join(app.config['UPLOAD_FOLDER'], image.filename)
            image.save(path_to_save)
            
            # convert to cv2
            query_image = cv2.imread(path_to_save)
            
            # Extract feature for query_image
            # query_feature = feature_extract(query_image, )
            
            # Compute similarity, dist
            
            
            # Sort ranked list + slice for k elements (optional)
            ranked_list = ['./static/1.jpg', './static/2.jpg', './static/3.jpg', './static/4.jpg', './static/5.jpg']
            
        return render_template("index.html", msg="retrieve successfully!", ranked_list=ranked_list)
    
    else: 
        return render_template('index.html')


if __name__ == "__main__":
    app.run(host='0.0.0.0', port='2503')


# SAU KHI XONG THI VIET REPORT NHA CHAU ANH 