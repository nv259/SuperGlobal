import os

from flask import Flask, render_template, request, jsonify
# from flask_cors import CORS
import cv2 


# create flask instance
app = Flask(__name__)

# Apply flask CORS
# CORS(app) 
# app.config['CORS_HEADERS'] = 'Content-Type'
app.config['UPLOAD_FOLDER'] = "static"

# Load model (checkpoint or else)
model = ...

# create FeatureExtractor class (optional)
# import feature_extract or else

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
            
            # Compute similarity
            
            
            # Sort ranked list + slice for k elements (optional)
            ranked_list = ['./static/1.jpg', './static/2.jpg', './static/3.jpg', './static/4.jpg', './static/5.jpg']
            
        return render_template("index.html", msg="retrieve successfully!", ranked_list=ranked_list)
        
    return render_template('index.html')


if __name__ == "__main__":
    app.run(host='0.0.0.0', port='2503')

# # search route
# @app.route('/search', methods=['POST'])
# def search():

#     if request.method == "POST":

#         RESULTS_ARRAY = []
#         # get url
#         image_url = request.files('file')

