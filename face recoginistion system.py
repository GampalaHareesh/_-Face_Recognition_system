import cv2
import numpy as np
from sklearn import preprocessing, svm
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image

# Load pre-trained FaceNet model
model = InceptionResnetV1(pretrained='vggface2').eval()

# Load an image containing a face
image = cv2.imread('path_to_image.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image = Image.fromarray(image)

# Initialize MTCNN for face detection
mtcnn = MTCNN()

# Detect faces and extract embeddings
face = mtcnn(image)
if face is not None:
    embedding = model(face.unsqueeze(0)).detach().numpy()
    
    # Load the SVM classifier trained on face embeddings
    classifier = joblib.load('svm_classifier.pkl')
    
    # Predict the identity of the person
    predicted_label = classifier.predict(embedding)
    print("Predicted Label:", predicted_label)
else:
    print("No face detected in the image.")
