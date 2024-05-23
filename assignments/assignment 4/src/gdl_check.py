from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image, ImageFile
import logging
import os

"""
Script for sanity-checking that up to 25 % of the pages in the GDL-newspaper does not contain faces, which the model indicates. 
It analyses the images the same way as the main script but is limited to pages in GDL 1790 to 1880 and logs the results for each image in this subset.
"""

# allow the analysis to continue even if some of the images might be corrupted or otherwise unable to be analysed
ImageFile.LOAD_TRUNCATED_IMAGES = True

# setting up the logging. Going to log each individual image and state whether it saw a image or not
log_file_path = os.path.join("out", "gdl_detection_1790_1880.log") # setting up the output folder
logging.basicConfig(filename=log_file_path, level=logging.INFO, format='%(message)s')

# Loading the pre-trained ResNet model
resnet = InceptionResnetV1(pretrained="casia-webface").eval()
# Set up MTCNN for face detection
mtcnn = MTCNN()

def detect_faces(image_path, mtcnn, resnet):
    """detects the faces in the newspapers using mtcnn and processes them using resnet."""
    img = Image.open(image_path) 
    aligned = mtcnn(img)  # detect and align faces in the page
    
    if aligned is not None:
        if aligned.ndimension() == 3:  # if only a single face is detected it will result in a 3d tensor. resnet expects a 4d tensor
            aligned = aligned.unsqueeze(0)  # therefore, reshaping to a 4d tensor if that is the case by unsqueeze
        embeddings = resnet(aligned).detach()  # process the aligned face tensors
        num_faces = len(embeddings)  # return the number of faces detected
        if num_faces > 0: #logging if detected face for that page
            logging.info(f"Detected {num_faces} face(s) in {image_path}")
            return True
    logging.info(f"No faces detected in {image_path}") # logging if it does not detect a face
    return False

def get_year(filename):
    """extract the year from the filename of the newspapers."""
    return int(filename.split("-")[1])

def process_newspapers(input_folder, newspaper):
    """process the newspaper folder to detect faces and log results up to 1890."""
    newspaper_folder = os.path.join(input_folder, newspaper)  # making the path to the newspaper folder
    files = sorted(os.listdir(newspaper_folder), key=get_year)  # list and sort all the pages by year

    for filename in files:
        year = get_year(filename)  # extract the year from the filename
        if year >= 1890:  # process only years before 1890
            continue

        image_path = os.path.join(newspaper_folder, filename)  # full path for the specific newspaper page
        detect_faces(image_path, mtcnn, resnet)  # detects faces in the page using defined function

if __name__ == "__main__":
    # define the folders
    input_folder = "in"
    newspapers = ["GDL"] #only for gdl now

    # process newspapers (main pipeline)
    process_newspapers(input_folder, newspapers[0])
