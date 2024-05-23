import matplotlib.pyplot as plt
from PIL import Image
import os

# paths
log_file_path = 'out/gdl_detection_1790_1880.log'  # path to log file
input_folder = 'in/GDL'  # path to gdl pages

# load in the log file and extract image paths with detected faces
detected_faces = []  # list to store paths of images where faces were detected
with open(log_file_path, 'r') as log_file:
    lines = log_file.readlines()  # reads all the lines from the log file
    for line in lines:
        if 'Detected' in line:  # check if the line contains "Detected" and if it does, it will proceed
            parts = line.strip().split(' ')  # split the line into parts based on space
            image_path = parts[-1]  # take the last part of the split, which is the file path to the image
            detected_faces.append(image_path)  # add the image path to the list

# organize images by decade
decade_images = {}  # dictionary to store images organized by decade
for image_path in detected_faces:
    filename = os.path.basename(image_path)  # filename from the image path
    year = int(filename.split('-')[1])
    decade = (year // 10) * 10
    if decade not in decade_images:  # if it has not organized a picture from that decade yet, it will set up a list for it
        decade_images[decade] = []
    decade_images[decade].append(image_path)  # add the image path to the list for the decade

# define the output folder for the plots
output_folder = '../out'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)  # create the output directory if it doesn't exist

# flatten the decade images dictionary into a list of tuples
image_list = [(decade, img) for decade, images in decade_images.items() for img in images]

# calculate number of plots needed
images_per_row = 11  # there is 22 images, so making it 11x2
num_rows = 2

# plot images across multiple rows
fig, axes = plt.subplots(num_rows, images_per_row, figsize=(20, 5 * num_rows))
fig.suptitle('Faces Detected in GDL Newspapers from 1790-1880')


for idx, (decade, img_path) in enumerate(image_list):
    row = idx // images_per_row
    col = idx % images_per_row
    ax = axes[row][col]
    ax.set_title(f'{decade}s')
    ax.axis('off')
    img = Image.open(img_path)
    ax.imshow(img, aspect='auto')

plt.tight_layout()
plt.savefig(os.path.join(output_folder, 'sanity_check_for_GDL_1790_1880.png'))
plt.close()
