from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image, ImageFile
import torch
import os
import pandas as pd
import matplotlib.pyplot as plt

# allow the analysis to continue even if some of the images might be corrupted or otherwise unable to be analysed
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Loading the pre-trained ResNet model
resnet = InceptionResnetV1(pretrained="casia-webface").eval()
# Set up MTCNN for face detection
mtcnn = MTCNN()

def detect_faces(image_path, mtcnn, resnet):
    """Detects the faces in the newspapers using MTCNN and processes them using ResNet."""
    img = Image.open(image_path) 
    aligned = mtcnn(img)  # detect and align faces in the page
    
    if aligned is not None:
        if aligned.ndimension() == 3:  # If only a single face is detected it will resutl in a 3d tensor. Resnet expects a 4D tensor 
            aligned = aligned.unsqueeze(0) # therefore, reshaping to a 4d tensor if that is the case by unsquueze
        embeddings = resnet(aligned).detach()  # process the aligned face tensors
        return len(embeddings)  # return the number of faces detected
    
    return 0

def get_year(filename):
    """Extract the year from the filename of the newspapers"""
    return int(filename.split("-")[1])  # splits the string into substrings divided by "-". so, eg. "GDL-1798-02-05-a-p0001" would be split into GDL, 1798, 02, and so on. Then we take the second element, which is the year and convert it to integer.

def process_newspapers(input_folder, newspapers):
    """Process each newspaper folder to detect faces and summarise results"""
    results = []  # a list to store the results

    for newspaper in newspapers:
        newspaper_folder = os.path.join(input_folder, newspaper)  # constructing the path to the newspaper folder
        face_counts = {}  # dictionary to store face counts by decade
        files = sorted(os.listdir(newspaper_folder), key=get_year)  # list and sort all the pages by year

        for filename in files:
            year = get_year(filename)  # Extract the year from the filename
            decade = (year // 10) * 10  # takes the year and divides by 10 while discarding the remainder; meaning that 1876 // 10 == 187. Then we multiply by 10 to get the decade, so 187*10 would be 1870.

            if decade not in face_counts: # processing the newspapers chronologically
                face_counts[decade] = {"pages": 0, "pages_with_faces": 0}  # start counts for the decade
                print(f"Processing {newspaper}, Decade: {decade}") # tally to keep count on the processing

            image_path = os.path.join(newspaper_folder, filename)  # full path for the specific newspaper page
            face_count = detect_faces(image_path, mtcnn, resnet)  # Detects faces in the page using defined function
            face_counts[decade]["pages"] += 1  # increases the newspaper page count for the decade, so that we can calculate the proportion with faces later

            if face_count > 0:
                face_counts[decade]["pages_with_faces"] += 1  # if the page contains over one face then we increase the face count for the decade

        for decade, counts in face_counts.items():
            if counts["pages"] > 0: #sanity check to gate keep only the decades that had their pages analysed
                pct_faces = (counts["pages_with_faces"] / counts["pages"]) * 100  # calculates the percentage of pages with faces
                results.append({ #append the results
                    "Newspaper": newspaper,
                    "Decade": decade,
                    "Total Pages": counts["pages"],
                    "Pages with Faces": pct_faces
                }) 

    return results

def save_results_and_plot(results, output_folder):
    """Save results to CSV and generate plots"""
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)  # create the output directory if it doesn't exist

    df = pd.DataFrame(results)  # converts the results to a DataFrame
    csv_path = os.path.join(output_folder, "face_detection_results.csv")
    df.to_csv(csv_path, index=False)  # save the DataFrame to a CSV file

    # Plotting
    for newspaper in df["Newspaper"].unique(): # makes a plot for each newspaper chronologically
        subset = df[df["Newspaper"] == newspaper]  # filter the df for the current newspaper
        plt.figure()
        plt.plot(subset["Decade"], subset["Pages with Faces"], marker="o")  # plot percentage of pages with faces per decade
        plt.title(f"Percentage of Pages That Contain Faces per Decade - {newspaper}")
        plt.xlabel("Decade")
        plt.ylabel("Percentage of Pages That Contain Faces")
        plt.grid(True)
        plot_path = os.path.join(output_folder, f"{newspaper}_faces_plot.png")
        plt.savefig(plot_path)  # saves the plot
        plt.close()

if __name__ == "__main__":
    # Define the folders
    input_folder = "in"
    output_folder = "out"
    newspapers = ["GDL", "JDG", "IMP"]

    # Process newspapers (main pipeline)
    results = process_newspapers(input_folder, newspapers)
    
    # Save results and plot
    save_results_and_plot(results, output_folder)
    print("Analysis complete. All outputs saved in the out folder.")
