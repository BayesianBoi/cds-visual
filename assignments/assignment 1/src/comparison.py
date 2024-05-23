import pandas as pd

def compare_results():
    # Load results from colour hist model
    img_search_csv = "../out/hist_image_0001.jpg.csv"
    img_search_df = pd.read_csv(img_search_csv)
    img_search_top5 = set(img_search_df["Filename"].iloc[1:6])  # taking the top five most similar images while skipping the first row (which is the chosen image itself)

    # loading the results from vgg16 model
    vgg16_csv = "../out/vgg16_image_0001.jpg.csv"
    vgg16_df = pd.read_csv(vgg16_csv)
    vgg16_top5 = set(vgg16_df["Filename"].iloc[1:6])  # taking the top five most similar images while skipping the first row (which is the chosen image itself)

    # comparing the results to see if any of the predicted pictures overlap
    common_images = img_search_top5.intersection(vgg16_top5)
    print(f"Common similar images for {chosen_image_filename}: {common_images}")
    print(f"Number of common similar images: {len(common_images)}")

if __name__ == "__main__":
    compare_results()
