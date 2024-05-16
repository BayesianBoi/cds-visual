import pandas as pd

def compare_results():
    # Load results from Image Search Script
    img_search_csv = f"../out/hist_image_0001.csv"
    img_search_df = pd.read_csv(img_search_csv)
    img_search_top5 = set(img_search_df['Filename'].iloc[1:6])  # Skip the first row (chosen image itself)

    # Load results from VGG16 Script
    vgg16_csv = f"../out/vgg16_image_0001.csv"
    vgg16_df = pd.read_csv(vgg16_csv)
    vgg16_top5 = set(vgg16_df['Filename'].iloc[1:6])  # Skip the first row (chosen image itself)

    # Compare results
    common_images = img_search_top5.intersection(vgg16_top5)
    print(f"Common similar images for {chosen_image_filename}: {common_images}")
    print(f"Number of common similar images: {len(common_images)}")

if __name__ == "__main__":
    compare_results()
