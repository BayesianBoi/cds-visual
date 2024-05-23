# Assignment 4 - Detecting faces in historical newspapers

## Repository overview
This repository contains scripts to analyse three historical French newspapers for the presence of human faces. The primary goal is to detect and track the prevalence of faces in newspaper images over the past 200 years. 
We use a pre-trained CNN model for face detection and analyse the results by decade.

### Assignment objective
1. For each of the three newspapers:
    - Gazette de Lausanne (GDL, 1790-2000)
    - Journal de Genève (JDG, 1820-1990)
    - Impartial (IMP, 1880-2010)
2. Go through each page and find how many faces are present
3. Group these results together by decade and then save the following:
    - A CSV showing the total number of faces per decade and the percentage of pages for that decade which have faces on them
    - A plot showing the percentage of pages with faces per decade for each newspaper.
  
## Data source
The data for this assignment is available [here](https://zenodo.org/records/3706863).
It should be inserted into the `/in` folder before proceeding with the analysis.

## Steps for running the analysis

### Setting up the environment
1. **Set up the virtual environment and install requirements:**
    ```bash
    bash setup.sh
    ```
2. **Activate the virtual environment:**
    ```bash
    source envVis4/bin/activate
    ```

### Running the Code

1. **[Download](https://zenodo.org/records/3706863) the data set and place the data in the `/in` folder**

2. **Run the main analysis script:**
    ```bash
    python src/main-code.py
    ```

## Extra scripts
The main analysis gave weird results for the early editions of GDL, indicating that up to 25% of the pages contained faces. Therefore, a sanity check was done to verify these results.

1. **To run the extra script which analyses only the GDL 1790-1880 editions:**
```bash
python src/gdl_check.py

```

2. **To run the plotting script for detected faces from GDL 1780-1880:**
```bash
python src/plot_faces.py
```
Note: It is required to run `gdl_check.py` first before running the plotting script.

## Summary of results
### CSV-file:
The CSV-file with all of the results can be found in the `/out` folder.

### Plots:

#### Gazette de Lausanne (GDL)
![GDL](https://github.com/BayesianBoi/cds-visual/blob/main/assignments/assignment%204/out/GDL_faces_plot.png)

*Notice the prevalance of faces in the early 1800's while the other newspaper almost features no faces in their pages before the late 1880s.*


#### Journal de Genève (JDG)
![JDG](https://github.com/BayesianBoi/cds-visual/blob/main/assignments/assignment%204/out/JDG_faces_plot.png)


#### Impartial (IMP)
![IMP](https://github.com/BayesianBoi/cds-visual/blob/main/assignments/assignment%204/out/IMP_faces_plot.png)


#### False positives in GDL newspaper (1790-1880)
![GDL False positives](https://github.com/BayesianBoi/cds-visual/blob/main/assignments/assignment%204/out/sanity_check_for_GDL_1790_1880.png)

**These are all of the pictures from GDL 1790-1880 that were detected to contain faces. Notice that none of the pages contain faces.**

*The log file, showing which pages were detected to contain faces, can be found in the `/out` folder*

## Discussion
### Key points
- The prevalence of faces in newspapers increases significantly over the decades, especially from the late 19th century onwards.
- All of the newspapers show a rise in the bringing of face images around the 20th century, which could reflect the advances in photographic technology through that centuary.

### False positives in GDL (1790-1880)
The primary analysis indicated that for some of the early 1800s decades, up to 25% of all the pages contained faces. However, as shown in the plot above those were all purely false positives.

### Historical context
The results align with historical developments in print media. The first image brought in a newspaper was in 1848 and the first publically available camera was the Kodak, which was introduced in 1888. As the first image was brought in 1848, any faces detected before 1848 should be assumed to be false positives. We only did a sanity check for the GDL but one could assume that the JDG also contains false positives, as it also detects face images before 1848. Generally, after the introduction of the first camera, it seems the newspapers also naturally began to print more pages with faces.

## Limitations and possible improvements
### Limitations
- **False positives:** The model shows a large number of false positives for the early editions of the GDL. These false positives might be caused by low print quality from that time. I also doubt that the ResNes model is trained on data that even remotely resembles pictures in old newspapers.
- **Data quality:** Variations in the quality and type of newspaper images can affect the accuracy of face detection.

### Possible improvements
- **Model tuning:** Further tuning of the face detection model to better predict this type of data could help reduce the number of false positives.
- **Image preprocessing:** Improved preprocessing steps might improve detection accuracy. The quality of the images were varying quite a lot.
