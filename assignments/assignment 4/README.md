# Assignment 4 - Detecting Faces in Historical Newspapers

## Repository Overview
This repository contains scripts to analyze historical newspapers for the presence of human faces. The primary goal is to detect and track the prevalence of faces in newspaper images over the past 200 years. 
We use a pre-trained CNN model for face detection and analyse the results by decade.

### Assignment Objective
1. For each of the three newspapers:
    - **Gazette de Lausanne (GDL, 1790-2000)**
    - **Journal de Genève (JDG, 1820-1990)**
    - **Impartial (IMP, 1880-2010)**
2. Analyse each page to find how many faces are present.
3. Group the results by decade and save the following:
    - A CSV showing the total number of faces per decade and the percentage of pages for that decade with faces.
    - A plot showing the percentage of pages with faces per decade for each newspaper.

## Data Source
The data for this assignment is available from here [here](https://zenodo.org/records/3706863).
It should be inserted into the **/in** folder before proceding with the analysis.

## Steps to running the code

### Running the Code
The project is structured with scripts saved in the `src` folder and the output saved in the `out` folder.

#### Main Script
1. To run the setup script:
  ```bash

sh setup.sh
  ```

2. To run the main analysis script:
  ```bash

python src/main_analysis.py
  ```

#### Extras
The main analysis gave weird results for the early editions of GDL, indicating that up to 25% of the pages contained faces. Therefore, a sanity check was done in order to see if that was true.
To run the extra script which run the same analysis but only the GDL 1780-1880 editions:

```bash
python src/gdl_check.py
```

To run the plotting script for detected faces from GDL 1780-1880:
```bash
python src/plot_faces.py
```

## Summary of Results
### Percentage of Pages That Contain Faces per Decade
#### Gazette de Lausanne (GDL)

#### Journal de Genève (JDG)

#### Impartial (IMP)

### Detected Faces in GDL Newspapers (1790-1880)

## Discussion
### Key Points
- The prevalence of faces in newspapers increases significantly over the decades, especially from the late 19th century onwards.
- All of the newspapers show a rise in face images around the 20th century, which could reflect the advances in photographic technology through that centuary.

### False Positives in GDL (1790-1880)
The analysis of the GDL newspaper from 1790 to 1880 indicates purely false positives. The model incorrectly identifies faces in images from this period, likely due to the lower quality and nature of the printed images.

### Historical Context
The results align with historical developments in print media. The introduction of photographic technology in the 19th century and its widespread adoption in the 20th century led to an increase in images, 
including human faces, in newspapers. This reflects broader cultural and technological shifts towards more visually-oriented media.

## Limitations and Possible Improvements
### Limitations
- **False Positives:** The model shows a significant number of false positives, especially in older newspapers where the print quality is lower.
- **Data Quality:** Variations in the quality and type of newspaper images can affect the accuracy of face detection.

### Possible Improvements
- **Model Tuning:** Further tuning of the face detection model could help reduce the number of false positives.
- **Image Preprocessing:** Improved preprocessing steps might improve detection accuracy.
