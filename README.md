# vehicle-classification-under-adverse-conditions
This project studies the robustness of vehicle type classification models under adverse visual conditions, with a focus on blurry images. We compare traditional image processing techniques and deep learning–based fine-tuning to evaluate their effectiveness across different distortion severities.

The project investigates:

* Whether image processing can recover useful visual information

* When fine-tuning outperforms image processing

* How distortion type and severity affect model performance

## Baseline Classification Model
The baseline vehicle classification model we used for this project comes from the MobileNetV2 CNN-based model from [this github repo](https://github.com/hoanhle/Vehicle-Type-Detection?).

## Dataset
This project uses a set of ~6000 distorted images of different types of vehicles under different type of distortion. 

Vehicle classes: bicycle, boat, bus, car, helicopter, motorcycle, truck

Distortions: Bad lighting condition, foggy condition, blurry condition

Images are stored in class-specific folders for each type of distortion. Due to file size constraints, datasets are not included in the repository. They are available for download separately in those google drive link:
* [Low-light dataset](https://drive.google.com/drive/u/0/folders/1m0AXwPMiRXJRXr7Ho5PGOJAOAmLyG743?q=sharedwith:public%20parent:1m0AXwPMiRXJRXr7Ho5PGOJAOAmLyG743)
* [Foggy dataset](https://drive.google.com/file/d/1kGqC6a4gG2upr7aw-FOwlAgkAq5rsjfn/view?usp=sharing)
* [Blurry dataset](https://drive.google.com/drive/folders/1MVzFhYq91adCOrwVLix0lWJHA2tPHj-w?usp=sharing)
Please place your images following the instruction within README of each type of distortion's folder to run the code.

---
# Setup Instructions
1. Create Environment
```bash
python -m venv venv
source venv/bin/activate
```
2. Install Dependencies
```bash
pip install -r requirements.txt
```
3. Run code:

For each type of distortion, to run the specific code, please `cd` into the corresponding folder, download the corresponding dataset and place it in that folder, then run the main file or follow specific execution instructions in that folder's instruction file.

---
## Release
The version corresponding to the final report is tagged as: `v1.0-report-version`

## License
This project is released under the MIT License.
