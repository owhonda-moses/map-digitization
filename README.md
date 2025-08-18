# Map Digitization

This project uses a deep learning pipeline to automatically digitize handwritten annotations from scanned map images. It performs two main tasks:
1.  **Image Segmentation**: Identifies and extracts pixel masks for handwritten boundaries and text using a fine-tuned U-Net model.
2.  **Optical Character Recognition (OCR)**: Recognizes the text within the extracted masks using a fine-tuned TrOCR model.

The final output is a GeoPackage file containing georeferenced polygons of the detected features, with the recognized text included as metadata.

    
## Setup and Installation

This project uses Conda for environment management and is designed to be run in an ephemeral cloud environment.

### Target Environment
This setup was developed and tested on a **Paperspace Gradient Notebook**. The `setup.sh` script and environment configuration are tailored for this platform. While it may work in other Debian-based Linux environments with a suitable GPU, adjustments might be necessary.

### Instructions
1.  **Clone the Repository**:
    ```bash
    git clone <your-repository-url>
    cd maps-cv
    ```

2.  **Run the Setup Script**: This script will install Miniconda (if not present) and build the project's Conda environment from the `environment.yml` file using Mamba for speed.
    ```bash
    bash setup.sh
    ```
    The installation will take several minutes the first time it is run.

3.  **Activate the Environment**: After setup is complete, you must activate the Conda environment in your terminal.
    ```bash
    conda init
    conda activate map-cv
    ```
    
## Usage Workflow
The project is designed to be run in a sequential workflow. All scripts should be run from the root of the repository after activating the Conda environment.


```mermaid
%%{init: {"themeVariables": {"fontSize": "10px", "fontFamily": "Arial", "padding": 2}, "flowchart": {"defaultRenderer": "elk"}} }%%
flowchart TD
    A@{ shape: rect, label: "Start" } --> 
    B@{ shape: rect, label: "Run src/draft_mask.py (optional)" } -->
    C@{ shape: rect, label: "Refine mask for ground truth (manual)" } -->
    D@{ shape: rect, label: "Run src/create_ocr_dataset.py" } -->
    E@{ shape: rect, label: "Edit metadata.csv with text labels (manual)" }
    E --> F@{ shape: rect, label: "Run src/train_segmentation.py" }
    E --> G@{ shape: rect, label: "Run src/train_ocr.py" }
    subgraph Training
        F --> H@{ shape: rect, label: "Segmentation Model" }
        G --> I@{ shape: rect, label: "OCR Model" }
    end
    H & I --> J@{ shape: rect, label: "Run src/process_map.py" }
    J --> K@{ shape: rect, label: "Run src/verify_output.py" }
    K --> L@{ shape: rect, label: "View final outputs" }
```

### Detailed Steps

1.  **Run `src/draft_mask.py`**: This optional step creates `draft_mask.png` in `data/input/` to serve as a starting point for your manual masks.

2.  **Refine Draft**: This step requires a layer-based image editor like **GIMP** or **Photoshop**.
    * Open the original `stockton_1.png`.
    * Import the `draft_mask.png` as a new layer.
    * Manually correct the draft by painting with pure white for features and pure black for background.
    * Export two separate files to `data/input/`: `boundaries_mask.png` and `text_mask.png`.

3.  **Run `src/create_ocr_dataset.py`**: This script uses your `text_mask.png` to generate image snippets and saves them in `data/ocr_data/images/`.

4.  **Populate `metadata.csv`**: Create a text file at `data/ocr_data/metadata.csv`. It must contain two columns, `file_name` and `text`, with the correct label for each image snippet generated in the previous step.

    **Example `metadata.csv` format:**
    ```csv
    file_name,text
    images/image_0.png,"W/1098"
    images/image_1.png,"96/0999"
    ```
5.  **Train Models**: Run `src/train_segmentation.py` and `src/train_ocr.py`.
6.  **Run Inference**: Run `src/process_map.py` to generate the final `GeoPackage`.
7.  **Verify Output**: Run `src/verify_output.py` to create a final annotated image.

    
## Scripts Description

All executable scripts are located in the `src/` directory.

* `draft_mask.py`: A utility to programmatically generate a draft mask for manual refinement.
* `create_ocr_dataset.py`: Extracts text image snippets from the ground truth text mask to prepare data for OCR training.
* `train_segmentation.py`: Trains the fine-tuned U-Net segmentation model.
* `train_ocr.py`: Fine-tunes the TrOCR model for handwriting recognition.
* `process_map.py`: The main inference script that runs the full pipeline on a source image to generate the final GeoPackage.
* `verify_output.py`: A utility to programmatically verify the final geospatial output by creating an annotated image.
  
### Manual Annotation (Ground Truth)

The manual steps in this project, (*creating the image masks and the `metadata.csv` file*) are necessary to produce ground truth data because we implement supervised learning which relies on this ground truth to train the models.

* **Segmentation Masks**: The black-and-white masks created with GIMP act as a perfect, pixel-level *answer key* for the U-Net model. By comparing its predictions to this ground truth, the model learns the visual features of what constitutes a _boundary_ or _text_.

* **Metadata CSV**: This file is the _answer key_ for the TrOCR model. It provides the exact, correct text for each cropped text image, allowing the model to learn the association between image pixels with specific characters.

Relying on high-quality ground truth is critical in this project, as the entire training process is derived from a **single source image**, and due to the limited data, the manually created masks and text labels represent the complete and authoritative definition of the features to be learned. Any inaccuracies in this single ground truth will be directly learned and repeated by the models.


### Limitations

* **Single-Source Training Data**: This is the primary limitation. Both the segmentation and OCR models were trained on data derived from a single map. This restricts their ability to generalize to new maps with different wear, tear, or handwriting.
* **Fixed Georeferencing**: The Ground Control Points (GCPs) are hardcoded in `src/process_map.py` for the specific map sheet `NZ 3616`. The system cannot automatically georeference a different map.
* **Static Post-processing**: The contour merging logic in `process_map.py` uses a fixed kernel size, which is tuned for the current image. It may perform sub-optimally on text with significantly different character spacing.

### Future Work

* **Expand the Dataset**: The most impactful improvement would be to acquire and annotate more maps. Training on a larger and more diverse dataset is the only way to significantly improve the generalization and accuracy of both the segmentation and OCR models.
* **Hyperparameter Tuning**: Conduct a systematic search for optimal training settings (e.g learning rate, class weights) to maximize model performance.
* **Automated Georeferencing**: Implement a feature to automatically detect grid lines and corner coordinates from the map image itself, allowing the system to process any map sheet without hardcoded values.
* **Quantitative Evaluation**: Establish a dedicated test set of annotated maps and implement a formal evaluation pipeline that calculates key metrics (e.g Intersection Over Union for segmentation, Character Error Rate for OCR) to objectively benchmark model performance.