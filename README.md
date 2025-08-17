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
flowchart TD
    A[Start] --> B[1. Run `src/draft_mask.py` (pptional)];
    B --> C{2. Refine draft_mask to create ground truth masks (manual)};
    C --> D[3. Run `src/create_ocr_dataset.py`];
    D --> E{4. Create `metadata.csv` and populate with text labels (manual)};
    E --> F[5. Run `src/train_segmentation.py`];
    E --> G[6. Run `src/train_ocr.py`];
    subgraph Training
        F --> H((Segmentation Model));
        G --> I((OCR Model));
    end
    H & I --> J[7. Run `src/process_map.py`];
    J --> K[8. Run `src/verify_cv.py`];
    K --> L[End: View Final Outputs];
```
    
    
## Scripts Description

All executable scripts are located in the `src/` directory.

* `draft_mask.py`: A utility to programmatically generate a draft mask for manual refinement.
* `create_ocr_dataset.py`: Extracts text image snippets from the ground truth text mask to prepare data for OCR training.
* `train_segmentation.py`: Trains the fine-tuned U-Net segmentation model.
* `train_ocr.py`: Fine-tunes the TrOCR model for handwriting recognition.
* `process_map.py`: The main inference script that runs the full pipeline on a source image to generate the final GeoPackage.
* `verify_cv.py`: A utility to programmatically verify the final geospatial output by creating an annotated image.
  
    
    