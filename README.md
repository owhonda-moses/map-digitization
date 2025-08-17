# Map Digitization Project

This project uses a deep learning pipeline to automatically digitize handwritten annotations from scanned map images. It performs two main tasks:
1.  **Image Segmentation**: Identifies and extracts pixel masks for handwritten boundaries and text using a fine-tuned U-Net model.
2.  **Optical Character Recognition (OCR)**: Recognizes the text within the extracted masks using a fine-tuned TrOCR model.

The final output is a GeoPackage file containing georeferenced polygons of the detected features, with the recognized text included as metadata.

## Repository Structure

The repository is organized to separate source code, data, and generated outputs.

```mermaid
graph TD
    A[maps-cv] --> R1[environment.yml]
    A --> R2[setup.sh]
    A --> R3[README.md]
    A --> R4[.gitignore]
    
    A --> B(data)
    B --> B1(input)
    B --> B2(ocr_data)
    
    A --> C(src)
    C --> C1[train_unet.py]
    C --> C2[train_ocr.py]
    C --> C3[process_map.py]
    C --> C4[...]

    A --> D(outputs)
    D --> D1(models)
    D --> D2(data)
    D --> D3(images)
    
    subgraph Input Data
        B1 --> F1[stockton_1.png]
        B1 --> F2[boundaries_mask.png]
        B1 --> F3[text_mask.png]
    end

    subgraph Generated Models
        D1 --> M1[tuned_model.pth]
        D1 --> M2[trocr-tuned-ocr/]
    end
    ```
    
## Setup and Installation

This project uses Conda for robust environment management and is designed to be run in an ephemeral cloud environment.

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
    conda activate map-digitization
    ```
    
## Usage Workflow
The project is designed to be run in a sequential workflow. All scripts should be run from the root of the repository after activating the Conda environment.


flowchart TD
    A[Start] --> B{1. Prepare Ground Truth Masks};
    B --> C[2. Run `src/ocr_dataset.py`];
    C --> D{3. Create `metadata.csv`};
    D --> E[4. Run `src/train_unet.py`];
    D --> F[5. Run `src/train_ocr.py`];
    subgraph Training
        E --> G((Segmentation Model));
        F --> H((OCR Model));
    end
    G & H --> I[6. Run `src/process_map.py`];
    I --> J[7. Run `src/verify_cv.py`];
    J --> K[End: View Final Outputs];
    
    
## Scripts Description

All executable scripts are located in the `src/` directory.

* `draft_mask.py`: A utility to programmatically generate a draft mask for manual refinement.
* `create_ocr_dataset.py`: Extracts text image snippets from the ground truth text mask to prepare data for OCR training.
* `train_segmentation.py`: Trains the fine-tuned U-Net segmentation model.
* `train_ocr.py`: Fine-tunes the TrOCR model for handwriting recognition.
* `process_map.py`: The main inference script that runs the full pipeline on a source image to generate the final GeoPackage.
* `verify_cv.py`: A utility to programmatically verify the final geospatial output by creating an annotated image.
  
    
    