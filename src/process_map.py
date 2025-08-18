import torch
import cv2
import numpy as np
import geopandas as gpd
from rasterio.transform import from_gcps
from rasterio.control import GroundControlPoint
from shapely.geometry import Polygon
import albumentations as A
from albumentations.pytorch import ToTensorV2
import os
from tqdm import tqdm
import segmentation_models_pytorch as smp
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image

def main():
    print("Starting inference process")
    
    # configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    IMAGE_PATH = os.path.join("data", "input", "stockton_1.png")
    SEG_MODEL_PATH = os.path.join("outputs", "models", "tuned_model.pth")
    OCR_MODEL_PATH = os.path.join("outputs", "models", "trocr-tuned-ocr")
    OUTPUT_GEOPACKAGE_PATH = os.path.join("outputs", "data", "final_output.gpkg")
    OUTPUT_MASK_PATH = os.path.join("outputs", "images", "prediction_mask.png")
    NUM_CLASSES, PATCH_SIZE, STRIDE = 3, 256, 128

    # load models
    print("Loading OCR model")
    ocr_processor = TrOCRProcessor.from_pretrained(OCR_MODEL_PATH)
    ocr_model = VisionEncoderDecoderModel.from_pretrained(OCR_MODEL_PATH).to(device)
    ocr_model.eval()

    print("Loading segmentation model")
    seg_model = smp.Unet(
        encoder_name="resnet34", encoder_weights=None,
        in_channels=3, classes=NUM_CLASSES,
    ).to(device)
    seg_model.load_state_dict(torch.load(SEG_MODEL_PATH, map_location=device, weights_only=True))
    seg_model.eval()
    
    # load and preprocess image
    image_rgb = cv2.cvtColor(cv2.imread(IMAGE_PATH), cv2.COLOR_BGR2RGB)
    h, w, _ = image_rgb.shape
    transform = A.Compose([
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])
    
    # run sliding window inference
    print("Running segmentation inference")
    full_preds_logits = np.zeros((h, w, NUM_CLASSES), dtype=np.float32)
    count_map = np.zeros((h, w), dtype=np.float32)
    for y in tqdm(range(0, h, STRIDE)):
        for x in range(0, w, STRIDE):
            y_end, x_end = min(y + PATCH_SIZE, h), min(x + PATCH_SIZE, w)
            patch = image_rgb[y:y_end, x:x_end]
            h_patch, w_patch, _ = patch.shape
            pad_h, pad_w = PATCH_SIZE - h_patch, PATCH_SIZE - w_patch
            if pad_h > 0 or pad_w > 0:
                patch = np.pad(patch, ((0, pad_h), (0, pad_w), (0, 0)), mode='constant')
            
            input_tensor = transform(image=patch)['image'].unsqueeze(0).to(device)
            with torch.no_grad():
                outputs = seg_model(input_tensor)
            
            logits_patch = outputs.squeeze().cpu().numpy().transpose(1, 2, 0)
            full_preds_logits[y:y_end, x:x_end] += logits_patch[:h_patch, :w_patch]
            count_map[y:y_end, x:x_end] += 1

    full_preds_logits /= (count_map[..., np.newaxis] + 1e-6)
    preds = np.argmax(full_preds_logits, axis=2).astype(np.uint8)
    cv2.imwrite(OUTPUT_MASK_PATH, preds * 100)
    print(f"Prediction mask saved to '{OUTPUT_MASK_PATH}'")

    # post-processing, vectorization, and ocr
    all_geometries, all_attributes = [], []
    boundary_mask = np.uint8(preds == 1) * 255
    contours, _ = cv2.findContours(boundary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        if cv2.contourArea(cnt) > 50:
            pixel_poly = Polygon(np.squeeze(cnt))
            all_geometries.append(pixel_poly)
            all_attributes.append({'class': 'boundary', 'text': None, 'pixel_geom': pixel_poly.wkt})

    text_mask = np.uint8(preds == 2) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 5))
    dilated_text_mask = cv2.dilate(text_mask, kernel, iterations=1)
    contours, _ = cv2.findContours(dilated_text_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print("Running OCR on text regions")
    for cnt in tqdm(contours):
        if cv2.contourArea(cnt) > 20:
            x_b, y_b, w_b, h_b = cv2.boundingRect(cnt)
            text_roi = Image.fromarray(image_rgb[y_b:y_b+h_b, x_b:x_b+w_b])
            pixel_poly = Polygon([(x_b, y_b), (x_b + w_b, y_b), (x_b + w_b, y_b + h_b), (x_b, y_b + h_b)])
            pixel_values = ocr_processor(images=text_roi, return_tensors="pt").pixel_values.to(device)
            with torch.no_grad():
                generated_ids = ocr_model.generate(pixel_values, max_new_tokens=64)
            ocr_text = ocr_processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
            all_geometries.append(pixel_poly)
            all_attributes.append({'class': 'text', 'text': ocr_text, 'pixel_geom': pixel_poly.wkt})

    print(f"Processed {len(all_attributes)} features.")

    # georeference and save
    if not all_geometries:
        print("No features found to georeference.")
        return
        
    gcps = [GroundControlPoint(157, 107, 336100, 516800), GroundControlPoint(159, 831, 336400, 516800),
            GroundControlPoint(1679, 105, 336100, 516100), GroundControlPoint(1682, 832, 336400, 516100)]
    geo_transform = from_gcps(gcps)
    georeferenced_polygons = [Polygon([geo_transform * pt for pt in geom.exterior.coords]) for geom in all_geometries]
    gdf = gpd.GeoDataFrame(all_attributes, geometry=georeferenced_polygons, crs="EPSG:27700")
    gdf.to_file(OUTPUT_GEOPACKAGE_PATH, driver='GPKG')
    print(f"Geospatial data saved to '{OUTPUT_GEOPACKAGE_PATH}'")

if __name__ == "__main__":
    main()