import cv2
import numpy as np
import geopandas as gpd
import rasterio
from rasterio.transform import from_gcp
from rasterio.control import GroundControlPoint
from shapely.geometry import Polygon, mapping
import pytesseract
import os


def segment_red_boundaries(image_path: str) -> np.ndarray:
    """
    Simulates the output of a trained U-Net segmentation model.
    """
    print("Segmenting red boundaries")
    image = cv2.imread(image_path)
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV) # convert image to HSV color space
    
    # Define two ranges for red color to capture variations
    lower_red1 = np.array([0, 120, 70])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 120, 70])
    upper_red2 = np.array([180, 255, 255])
    
    # Create masks for both ranges and combine them
    mask1 = cv2.inRange(hsv_image, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv_image, lower_red2, upper_red2)
    binary_mask = mask1 + mask2
    
    # Clean up the mask using morphological operations
    kernel = np.ones((3, 3), np.uint8)
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    print("Segmentation complete.")
    return binary_mask

def detect_and_recognize_text(image_path: str, boundary_mask: np.ndarray) -> dict:
    """
    Simulates the output of a trained text detection and OCR pipeline.
    """
    print("Detecting and recognizing text")
    image = cv2.imread(image_path)
    red_only_image = cv2.bitwise_and(image, image, mask=boundary_mask) # isolate only red parts
    
    # configure pytesseract & find text data
    custom_config = r'-c tessedit_char_whitelist=0123456789abcdefghijklmnopqrstuvwxyz/ABCDEFGHIJKLMNOPQRSTUVWXYZ --psm 6'
    ocr_data = pytesseract.image_to_data(red_only_image, config=custom_config, output_type=pytesseract.Output.DICT)
    
    extracted_text = {}
    n_boxes = len(ocr_data['level'])
    for i in range(n_boxes):
        text = ocr_data['text'][i].strip()
        conf = int(ocr_data['conf'][i])
        
        # filter low-confidence and non-relevant detections
        if conf > 40 and len(text) > 2:
            (x, y, w, h) = (ocr_data['left'][i], ocr_data['top'][i], ocr_data['width'][i], ocr_data['height'][i])
            center_point = (x + w // 2, y + h // 2)
            # store text and its central coordinate
            extracted_text[text] = center_point
            print(f"...found text: '{text}' at ({center_point[0]}, {center_point[1]})")

    print("Text recognition complete.")
    return extracted_text



def vectorize_and_georeference(mask: np.ndarray, ocr_results: dict, image_shape: tuple) -> gpd.GeoDataFrame:
    """
    Converts the binary mask to georeferenced polygons and adds OCR text as attributes.
    """
    print("Vectorizing and Georeferencing")
    
    # find contours in the binary mask. these become our polygons
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Based on the map "PLAN NZ 3616", we define our Ground Control Points (GCPs).
    # These map pixel coordinates (row, col) to real-world coordinates (x, y).
    # Real-world coordinates are in British National Grid (EPSG:27700).
    # Format: GroundControlPoint(row, col, x (Easting), y (Northing))
    # We visually inspect the image to find these pixel values.
    gcps = [
        GroundControlPoint(157, 107, 336100, 516800), # top-left grid intersection
        GroundControlPoint(159, 831, 336400, 516800), # top-right grid intersection
        GroundControlPoint(1679, 105, 336100, 516100),# bottom-left grid intersection
        GroundControlPoint(1682, 832, 336400, 516100) # bottom-right grid intersection
    ]
    
    # compute the affine transformation from pixel space to world coordinates
    transform = from_gcp(gcps)
    
    polygons = []
    attributes = []
    
    for contour in contours:
        # filter out very small contours that are likely noise
        if cv2.contourArea(contour) < 500:
            continue
            
        poly_pixels = np.squeeze(contour) # convert contour (pixels) to polygon
        
        # use the transformation to convert pixel coords to real-world coords
        poly_world_coords = [transform * (px[0], px[1]) for px in poly_pixels]
        
        # create a Shapely Polygon
        shapely_poly = Polygon(poly_world_coords)
        polygons.append(shapely_poly)
        
        # find which text belongs to this polygon by checking if the text's center point is inside
        poly_ref = "N/A"
        for text, point in ocr_results.items():
             # Check if text's center point is within the *pixel* contour
            if cv2.pointPolygonTest(contour, point, False) >= 0:
                poly_ref = text
                break
        attributes.append({'reference': poly_ref, 'area_sqm': shapely_poly.area})

    # create the GeoDataFrame
    gdf = gpd.GeoDataFrame(attributes, geometry=polygons, crs="EPSG:27700")
    
    print("Georeferencing complete.")
    return gdf



def main():
    """
    Main function to run the entire pipeline.
    """
    input_image_path = "stockton_1.jpg"
    output_gpkg_path = "extracted_land_parcels.gpkg"
    
    if not os.path.exists(input_image_path):
        print(f"Input image not found at '{input_image_path}'")
        return
    boundary_mask = segment_red_boundaries(input_image_path) # run segmentation model
    ocr_results = detect_and_recognize_text(input_image_path, boundary_mask) # run OCR model
    
    # vectorize, georeference, and save
    image_height, image_width = boundary_mask.shape
    geo_dataframe = vectorize_and_georeference(boundary_mask, ocr_results, (image_height, image_width))
    
    # save the final output
    if not geo_dataframe.empty:
        geo_dataframe.to_file(output_gpkg_path, driver='GPKG')
        print(f"\n Geospatial data saved to '{output_gpkg_path}'")
        print("You can now load file in GIS.")
    else:
        print("\n No valid polygons were extracted.")

if __name__ == "__main__":
    main()