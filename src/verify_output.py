import cv2
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import os
from shapely import wkt

def main():
    print("Starting verification")
    
    # paths
    IMAGE_PATH = os.path.join("data", "input", "stockton_1.png")
    GEOPACKAGE_PATH = os.path.join("outputs", "data", "final_output.gpkg")
    OUTPUT_IMAGE_PATH = os.path.join("outputs", "images", "verification.png")

    # load data
    image_to_draw = cv2.cvtColor(cv2.imread(IMAGE_PATH), cv2.COLOR_BGR2RGB)
    gdf = gpd.read_file(GEOPACKAGE_PATH)
    print(f"Loaded {len(gdf)} features from {GEOPACKAGE_PATH}")

    # draw features
    for _, row in gdf.iterrows():
        pixel_poly = wkt.loads(row['pixel_geom'])
        contour = np.array(pixel_poly.exterior.coords, dtype=np.int32).reshape((-1, 1, 2))
        if row['class'] == 'boundary':
            cv2.drawContours(image_to_draw, [contour], -1, (0, 255, 0), 2)
        elif row['class'] == 'text':
            cv2.drawContours(image_to_draw, [contour], -1, (0, 0, 255), 2)
            centroid = pixel_poly.centroid
            cx, cy = int(centroid.x), int(centroid.y)
            ocr_text = row['text']
            cv2.putText(image_to_draw, ocr_text, (cx - 40, cy - 20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    # save and display
    plt.figure(figsize=(15, 15))
    plt.imshow(image_to_draw)
    plt.title("Verification from GeoPackage Data")
    plt.axis('off')
    plt.savefig(OUTPUT_IMAGE_PATH, dpi=300, bbox_inches='tight')
    
    print(f"Verification image saved to '{OUTPUT_IMAGE_PATH}'")

if __name__ == "__main__":
    main()