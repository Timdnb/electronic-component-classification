import argparse
import os
import sys
sys.path.append('..')
from global_utils import remove_overlapping_junctions, non_max_suppression_fast, create_window, apply_transformations
import cv2
import torch
from huggingface_hub import hf_hub_download
import pandas as pd
from tqdm import tqdm

if __name__ == '__main__':
    parset = argparse.ArgumentParser(description='Detect components and junctions in image')
    parset.add_argument('--img-path', type=str, help='Path to image', required=True)
    parset.add_argument('--outdir', type=str, help='Path to output directory')

    args = parset.parse_args()

    # File and directory paths
    img_path = args.img_path
    out_path = args.outdir

    if out_path is None and os.path.isfile(img_path):
        out_path = os.path.dirname(img_path)
    elif out_path is None:
        out_path = img_path

    if os.path.isfile(img_path):
        filenames = [img_path]
    else:
        filenames = os.listdir(img_path)
        filenames = [os.path.join(img_path, filename) for filename in filenames if not filename.startswith('.')]
        filenames.sort()
    
    os.makedirs(out_path, exist_ok=True)

    # Download components model and run it
    c_hf_model = hf_hub_download('Timdb/electronic-circuit-detection', 'components.pt')
    c_model = torch.hub.load('ultralytics/yolov5', 'custom', c_hf_model, verbose=False, force_reload=True)
    c_model.eval()

    # Download junctions model and run it
    j_hf_model = hf_hub_download('Timdb/electronic-circuit-detection', 'junctions.pt')
    j_model = torch.hub.load('ultralytics/yolov5', 'custom', j_hf_model, verbose=False, force_reload=True)
    j_model.eval()

    print("Adjust the sliders to apply transformations to the image. Press 'Enter' to continue.\n")
    for filename in tqdm(filenames):
        # Load image and apply transformations
        tf = create_window(filename)
        img = cv2.imread(filename)
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = apply_transformations(gray_img, tf['contrast'], tf['blur'], tf['threshold'], tf['erode'], tf['dilate'], tf['invert'])

        c_results = c_model(img)
        j_results = j_model(img)

        # Remove overlapping junctions
        c_coords, j_coords = remove_overlapping_junctions(j_results, c_results, overlap_threshold=0.1)

        # Remember the old junctions
        columns = ['xmin', 'ymin', 'xmax', 'ymax', 'confidence', 'class']
        old_j_df = pd.DataFrame(j_coords, columns=columns)
        old_c_df = pd.DataFrame(c_coords, columns=columns)

        # Perform non-maximum suppression on coords and junctions
        c_coords = non_max_suppression_fast(c_coords, iou_threshold=0.4)
        j_coords = non_max_suppression_fast(j_coords, iou_threshold=0.4)

        # Create dataframes from the components and remaining junctions
        c_df = pd.DataFrame(c_coords, columns=columns)
        j_df = pd.DataFrame(j_coords, columns=columns)

        # using c_df containing the components and j_df containing the junctions, draw all bounding boxes on the original image
        img = cv2.imread(filename)
        c_labels = c_model.model.names
        j_labels = j_model.model.names

        # Draw components
        for index, row in c_df.iterrows():
            xmin = int(row['xmin'])
            ymin = int(row['ymin'])
            xmax = int(row['xmax'])
            ymax = int(row['ymax'])
            cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)

            label = c_labels[int(row['class'])]
            cv2.putText(img, label, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

        # Draw junctions
        for index, row in j_df.iterrows():
            xmin = int(row['xmin'])
            ymin = int(row['ymin'])
            xmax = int(row['xmax'])
            ymax = int(row['ymax'])
            cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 0, 255), 2)

            label = j_labels[int(row['class'])]
            cv2.putText(img, label, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

        # Save the image
        filename = os.path.basename(filename)
        cv2.imwrite(os.path.join(out_path, filename[:filename.rfind('.')] + '_labeled.jpg'), img)

    print(f"\nOutput saved to {out_path}")