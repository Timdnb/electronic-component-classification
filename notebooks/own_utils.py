import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

def visualize_image(image_path, label_path):
    """ Visualize the image with bounding boxes

    Args:
        image_path (str): Path to the image file
        label_path (str): Path to the label file
    """
    # Load the image
    image = plt.imread(image_path)
    
    # Create a figure and axes
    fig, ax = plt.subplots()
    
    # Display the image with opacity lower
    ax.imshow(image)
    
    # Read the labels from the label file
    with open(label_path, 'r') as f:
        labels = f.readlines()
    
    # Parse and draw bounding boxes
    for label in labels:
        label = label.strip().split(' ')
        folder = label[0]
        x = float(label[1])
        y = float(label[2])
        width = float(label[3])
        height = float(label[4])
        
        # Calculate the coordinates of the bounding box
        left = x - width / 2
        top = y - height / 2

        # Denormalize the coordinates
        left *= image.shape[1]
        top *= image.shape[0]
        width *= image.shape[1]
        height *= image.shape[0]
        
        # Create a rectangle patch
        rect = patches.Rectangle((left, top), width, height, linewidth=1, edgecolor='r', facecolor='none')
        
        # Add the rectangle patch to the axes
        ax.add_patch(rect)
    
    # Remove axis ticks and labels
    ax.axis('off')
    
    # Show the plot
    plt.show()

# Remove overlapping junctions
def remove_overlapping_junctions(j_results, c_results, overlap_threshold=0.1):
    """ Remove junctions that are overlapping with components

    Args:
        j_results (yolov5.results): Results of the junction detection model
        c_results (yolov5.results): Results of the component detection model

    Returns:
        list: List of coordinates of components with confidence scores and class labels
        list: List of coordinates of junctions that are not overlapping with components with confidence scores and class labels
    """
    # Create list to store junctions to be removed
    j_to_remove = []

    # Get the bounding boxes of the junctions and the components
    j_boxes = j_results.xyxy[0]
    c_boxes = c_results.xyxy[0]

    # Get the coordinates of the junctions and the components
    j_coords = [(int(box[0]), int(box[1]), int(box[2]), int(box[3]), np.round(float(box[4]), 4), int(box[5])) for box in j_boxes]
    c_coords = [(int(box[0]), int(box[1]), int(box[2]), int(box[3]), np.round(float(box[4]), 4), int(box[5])) for box in c_boxes]

    # Remove junctions that are overlapping with components
    for c_coord in c_coords:
        for j_coord in j_coords:
            # Calculate percentage of junction that is overlapping with component
            x1 = max(c_coord[0], j_coord[0])
            y1 = max(c_coord[1], j_coord[1])
            x2 = min(c_coord[2], j_coord[2])
            y2 = min(c_coord[3], j_coord[3])

            # Calculate area of intersection
            intersection = max(0, x2 - x1 + 1) * max(0, y2 - y1 + 1)

            # Calculate area of junction
            j_area = (j_coord[2] - j_coord[0] + 1) * (j_coord[3] - j_coord[1] + 1)

            # Calculate percentage of junction that is overlapping with component
            overlap = intersection / j_area

                        # Remove junction if it is overlapping with component
            if overlap > overlap_threshold:
                j_to_remove.append(j_coord)
    
    # Remove junctions that are overlapping with components
    j_coords = [j_coord for j_coord in j_coords if j_coord not in j_to_remove]

    return c_coords, j_coords

# Perform non-maximum suppression
def non_max_suppression_fast(coords, iou_threshold=0.5):
    """ Apply non-maximum suppression to the coordinates of junctions or components
    Args:
        coords (list): List of coordinates with confidence scores and class labels
        iou_threshold (float): Intersection over Union (IoU) threshold
    Returns:
        list: List of coordinates after non-maximum suppression with confidence scores and class labels
    """
    if len(coords) == 0:
        return []
    
    # Transform the coordinates into a numpy array
    boxes = np.array([[x1, y1, x2, y2] for x1, y1, x2, y2, score, label in coords])
    scores = np.array([score for x1, y1, x2, y2, score, label in coords])

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    # Compute the area of the bounding boxes
    area = (x2 - x1 + 1) * (y2 - y1 + 1)

    # Sort by confidence
    indices = np.argsort(scores)[::-1]

    keep = []

    # Iterate over the bounding boxes
    while len(indices) > 0:
        current = indices[0]
        keep.append(current)

        xx1 = np.maximum(x1[current], x1[indices[1:]])
        yy1 = np.maximum(y1[current], y1[indices[1:]])
        xx2 = np.minimum(x2[current], x2[indices[1:]])
        yy2 = np.minimum(y2[current], y2[indices[1:]])

        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        overlap = (w * h) / area[indices[1:]]

        suppressed_indices = np.where(overlap <= iou_threshold)[0]

        indices = indices[suppressed_indices + 1]

    return [coords[i] for i in keep]