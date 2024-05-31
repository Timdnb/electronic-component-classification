import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import cv2
from tkinter import Tk, Label, Scale, HORIZONTAL, Checkbutton, IntVar
from PIL import Image, ImageTk

# Helper functions
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

# Code for slider
MAX_WIDTH = 800  # Maximum width of the image display
MAX_HEIGHT = 800  # Maximum height of the image display

def apply_transformations(img, contrast, blur, threshold, erode, dilate, invert):
    """ Apply transformations to the input image

    Args:
        img (cv2 image): input image
        contrast (float): contrast value
        blur (int): blur value
        threshold (int): threshold value
        erode (int): erode value
        dilate (int): dilate value
        invert (bool): invert value

    Returns:
        cv2 image: transformed image
    """
    # Apply contrast
    img = cv2.convertScaleAbs(img, alpha=contrast, beta=0)

    # Apply blur
    if blur > 0:
        img = cv2.GaussianBlur(img, (2 * blur + 1, 2 * blur + 1), 0)
    
    # Apply threshold
    _, img = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY)
    
    # Apply erosion
    if erode > 0:
        kernel = np.ones((2, 2), np.uint8)
        img = cv2.erode(img, kernel, iterations=erode)
    
    # Apply dilation
    if dilate > 0:
        kernel = np.ones((2, 2), np.uint8)
        img = cv2.dilate(img, kernel, iterations=dilate)
    
    # Invert image
    if invert:
        img = cv2.bitwise_not(img)
    
    return img

def resize_image(img, max_width, max_height):
    """ Resize image for display

    Args:
        img (cv2 image): input image
        max_width (int): maximum width
        max_height (int): maximum height

    Returns:
        cv2 image: resized image
    """
    h, w = img.shape[:2]
    if h > max_height or w > max_width:
        scaling_factor = min(max_width / w, max_height / h)
        new_size = (int(w * scaling_factor), int(h * scaling_factor))
        img = cv2.resize(img, new_size, interpolation=cv2.INTER_AREA)
    return img

def update_image():
    # Get the current values from the sliders and checkbox
    contrast = contrast_scale.get()
    blur = blur_scale.get()
    threshold = threshold_scale.get()
    erode = erode_scale.get()
    dilate = dilate_scale.get()
    invert = invert_var.get()
    
    # Apply transformations
    transformed_img = apply_transformations(gray_img.copy(), contrast, blur, threshold, erode, dilate, invert)
    
    # Resize the image to fit within the constraints
    resized_img = resize_image(transformed_img, MAX_WIDTH, MAX_HEIGHT)
    
    # Convert the image to a format suitable for Tkinter
    img_tk = ImageTk.PhotoImage(image=Image.fromarray(resized_img))
    
    # Update the label with the new image
    img_label.config(image=img_tk)
    img_label.image = img_tk

def store_and_close(event=None):
    # Get the final transformation values
    final_values['contrast'] = contrast_scale.get()
    final_values['blur'] = blur_scale.get()
    final_values['threshold'] = threshold_scale.get()
    final_values['erode'] = erode_scale.get()
    final_values['dilate'] = dilate_scale.get()
    final_values['invert'] = invert_var.get()
    
    # Close the window
    root.destroy()

def create_window(image_path):
    global gray_img, img_label, contrast_scale, blur_scale, threshold_scale, erode_scale, dilate_scale, invert_var, final_values, root
    
    final_values = {}
    
    # Load the image and convert to grayscale
    img = cv2.imread(image_path)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Resize the image to fit within the constraints
    gray_img = resize_image(gray_img, MAX_WIDTH, MAX_HEIGHT)
    
    # Create the main window
    root = Tk()
    root.title("Image Processing")
    
    # Display the initial image
    img_tk = ImageTk.PhotoImage(image=Image.fromarray(gray_img))
    img_label = Label(root, image=img_tk)
    img_label.image = img_tk
    img_label.grid(row=0, column=0, rowspan=6)
    
    # Create sliders and checkbox
    contrast_scale = Scale(root, from_=0.1, to=3.0, resolution=0.1, orient=HORIZONTAL, label="Contrast")
    contrast_scale.set(1.1)
    contrast_scale.grid(row=0, column=1, padx=10, pady=5)
    
    blur_scale = Scale(root, from_=0, to=10, orient=HORIZONTAL, label="Blur")
    blur_scale.set(0)
    blur_scale.grid(row=1, column=1, padx=10, pady=5)
    
    threshold_scale = Scale(root, from_=0, to=255, orient=HORIZONTAL, label="Threshold")
    threshold_scale.set(127)
    threshold_scale.grid(row=2, column=1, padx=10, pady=5)
    
    erode_scale = Scale(root, from_=0, to=10, orient=HORIZONTAL, label="Erosion")
    erode_scale.set(0)
    erode_scale.grid(row=3, column=1, padx=10, pady=5)
    
    dilate_scale = Scale(root, from_=0, to=10, orient=HORIZONTAL, label="Dilation")
    dilate_scale.set(0)
    dilate_scale.grid(row=4, column=1, padx=10, pady=5)
    
    invert_var = IntVar(value=1)
    invert_check = Checkbutton(root, text="Invert", variable=invert_var)
    invert_check.grid(row=5, column=1, padx=10, pady=5)
    
    # Update the image when sliders or checkbox change
    contrast_scale.bind("<Motion>", lambda event: update_image())
    blur_scale.bind("<Motion>", lambda event: update_image())
    threshold_scale.bind("<Motion>", lambda event: update_image())
    erode_scale.bind("<Motion>", lambda event: update_image())
    dilate_scale.bind("<Motion>", lambda event: update_image())
    invert_check.bind("<ButtonRelease-1>", lambda event: update_image())
    
    # Bind the Enter key to store values and close the window
    root.bind('<Return>', store_and_close)
    
    # Start the Tkinter event loop
    root.mainloop()
    
    return final_values