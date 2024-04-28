import matplotlib.pyplot as plt
import matplotlib.patches as patches

def visualize_image(image_path, label_path):
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