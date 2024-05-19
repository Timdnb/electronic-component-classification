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


def visualize_image_grid(image_paths, label_paths):
    # Ensure the number of image paths and label paths are the same
    assert len(image_paths) == len(label_paths), "Number of images and labels must match"
    
    # Create a figure with 4x4 subplots
    fig, axs = plt.subplots(4, 4, figsize=(16, 16))
    axs = axs.flatten()

    for idx, (image_path, label_path) in enumerate(zip(image_paths, label_paths)):
        # Load the image
        image = plt.imread(image_path)
        
        # Display the image
        axs[idx].imshow(image)
        
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
            axs[idx].add_patch(rect)
        
        # Remove axis ticks and labels
        axs[idx].axis('off')
    
    # Hide any unused subplots
    for j in range(idx + 1, len(axs)):
        axs[j].axis('off')
    
    # Adjust layout to avoid overlap
    plt.tight_layout()
    plt.show()

