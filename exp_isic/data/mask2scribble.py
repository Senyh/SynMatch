import numpy as np
from skimage import io
from skimage import measure, draw, morphology
import matplotlib.pyplot as plt
from skimage.morphology import convex_hull_image, binary_closing, disk
from scipy.ndimage import gaussian_filter
import random
import os
from tqdm import tqdm


def seed(seed=1234):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)


def find2Dlabel_boundaries(segmentation_map):
    target_indexs = np.where(segmentation_map > 0)
    maxs = np.max(np.array(target_indexs), axis=1).max()
    mins = np.min(np.array(target_indexs), axis=1).min()
    return maxs, mins


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def draw_polygons_with_width(image, polygons, value, width=1):
    """
    Draw polygon contours with a specified width on the image.
    :param image: Target image
    :param polygons: List of polygons
    :param value: Label value
    :param width: Line width
    """
    # Create a temporary binary image for drawing contours
    temp = np.zeros_like(image, dtype=bool)
    
    for polygon in polygons:
        if len(polygon) < 2:  # At least two points are required to form a line
            continue
            
        # Convert polygon point coordinates to integers
        polygon = np.round(polygon).astype(int)
        
        # Draw polygon contours
        for i in range(len(polygon)):
            start = polygon[i]
            end = polygon[(i + 1) % len(polygon)]
            
            # Draw line segment
            rr, cc = draw.line(start[0], start[1], end[0], end[1])
            
            # Ensure coordinates are within image bounds
            valid = (rr >= 0) & (rr < image.shape[0]) & (cc >= 0) & (cc < image.shape[1])
            temp[rr[valid], cc[valid]] = True
    
    # Perform dilation if line width is greater than 1
    if width > 1:
        # Create structuring element (disk-shaped)
        selem = morphology.disk(width // 2)
        temp = morphology.dilation(temp, selem)
    
    # Set label values on the target image
    image[temp] = value


def mask_to_smooth_convex(mask, smoothing_sigma=2.0, closing_radius=0):
    """
    Convert a binary mask to a smooth convex shape.

    Parameters:
        mask (ndarray): Binary input mask.
        smoothing_sigma (float): Sigma for Gaussian smoothing (0 = no smoothing).
        closing_radius (int): Radius for morphological closing (0 = no closing).

    Returns:
        smooth_convex_mask (ndarray): Binary mask of smooth convex shape.
    """
    # Ensure mask is boolean
    mask = mask.astype(bool)

    # Step 1: Convex hull
    convex = convex_hull_image(mask)

    # Step 2a: Morphological closing (optional)
    if closing_radius > 0:
        convex = binary_closing(convex, disk(closing_radius))

    # Step 2b: Gaussian smoothing and thresholding (optional)
    if smoothing_sigma > 0:
        smoothed = gaussian_filter(convex.astype(float), sigma=smoothing_sigma)
        convex = smoothed > 0.5  # Binarize

    return convex.astype(np.uint8)


def mask_to_scribble_image_skimage(mask, epsilon=1.0, scribble_thickness=1):
    """
    Convert a binary mask into a scribble annotation image using scikit-image.
    :param mask: Binary mask (H x W)
    :param epsilon: Contour simplification parameter
    :param scribble_thickness: Thickness of the scribble lines
    :return: Scribble annotation image
             (background scribble = 0,
              foreground scribble = 1,
              unlabeled region = 4)
    """
    # Ensure the mask is binary
    if mask.dtype != bool:
        mask = mask > 0
    
    # Create annotation image (initialized as unlabeled region with value 4)
    scribble_img = np.full(mask.shape, 4, dtype=np.uint8)
    
    # Labels guidance
    maxs, mins = find2Dlabel_boundaries(mask)
    min_disk = min(mask.shape[-1]-maxs, mins)
    min_radius = (maxs - mins) // 8
    # Process background (invert the mask)
    if min_disk > 20:
        bg_mask = ~morphology.dilation(mask, morphology.disk(random.randint(20, min_disk)))
    else:
        bg_mask = ~morphology.dilation(mask, morphology.disk(random.randint(5, 20)))
    # Find background contours
    bg_contours = measure.find_contours(bg_mask, 0.5)
    # Simplify background contours
    bg_simplified = [measure.approximate_polygon(contour, epsilon) for contour in bg_contours]
    
    # Draw background scribbles (labeled as 0)
    draw_polygons_with_width(scribble_img, bg_simplified, 0, width=scribble_thickness)
    # Remove overlapping regions
    scribble_img[mask] = 4
    
    # Process foreground
    # Find foreground contours
    if min_radius > 20:
        mask = morphology.erosion(mask, morphology.disk(random.randint(10, min_radius)))

    fg_contours = measure.find_contours(mask, 0.5)
    # Simplify foreground contours
    fg_simplified = [measure.approximate_polygon(contour, epsilon) for contour in fg_contours]
    
    # Draw foreground scribbles (labeled as 1)
    draw_polygons_with_width(scribble_img, fg_simplified, 1, width=scribble_thickness)
    
    return scribble_img


if __name__ == "__main__":

    seed()
    mask_path = 'YOUR_DATA_PATH/masks'
    save_path = 'YOUR_DATA_PATH/scribbles'
    visual_path = 'YOUR_DATA_PATH/visualization'
    ensure_dir(save_path)
    ensure_dir(visual_path)
    mask_files = [os.path.join(mask_path, x) for x in os.listdir(mask_path)]
    print(mask_files)
    for i in tqdm(range(len(mask_files))):
        mask_file = mask_files[i]
        case_name = mask_file.split('/')[-1]
        mask = io.imread(mask_file).astype('uint8')
        convex = convex_hull_image(mask)
        # Generate scribble image
        scribble_img = mask_to_scribble_image_skimage(mask, epsilon=2.0, scribble_thickness=2)
        io.imsave(os.path.join(save_path, case_name), scribble_img)
        if len(np.unique(scribble_img)) < 3:
            print(i, len(mask_files), case_name, np.unique(scribble_img))
            print(i, len(mask_files), case_name, np.unique(scribble_img))
            print(i, len(mask_files), case_name, np.unique(scribble_img))
            print(i, len(mask_files), case_name, np.unique(scribble_img))
            print(i, len(mask_files), case_name, np.unique(scribble_img))
        # Visualization
        plt.figure(figsize=(12, 6))
        plt.subplot(121)
        plt.imshow(mask, cmap='gray')
        plt.title('Original Mask')
        plt.axis('off')
        plt.subplot(122)
        plt.imshow(scribble_img, cmap='viridis')
        plt.title('Scribble Annotation (Values)')
        plt.axis('off')
        plt.savefig(os.path.join(visual_path, case_name))
        plt.close()
    
    
    
    