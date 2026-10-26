import os
import sys
from skimage import io
from skimage import transform
from tqdm import tqdm

sep = '\\' if sys.platform[:3] == 'win' else '/'


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


image_path = 'YOUR_DATA_PATH'

sample_list = [y.split('.')[0] for x in os.listdir(image_path) if x.endswith('images')
               for y in os.listdir(os.path.join(image_path, x)) if y.endswith('.jpg')]

save_image_path = os.path.join(image_path, 'images')
save_label_path = os.path.join(image_path, 'masks')
ensure_dir(save_image_path)
ensure_dir(save_label_path)

for item in tqdm(range(len(sample_list))):
    case = sample_list[item]
    image = io.imread(os.path.join(image_path, 'oimages', case + '.jpg'))
    label = io.imread(os.path.join(image_path, 'omasks', case + '_segmentation.png'))
    image = transform.resize(image, output_shape=(256, 256), order=0)
    label = transform.resize(label, output_shape=(256, 256), order=0)
    io.imsave(save_image_path + '/' + case + '.png', image)
    io.imsave(save_label_path + '/' + case + '.png', label)




