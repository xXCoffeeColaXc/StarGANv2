import os
import config
import shutil
import random

'''
def update_condition_labels( new_condition_name):
    conditions.append(new_condition_name)

    # Add 0 to the end of each existing condition
    for key in condition_labels:
        condition_labels[key].append(0)
    
    # Add the new condition 'daytime'
    condition_labels[new_condition_name] = [0] * (config.NUM_DOMAINS - 1) + [1]
'''

# TODO remove ref folders for space efficiency

def create_daytime_folder(no_img):
    all_ref_images = []
    # get all ref images
    for condition in ['fog', 'night', 'rain', 'snow']:
        condition_path = os.path.join(config.ACDC_DATASET_DIR, condition)
        for folder in os.listdir(condition_path):
            if folder.endswith('_ref'): 
                folder_path = os.path.join(condition_path, folder)
                for img_folder in os.listdir(folder_path):
                    img_folder_path = os.path.join(folder_path, img_folder)
                    for img_file in os.listdir(img_folder_path):
                        if img_file.endswith('.jpg') or img_file.endswith('.png'):  # Assuming images are .jpg or .png
                            all_ref_images.append(os.path.join(img_folder_path, img_file))

    # Create a new "daytime" directory and copy the selected images there
    daytime_dir = os.path.join(config.ACDC_DATASET_DIR, 'daytime')
    os.makedirs(daytime_dir, exist_ok=True)

    # create train, val, test subfolder in daytime
    for mode in ['train', 'val', 'test']:
        daytime_dir_mode = os.path.join(daytime_dir, mode)
        os.makedirs(daytime_dir_mode, exist_ok=True)

        # create subfolder in daytime/mode for matching original datastructure
        daytime_dir_mode_sub = os.path.join(daytime_dir_mode, "GOPRP000")
        os.makedirs(daytime_dir_mode_sub, exist_ok=True)

        # Randomly select x amount
        if mode == 'train':
            selected_images = random.sample(all_ref_images, no_img) # OVERLAP ???
        elif mode == 'val':
            selected_images = random.sample(all_ref_images, int(no_img*0.1)) # OVERLAP ???
        elif mode == 'test':
            selected_images = random.sample(all_ref_images, int(no_img*0.05)) # OVERLAP ???

        for idx, img_path in enumerate(selected_images, 1):
            shutil.copy(img_path, os.path.join(daytime_dir_mode_sub, f"daytime_{idx}.jpg"))



if __name__ == "__main__":
    create_daytime_folder(400)
