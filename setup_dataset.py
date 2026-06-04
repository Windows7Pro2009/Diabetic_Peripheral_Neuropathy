import os
import shutil
import random
import glob
import zipfile

def main():
    print("Step 1: Downloading dataset from Kaggle...")
    raw_dir = "data_raw"
    os.makedirs(raw_dir, exist_ok=True)
    
    # We use the python kaggle api directly
    try:
        import kaggle
        print("Imported kaggle module. Downloading 'laithjj/diabetic-foot-ulcer-dfu'...")
        kaggle.api.dataset_download_files('laithjj/diabetic-foot-ulcer-dfu', path=raw_dir, unzip=True)
        print("Download and extraction completed successfully!")
    except Exception as e:
        print(f"Error downloading via Kaggle Python API: {e}")
        print("Attempting to run kaggle command line tool as fallback...")
        import subprocess
        # Fallback to subprocess using the .venv kaggle executable
        kaggle_exe = os.path.join(".venv", "Scripts", "kaggle.exe")
        if not os.path.exists(kaggle_exe):
            kaggle_exe = "kaggle"
        cmd = [kaggle_exe, "datasets", "download", "-d", "laithjj/diabetic-foot-ulcer-dfu", "-p", raw_dir, "--unzip"]
        print(f"Running command: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            print("Download and extraction completed successfully via fallback!")
        else:
            print(f"Fallback download failed:\nSTDOUT: {result.stdout}\nSTDERR: {result.stderr}")
            return

    # Check downloaded contents
    print("\nStep 2: Checking downloaded raw files...")
    # Recursively list folders to find the patches or original images
    all_dirs = []
    for root, dirs, files in os.walk(raw_dir):
        for d in dirs:
            all_dirs.append(os.path.join(root, d))
            
    print(f"Found {len(all_dirs)} folders in {raw_dir}:")
    for d in all_dirs:
        print(f"  - {d}")

    # Locate normal and abnormal folders
    # We prefer the 'Patches' directory as they are 224x224 cropped regions, ideal for MobileNetV2
    normal_dirs = []
    abnormal_dirs = []
    
    # Try searching specifically in directories containing 'patches' or 'Patches' first
    patches_dirs = [d for d in all_dirs if 'patches' in d.lower()]
    search_dirs = patches_dirs if patches_dirs else all_dirs
    
    for d in search_dirs:
        d_lower = os.path.basename(d).lower()
        if 'abnormal' in d_lower or 'ulcer' in d_lower or 'neuropathy' in d_lower:
            abnormal_dirs.append(d)
        elif 'normal' in d_lower or 'healthy' in d_lower:
            normal_dirs.append(d)

    if not normal_dirs or not abnormal_dirs:
        # Fallback to search all folders if patches filter didn't work or yielded nothing
        print("Patches folder search didn't find both classes. Searching all directories...")
        normal_dirs = []
        abnormal_dirs = []
        for d in all_dirs:
            d_lower = os.path.basename(d).lower()
            if 'abnormal' in d_lower or 'ulcer' in d_lower or 'neuropathy' in d_lower:
                abnormal_dirs.append(d)
            elif 'normal' in d_lower or 'healthy' in d_lower:
                normal_dirs.append(d)

    print(f"\nNormal directories identified: {normal_dirs}")
    print(f"Abnormal/Neuropathy directories identified: {abnormal_dirs}")

    if not normal_dirs or not abnormal_dirs:
        print("CRITICAL ERROR: Could not locate both normal and abnormal folders in the dataset.")
        return

    # Gather all images
    def get_images(dirs_list):
        images = []
        exts = ('*.jpg', '*.jpeg', '*.png', '*.webp', '*.JPG', '*.JPEG', '*.PNG', '*.WEBP')
        for d in dirs_list:
            for ext in exts:
                images.extend(glob.glob(os.path.join(d, ext)))
        return list(set(images)) # unique paths

    normal_images = get_images(normal_dirs)
    abnormal_images = get_images(abnormal_dirs)

    print(f"\nTotal Normal images found: {len(normal_images)}")
    print(f"Total Abnormal/Neuropathy images found: {len(abnormal_images)}")

    if len(normal_images) == 0 or len(abnormal_images) == 0:
        print("CRITICAL ERROR: No images found in normal or abnormal directories.")
        return

    # Target directory setup
    dataset_dir = "dataset"
    print(f"\nStep 3: Cleaning old files in '{dataset_dir}' directory...")
    splits = ["train", "val", "test"]
    classes = ["normal", "neuropathy"]
    
    for split in splits:
        for cls in classes:
            target_path = os.path.join(dataset_dir, split, cls)
            if os.path.exists(target_path):
                # Clean all files inside it
                for f in os.listdir(target_path):
                    file_path = os.path.join(target_path, f)
                    try:
                        if os.path.isfile(file_path) or os.path.islink(file_path):
                            os.unlink(file_path)
                        elif os.path.isdir(file_path):
                            shutil.rmtree(file_path)
                    except Exception as e:
                        print(f"Failed to delete {file_path}. Reason: {e}")
            else:
                os.makedirs(target_path, exist_ok=True)

    # Shuffle and split function
    def split_and_copy(images, class_name, train_ratio=0.8, val_ratio=0.1):
        random.seed(42)
        random.shuffle(images)
        
        n_total = len(images)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)
        
        train_imgs = images[:n_train]
        val_imgs = images[n_train:n_train+n_val]
        test_imgs = images[n_train+n_val:]
        
        split_map = {
            "train": train_imgs,
            "val": val_imgs,
            "test": test_imgs
        }
        
        for split, imgs in split_map.items():
            dest_dir = os.path.join(dataset_dir, split, class_name)
            print(f"Copying {len(imgs)} images to {dest_dir}...")
            for img_path in imgs:
                dest_path = os.path.join(dest_dir, os.path.basename(img_path))
                # If filename collision, prepend index
                if os.path.exists(dest_path):
                    base, ext = os.path.splitext(os.path.basename(img_path))
                    dest_path = os.path.join(dest_dir, f"{base}_alt{ext}")
                shutil.copy2(img_path, dest_path)

    print("\nStep 4: Splitting and copying Normal images...")
    split_and_copy(normal_images, "normal")

    print("\nStep 5: Splitting and copying Neuropathy images...")
    split_and_copy(abnormal_images, "neuropathy")

    # Print final summary counts
    print("\nStep 6: Final dataset counts:")
    for split in splits:
        for cls in classes:
            target_path = os.path.join(dataset_dir, split, cls)
            n_files = len(os.listdir(target_path))
            print(f"  {split}/{cls}: {n_files} files")
            
    print("\nDataset preparation completed successfully!")

if __name__ == "__main__":
    main()
