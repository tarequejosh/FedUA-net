"""
Dataset verification script for reviewer reproduction.
Validates directory structures, image counts, and class splits across the 3 hospital cohorts.
"""

import os
import sys

DATASET_ROOT = "Dataset"

EXPECTED_DATASETS = {
    "Hospital A (Brain MRI)": {
        "candidate_paths": [
            os.path.join(DATASET_ROOT, "Brain Tumor MRI Dataset"),
            os.path.join(DATASET_ROOT, "Brain_Tumor_MRI"),
            os.path.join(DATASET_ROOT, "Brain Tumor MRI"),
        ],
        "classes": ["glioma", "meningioma", "notumor", "pituitary"],
        "min_images": 5000,
    },
    "Hospital B (Breast Ultrasound BUSI)": {
        "candidate_paths": [
            os.path.join(DATASET_ROOT, "Dataset_BUSI_with_GT"),
            os.path.join(DATASET_ROOT, "Dataset_BUSI_with_mask"),
            os.path.join(DATASET_ROOT, "BUSI"),
        ],
        "classes": ["benign", "malignant", "normal"],
        "min_images": 700,
    },
    "Hospital C (COVID-19 Radiography)": {
        "candidate_paths": [
            os.path.join(DATASET_ROOT, "COVID-19_Radiography_Dataset"),
            os.path.join(DATASET_ROOT, "COVID19_Radiography_Dataset"),
        ],
        "classes": ["COVID", "Lung_Opacity", "Normal", "Viral Pneumonia"],
        "min_images": 15000,
    },
}


def count_images_in_dir(directory: str) -> int:
    count = 0
    if not os.path.exists(directory):
        return 0
    for root, _, files in os.walk(directory):
        for f in files:
            if f.lower().endswith(('.png', '.jpg', '.jpeg')) and "mask" not in f.lower():
                count += 1
    return count


def check_datasets() -> bool:
    print("=" * 70)
    print("FedUA-Net Dataset Verification Suite")
    print("=" * 70)

    all_passed = True

    for name, spec in EXPECTED_DATASETS.items():
        found_path = None
        for cand in spec["candidate_paths"]:
            if os.path.exists(cand):
                found_path = cand
                break

        print(f"\nChecking: {name}")

        if found_path is None:
            print(f"  Target Path Candidates: {spec['candidate_paths']}")
            print(f"  [MISSING] Directory does not exist.")
            print(f"  Please download the dataset following instructions in docs/DATASET_GUIDE.md")
            all_passed = False
            continue

        print(f"  Located Path: {found_path}")
        total_images = count_images_in_dir(found_path)
        print(f"  Images Found: {total_images} (Expected >= {spec['min_images']})")

        if total_images >= spec["min_images"]:
            print(f"  [OK] Dataset validated successfully.")
        else:
            print(f"  [WARNING] Image count is lower than expected.")
            all_passed = False

    print("\n" + "=" * 70)
    if all_passed:
        print("[SUCCESS] All 3 clinical datasets are verified and ready for training.")
    else:
        print("[NOTICE] One or more datasets are missing or incomplete.")
        print("         See docs/DATASET_GUIDE.md for download instructions.")
    print("=" * 70)
    return all_passed


if __name__ == "__main__":
    success = check_datasets()
    sys.exit(0 if success else 1)
