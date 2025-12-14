import torch
import numpy as np
import pandas as pd
from torchvision.io import decode_image
from tqdm import tqdm
from torchvision.io import decode_image, ImageReadMode


def export_hf(
    model,
    processor,
    test_df,
    test_images_base_path,
    device,
    batch_size,
    num_workers=0,   # kept for signature compatibility
    pin_memory=False,
    output_path="submission.csv",
    kaggle_labels_start_at_1=True,
):
    """
    Export Kaggle submission for HuggingFace image classification models.
    Based DIRECTLY on your working notebook code.
    """

    model.to(device)
    model.eval()

    image_paths = test_df["image_path"].values

    all_preds = []

    with torch.no_grad():
        for i in tqdm(range(0, len(image_paths), batch_size), desc="Exporting HF"):
            batch_paths = image_paths[i:i + batch_size]

            images = [
                decode_image(test_images_base_path + p, mode=ImageReadMode.RGB)
                for p in batch_paths
            ]

            inputs = processor(images, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}

            outputs = model(**inputs)
            preds = outputs.logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)

    all_preds = np.array(all_preds)

    if kaggle_labels_start_at_1:
        all_preds = all_preds + 1

    submission = pd.DataFrame({
        "id": test_df["id"].values,
        "label": all_preds
    })

    submission.to_csv(output_path, index=False)
    print(f"HF submission saved to {output_path}")

def export_model(
    model,
    test_df,
    test_images_base_path,
    transform,
    device,
    batch_size,
    output_path="submission.csv",
    kaggle_labels_start_at_1=True,
):
    """
    Export Kaggle submission for your custom PyTorch CNN models.
    Mirrors the HF export logic.
    """

    model.to(device)
    model.eval()

    image_paths = test_df["image_path"].values
    all_preds = []

    with torch.no_grad():
        for i in tqdm(range(0, len(image_paths), batch_size), desc="Exporting CNN"):
            batch_paths = image_paths[i:i + batch_size]

            images = []
            for p in batch_paths:
                img = decode_image(test_images_base_path + p, mode=ImageReadMode.RGB)
                img = transform(img)
                images.append(img)

            images = torch.stack(images).to(device)

            logits = model(images)
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)

    all_preds = np.array(all_preds)

    if kaggle_labels_start_at_1:
        all_preds = all_preds + 1

    submission = pd.DataFrame({
        "id": test_df["id"].values,
        "label": all_preds
    })

    submission.to_csv(output_path, index=False)
    print(f"Model submission saved to {output_path}")