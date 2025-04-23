import streamlit as st
import boto3
import botocore.exceptions
from io import BytesIO
from dotenv import load_dotenv
load_dotenv()
import os
import clip
import torch
from PIL import Image
import math
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List
import base64
import openai
from sqlalchemy import create_engine, text
from prompt import tagger
#-------------------------------------------------------------------------#

bucket_name        = os.getenv("AWS_BUCKET_NAME")
aws_access_key_id  = os.getenv("AWS_ACCESS_KEY_ID")
aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY")
region_name        = os.getenv("AWS_REGION_NAME")

# S3 client setup
s3_client = boto3.client(
    service_name="s3",
    aws_access_key_id=aws_access_key_id,
    aws_secret_access_key=aws_secret_access_key,
    region_name=region_name
)

# Streamlit app
st.title("S3 Image Viewer and Editor")

# Folder name input
folder_name = "Wardrobe"
# Function to list images in the specified folder
def list_images_in_folder(folder_name):
    try:
        objects = s3_client.list_objects_v2(Bucket=bucket_name, Prefix=folder_name)['Contents']
        image_urls = [f"https://{bucket_name}.s3.{region_name}.amazonaws.com/{obj['Key']}" for obj in objects]
        return image_urls
    except botocore.exceptions.NoCredentialsError:
        st.error("AWS credentials not available.")
        return []
    except Exception as e:
        st.error(f"Error listing images in folder: {e}")
        return []


#-------------------------------------------------------------------------#

# Display images in the specified folder
# image_urls = list_images_in_folder(folder_name)
# if image_urls:
#     for image_url in image_urls:
#         st.image(image_url, caption=image_url, use_column_width=True)

# Option to delete an image
# image_to_delete = st.text_input("Enter the name of the image to delete (e.g., image.jpg)")
# if st.button("Delete Image"):
#     try:
#         s3_client.delete_object(Bucket=bucket_name, Key=f"{folder_name}/{image_to_delete}")
#         st.success(f"Image {image_to_delete} deleted successfully!")
#     except botocore.exceptions.NoCredentialsError:
#         st.error("AWS credentials not available.")
#     except Exception as e:
#         st.error(f"Error deleting image: {e}")

#-------------------------------------------------------------------------#

# Option to add an image
uploaded_file = st.file_uploader("Choose an image file to add", type=["jpg", "jpeg", "png"])
if uploaded_file:
    if st.button("Add Image"):
        try:
            object_key = f"{folder_name}/{uploaded_file.name}"
            s3_client.upload_fileobj(uploaded_file, bucket_name, object_key)
            st.success(f"Image {uploaded_file.name} added successfully!")
        except botocore.exceptions.NoCredentialsError:
            st.error("AWS credentials not available.")
        except Exception as e:
            st.error(f"Error adding image: {e}")

#-------------------------------------------------------------------------#

# Load the open CLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# Function that computes the feature vectors for a batch of images
def compute_clip_features(images: List[Image.Image]) -> np.ndarray:
    """
    Given a list of PIL.Image objects, returns an (N,512) numpy array of normalized features.
    """
    inputs = torch.stack([preprocess(img) for img in images]).to(device)
    with torch.no_grad():
        feats = model.encode_image(inputs)
        feats /= feats.norm(dim=-1, keepdim=True)
    return feats.cpu().numpy()

#-------------------------------------------------------------------------#


# ── Step 1: List & download all images under "Wardrobe/" in S3 ───────────────────

prefix = "Wardrobe/"
resp = s3_client.list_objects_v2(Bucket=bucket_name, Prefix=prefix)

# Filter to image files
keys = [
    obj["Key"]
    for obj in resp.get("Contents", [])
    if obj["Key"].lower().endswith((".jpg", ".jpeg", ".png"))
]

images: List[Image.Image] = []
image_ids: List[str]       = []

for key in keys:
    body = s3_client.get_object(Bucket=bucket_name, Key=key)["Body"].read()
    img  = Image.open(BytesIO(body)).convert("RGB")
    images.append(img)
    # use filename (sans extension) as your ID
    image_ids.append(os.path.splitext(os.path.basename(key))[0])

# ── Step 2: Compute the CLIP features ───────────────────────────────────────────

features_array = compute_clip_features(images)  # shape: (N,512)

# ── Step 3: Serialize & upload feature.npy ────────────────────────────────────

buf = BytesIO()
np.save(buf, features_array, allow_pickle=False)
buf.seek(0)

s3_client.put_object(
    Bucket     = bucket_name,
    Key        = "features/feature.npy",
    Body       = buf.getvalue(),
    ContentType= "application/octet-stream"
)

# ── Step 4: Serialize & upload image_ids.csv ──────────────────────────────────

ids_buf = BytesIO()
pd.DataFrame({"image_id": image_ids}).to_csv(ids_buf, index=False)
ids_buf.seek(0)

s3_client.put_object(
    Bucket     = bucket_name,
    Key        = "features/image_ids.csv",
    Body       = ids_buf.getvalue(),
    ContentType= "text/csv"
)

