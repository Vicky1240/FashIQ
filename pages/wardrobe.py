import boto3
import streamlit as st
import botocore.exceptions
from dotenv import load_dotenv
load_dotenv()
import os
from io import BytesIO
from PIL import Image
import botocore
import clip
import torch
from typing import List
import numpy as np
import pandas as pd
import base64, json, time
from openai import OpenAI
from sqlalchemy import create_engine, text
from prompt import tagger    
from pinecone import Pinecone,ServerlessSpec

# Credentials
bucket_name        = os.getenv("AWS_BUCKET_NAME")
service_name       = os.getenv("AWS_SERVICE_NAME")
aws_access_key_id  = os.getenv("AWS_ACCESS_KEY_ID")
aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY")
region_name        = os.getenv("AWS_REGION_NAME")

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
SNOWFLAKE_URL   = os.getenv("SNOWFLAKE_URL")

p = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = p.Index(os.getenv("PINECONE_INDEX_NAME"))

# S3 client setup
s3_client = boto3.client(
    service_name=service_name,
    aws_access_key_id=aws_access_key_id,
    aws_secret_access_key=aws_secret_access_key,
    region_name=region_name
)

# Function to display wardrobe images
def display_wardrobe(all_image_ids):
    
    if all_image_ids == None:
        return None
    # Define the number of columns
    num_columns = 3

    # Create columns for displaying images
    columns = st.columns(num_columns)

    # Iterate through all images and display them in columns
    for i, image_id in enumerate(all_image_ids):
        img = s3_client.get_object(Bucket=bucket_name, Key=image_id)
        image_bytes = img['Body'].read()

        # Determine the column to display the image in
        current_column = columns[i % num_columns]

        # Display the image in the current column
        current_column.image(image_bytes, caption=f'Image: {image_id}', use_column_width=True)

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


def clear_screen():
    st.image([])

# Streamlit app
def main():
    # Title
    st.title("My Wardrobe")

    col1,col2,col3 = st.columns(3)
    
    with col1:
        # Select box to trigger wardrobe display
        selected_option = st.selectbox("Select an option", ["Show My Wardrobe","Add Image into Wardrobe"])
    

    # List all objects in the specified folder
    folder_path = 'Wardrobe/'
    response = s3_client.list_objects_v2(Bucket=bucket_name, Prefix=folder_path)

    # Extract object keys from the response
    all_image_ids = [obj['Key'] for obj in response.get('Contents', [])]

    
    # Check if the select box option is chosen
    if selected_option == "Show My Wardrobe":
        with col2:
            st.write("")
            st.write("")
            fetch_button = st.button("fetch")

        # Display the wardrobe images
        if fetch_button:
            display_wardrobe(all_image_ids)
        else:
            display_wardrobe(None)
    else:
        st.header("Add your Images!")
        uploaded_file = st.file_uploader("Choose an image file to add", type=["jpg", "jpeg", "png"])
        if uploaded_file and st.button("Add Image"):
            try:
                # 1) Read bytes from the upload
                file_bytes = uploaded_file.read()
                #new_id = os.path.splitext(uploaded_file.name)[0]
                new_id = uploaded_file.name
                object_key = f"{folder_name}/{uploaded_file.name}"

                # 2) Upload the raw image
                s3_client.put_object(
                    Bucket=bucket_name,
                    Key=object_key,
                    Body=file_bytes,
                    ContentType=uploaded_file.type
                )

                # 3) Compute CLIP features for this image
                img = Image.open(BytesIO(file_bytes)).convert("RGB")
                new_feats = compute_clip_features([img])  # (1,512) numpy array

                # ─── UPDATE features/feature.npy ──────────────────────────────────────
                feats_key = "Features/feature.npy"
                try:
                    resp = s3_client.get_object(Bucket=bucket_name, Key=feats_key)
                    buf = BytesIO(resp["Body"].read())
                    existing_feats = np.load(buf)
                    updated_feats = np.vstack([existing_feats, new_feats])
                except botocore.exceptions.ClientError as e:
                    code = e.response["Error"]["Code"]
                    if code in ("NoSuchKey", "404"):
                        updated_feats = new_feats
                    else:
                        raise

                out_buf = BytesIO()
                np.save(out_buf, updated_feats, allow_pickle=False)
                out_buf.seek(0)
                s3_client.put_object(
                    Bucket=bucket_name,
                    Key=feats_key,
                    Body=out_buf.getvalue(),
                    ContentType="application/octet-stream"
                )
                # ─── UPDATE features index in Pinecone ──────────────────────────────────────
                vec = new_feats[0].tolist()  # extract the single vector
                index.upsert([(new_id, vec)])

                # ─── UPDATE features/image_ids.csv ────────────────────────────────────
                ids_key = "Features/image_ids.csv"
                try:
                    resp = s3_client.get_object(Bucket=bucket_name, Key=ids_key)
                    ids_buf = BytesIO(resp["Body"].read())
                    df = pd.read_csv(ids_buf)
                    df = pd.concat([df, pd.DataFrame({"image_id":[new_id]})], ignore_index=True)
                except botocore.exceptions.ClientError as e:
                    code = e.response["Error"]["Code"]
                    if code in ("NoSuchKey", "404"):
                        df = pd.DataFrame({"image_id":[new_id]})
                    else:
                        raise

                out_ids = BytesIO()
                df.to_csv(out_ids, index=False)
                out_ids.seek(0)
                s3_client.put_object(
                    Bucket=bucket_name,
                    Key=ids_key,
                    Body=out_ids.getvalue(),
                    ContentType="text/csv"
                )

                # ─── 4) Annotate with GPT‑4 Vision ────────────────────────────────────────
                # Base64‑encode the image
                b64 = base64.b64encode(file_bytes).decode("utf-8")
                messages = [{
                    "role": "user",
                    "content": [
                        {"type":"text", "text": tagger},
                        {"type":"image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}
                    ]
                }]

                resp = client.chat.completions.create(
                    model="gpt-4o",
                    messages=messages,
                    max_tokens=1000
                )
                raw = resp.choices[0].message.content
                # strip ```json blocks if present
                if raw.strip().startswith("```"):
                    raw = raw.strip().strip("```json").strip("```")
                annotations = json.loads(raw)["annotations"]

                # ─── 5) Upsert into Snowflake ────────────────────────────────────────────
                engine = create_engine(SNOWFLAKE_URL)
                upsert_sql = """
                MERGE INTO WARDROBE_TAGS t
                USING (
                SELECT :image AS IMAGE,
                        PARSE_JSON(:ann) AS ANNOTATIONS
                ) v
                ON t.IMAGE = v.IMAGE
                WHEN MATCHED THEN
                UPDATE SET t.ANNOTATIONS = v.ANNOTATIONS
                WHEN NOT MATCHED THEN
                INSERT (IMAGE, ANNOTATIONS)
                    VALUES (v.IMAGE, v.ANNOTATIONS)
                """
                with engine.begin() as conn:
                    conn.execute(text(upsert_sql),
                                {"image": object_key,
                                "ann": json.dumps({"annotations": annotations})})

                st.success(f"✅ Added '{uploaded_file.name}', features updated, and annotations stored!")

            except Exception as e:
                st.error(f"Error adding image or updating index/annotations: {e}")


    

# Run the Streamlit app
if __name__ == "__main__":
    main()
