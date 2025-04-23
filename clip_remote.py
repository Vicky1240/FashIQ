# clip_remote.py

from dotenv import load_dotenv
load_dotenv()                  # 1) read .env into os.environ

import os
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from pinecone import Pinecone,ServerlessSpec
import torch, clip, numpy as np, pandas as pd
from io import BytesIO
import boto3
import time
from snowflake_list import annotations_list
from macys_items import fetch_product_info as scrape_products

app = FastAPI()

p = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = p.Index(os.getenv("PINECONE_INDEX_NAME"))

# ─── Pydantic models ────────────────────────────────────────────────────────────

class ImageSearchRequest(BaseModel):
    query: str

class ImageSearchResponse(BaseModel):
    image_ids: list[str]

class ProductInfoRequest(BaseModel):
    base_url: str

class ProductInfoResponse(BaseModel):
    products: list[dict]

# ─── Load CLIP model once ───────────────────────────────────────────────────────

device = "cuda" if torch.cuda.is_available() else "cpu"
model, _ = clip.load("ViT-B/32", device=device)

# ─── S3 client ─────────────────────────────────────────────────────────────────

bucket_name = os.getenv("AWS_BUCKET_NAME")
service_name = os.getenv("AWS_SERVICE_NAME")
aws_access_key_id     = os.getenv("AWS_ACCESS_KEY_ID")
aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY")
region_name           = os.getenv("AWS_REGION_NAME")

s3_client = boto3.client(
    service_name=service_name,
    aws_access_key_id=aws_access_key_id,
    aws_secret_access_key=aws_secret_access_key,
    region_name=region_name
)

# ─── Snowflake tags ────────────────────────────────────────────────────────────

wardrobe_list = annotations_list(os.getenv("SNOWFLAKE_URL"))

# ─── Feature loader ────────────────────────────────────────────────────────────

def search(query: str, k: int = 2) -> list[str]:
    # 1) Encode your text prompt into a 512‐d vector
    with torch.no_grad():
        txt = clip.tokenize([query]).to(device)
        feats = model.encode_text(txt)
        feats = feats / feats.norm(dim=-1, keepdim=True)
        vector = feats.cpu().numpy()[0].tolist()

    # 2) Query Pinecone
    resp = index.query(
        vector=vector,
        top_k=k,
        include_values=False   # we only need the IDs
    )

    # 3) Extract and return the matched IDs
    return [match.id for match in resp.matches]


# ─── Endpoints ─────────────────────────────────────────────────────────────────

@app.post("/image-search", response_model=ImageSearchResponse)
def image_search(req: ImageSearchRequest):
    try:
        results = search(req.query)
        return {"image_ids": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/get_product_info", response_model=ProductInfoResponse)
def get_product_info(req: ProductInfoRequest):
    try:
        products = scrape_products(req.base_url)
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e))

    # None = real error, [] = no hits
    if products is None:
        raise HTTPException(status_code=502, detail="Failed to scrape products")
    return {"products": products}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
