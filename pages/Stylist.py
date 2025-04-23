# pages/Stylist.py

import streamlit as st
import os, json, requests
from openai import OpenAI
from dotenv import load_dotenv
from snowflake_list import annotations_list
from typing import Optional
import boto3
from PIL import Image
from io import BytesIO


load_dotenv()

# FastAPI endpoints
URL_CLIP    = "http://127.0.0.1:8000/image-search"
URL_PRODUCT = "http://127.0.0.1:8000/get_product_info"

try:
    # AWS credentials    
    bucket_name        = os.getenv("AWS_BUCKET_NAME")
    service_name       = os.getenv("AWS_SERVICE_NAME")
    aws_access_key_id  = os.getenv("AWS_ACCESS_KEY_ID")
    aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY")
    region_name        = os.getenv("AWS_REGION_NAME")

    # S3 client setup
    s3_client = boto3.client(
        service_name=service_name,
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        region_name=region_name
    )
except Exception as e:
    st.error(f"Error collecting AWS secrets: {str(e)}")

# OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Fetch wardrobe tags once
wardrobe = annotations_list(os.getenv("SNOWFLAKE_URL"))
wardrobe_str = "\n".join(wardrobe)

role = f"""
You are my personal wardrobe assistant who has immense experience in the fashion industry. You know the latest trends and people appreciate you often for your fashion choices. You are also funny and give the best advice based on the event. Now, these are my wardrobe data:
Now, answer only the questions I ask and suggest only from the wardrobe.\n {wardrobe_str}
Return in the following format:\n.
"Top : (color: ,clothing type: ,pattern:), 
Bottom : (color: ,clothing type:,pattern:) 
Reason:"\n 
Please only give me responses in the specified format. No spelling error in keys. Remember, the choice should match the exact keyword from the wardrobe. Also, try to include a funny thing in the reason based on the occasion. Return the output in json only.
If any questions other than fashion are asked kindly reply in your words you are not able to. 
"""

def interact_with_gpt(question: str) -> Optional[dict]:
    try:
        resp = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role":"system", "content": role},
                {"role":"user",   "content": question}
            ],
            temperature=0.7,
        )
        content = resp.choices[0].message.content
        return json.loads(content)
    except Exception as e:
        st.error(f"GPT error: {e}")
        return None
    
# Function to display images from S3

def display_images_from_s3(image_ids):
    columns = st.columns(2)
    for j, image_id in enumerate(image_ids):
        image_data = s3_client.get_object(Bucket=bucket_name, Key=f"Wardrobe/{image_id}")['Body'].read()
        image = Image.open(BytesIO(image_data))
        columns[j].image(image, caption=f"Image {j+1}")

def main():
    st.title("Personal Wardrobe Stylist")
    q = st.text_input("Describe your event and style:")
    if st.button("Get Recommendations"):
        return q
    return None

if __name__ == "__main__":
    question = main()
    if not question:
        st.info("Enter an event description above and click the button.")
    else:
        recs = interact_with_gpt(question)
        #st.write(recs)
        if not recs:
            st.error("Failed to get outfit from GPT.")
        else:
            # Build Macy’s search URL
            top = recs["Top"]
            bottom = recs["Bottom"]
            gender = "Men"
            for part in ("Top","Bottom"):
                item = recs[part]
                keyword = "+".join([item["color"],item["clothing type"],item["pattern"],gender])
                # 1) fetch similar images via your FastAPI
                img_resp = requests.post(URL_CLIP, json={"query": keyword})
                imgs = img_resp.json()["image_ids"]
                #st.image(imgs, width=100, caption=[f"{part} image"]*len(imgs))
                st.write(f"**{part} options:**")
                display_images_from_s3(imgs)
                # 2) fetch Macy's product links
                prod_resp = requests.post(URL_PRODUCT, json={"base_url":f"https://www.macys.com/shop/search?keyword={keyword}"})
                products = prod_resp.json()["products"]
                if products:
                    for p in products:
                        st.write(f"- [{p['product_url']}]({p['product_url']})")
                else:
                    st.write(f"No {part} found online.")
            st.write("**Suggestion:**", recs["Reason"])
