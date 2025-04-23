import requests
from bs4 import BeautifulSoup
import time
from typing import List, Dict, Optional


def fetch_product_info(base_url: str) -> Optional[List[Dict[str, object]]]:
    """
    Scrapes Macy's website for product image and product URLs based on a search URL.

    Args:
        base_url: The full Macy's search URL, e.g.
                  "https://www.macys.com/shop/search?keyword=white+blouse+solid+Men"

    Returns:
        A list of dictionaries with keys 'image_url' and 'product_url'.
        Returns an empty list if no products are found.
        Returns None on network or parsing errors.
    """
    headers = {
        'user-agent': (
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
            'AppleWebKit/537.36 (KHTML, like Gecko) '
            'Chrome/92.0.4515.131 Safari/537.36'
        )
    }

    try:
        resp = requests.get(base_url, headers=headers, timeout=10)
        resp.raise_for_status()
    except requests.RequestException as e:
        print(f"[fetch_product_info] network error: {e}")
        return None

    soup = BeautifulSoup(resp.text, "html.parser")
    cards = soup.find_all("div", class_="product-thumbnail-container vertical-alignment")
    results: List[Dict[str, object]] = []

    for card in cards[:10]:
        # 1) the product link
        a = card.find("a", href=True)
        if not a:
            continue
        href = a["href"]
        product_url = href if href.startswith("http") else "https://www.macys.com" + href

        # 2) gather _all_ slideshow images
        images = []
        for img in card.find_all("img", class_="picture-image"):
            # prefer src if available, else data-src
            url = img.get("src") or img.get("data-src")
            if url and url not in images:
                images.append(url)

        # if you want at least one image
        if not images:
            continue

        results.append({
            "product_url": product_url,
            "image_urls": images
        })

    time.sleep(2)
    return results


if __name__ == '__main__':
    # quick sanity check
    test_url = (
        'https://www.macys.com/shop/search?keyword=white+blouse+solid+Men'
    )
    result = fetch_product_info(test_url)
    if result is None:
        print("Fetch error")
    elif not result:
        print("No products found")
    else:
        for prod in result:
            print(f"Image: {prod['image_urls']} | URL: {prod['product_url']}")
