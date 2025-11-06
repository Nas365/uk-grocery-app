# UK Grocery Recommender

**Where should I buy — Tesco or Sainsbury’s?**  
This project compares like-for-like grocery items across **Tesco** and **Sainsbury’s** and recommends the **cheaper, similar** option. It uses real product listing pages we captured (HTML), turns them into a clean dataset, and powers a FastAPI + web UI with dropdowns for **Category**, **Brand**, and optional **Size (grams)**.

> **Live Demo:** _[Access App here](https://uk-grocery-app.onrender.com)_  
> **Project by:** **Nasir Abubakar**

---

## Why this matters
Prices move quickly and small differences add up. A clear, side-by-side view helps shoppers decide in seconds — especially when both retailers sell the *same* brand or a very similar pack size.

---

## Key Highlights
- **Dataset (real web pages):** Category pages for **pasta, rice, cereals, cooking_oil, tinned_tomatoes** saved from Tesco & Sainsbury’s.  
- **Feature engineering (compact table):**  
  `category, brand, price_gbp_sainsburys, price_gbp_tesco, size_grams` + derived savings metrics.  
  A small name-lookup improves display names per retailer.
- **Recommendation logic:**  
  1) Filter by **category**, optional **brand**, optional **size window** (±20%).  
  2) Keep rows with prices at **both** retailers.  
  3) Compute **cheapest_retailer**, **cheapest_price**, **other_price**, **saving %**.  
  4) Sort by highest saving and return **Top N**.  
- **ML touch:** Upstream text normalization + brand/size standardisation to create consistent “matched pairs.”  
  *(Roadmap: TF-IDF + KMeans to find similar alternatives even when exact matches aren’t present.)*

---

## Data Source & Currency (Price Changes Explained)
- **Source:** Real product listing pages captured from Tesco and Sainsbury’s websites (HTML).  
- **Processing:** Local parsing → cleaning → **`data/processed/features_v1.csv.gz`**.  
- **Scrape date(s):**  **06-11-2025**.  
- **About price changes:** Retail prices change frequently. The app shows prices **from the scrape date**, not live prices.  

> This is an educational/demo project. It is not affiliated with Tesco or Sainsbury’s.

---

## App Walkthrough
1. Choose **Category** (dropdown).  
2. (Optional) Choose **Brand** and **Target Size (grams)** (dropdowns).  
3. Choose **Number of suggestions** (Top N).  
4. Click **Recommend** to see a table with:  
   - Product @ Sainsbury’s  
   - Product @ Tesco  
   - Cheapest retailer  
   - Cheapest price (£) / Other price (£)  
   - Saving (%)

---


