import pandas as pd
import os
import re

# 1. SETUP: Define paths and regex patterns
input_folder = './data'  # Update this to your source folder
listing_pattern = re.compile(r"Listing", re.I)
sold_pattern = re.compile(r"Sold", re.I)

# Containers for DataFrames
listing_frames = []
sold_frames = []

# Audit variables to track row counts from individual files
source_listing_total = 0
source_sold_total = 0

# 2. FILE PROCESSING LOOP
# Iterate through files in the directory
for filename in sorted(os.listdir(input_folder)):
    if not filename.lower().endswith(".csv"):
        continue

    file_path = os.path.join(input_folder, filename)

    if listing_pattern.search(filename):
        df = pd.read_csv(file_path, low_memory=False)
        source_listing_total += len(df)
        listing_frames.append(df)
    elif sold_pattern.search(filename):
        df = pd.read_csv(file_path, low_memory=False)
        source_sold_total += len(df)
        sold_frames.append(df)
    else:
        print(f"Skipping file without listing/sold marker: {filename}")

if not listing_frames and not sold_frames:
    raise RuntimeError(f"No listing or sold files found in {input_folder}")

# 3. CONCATENATION & AUDIT 1
# Concatenate lists into master DataFrames
all_listings = pd.concat(listing_frames, ignore_index=True) if listing_frames else pd.DataFrame()
all_sold = pd.concat(sold_frames, ignore_index=True) if sold_frames else pd.DataFrame()

# COMMENT: Confirming row counts before and after concatenation
# Listing: Source Total = 853698 | Combined Total = 853698
# Sold:    Source Total = 591768 | Combined Total = 591768
print(f"Listings Concatenation Check: {'Pass' if source_listing_total == len(all_listings) else 'Fail'}")
print(f"Sold Concatenation Check: {'Pass' if source_sold_total == len(all_sold) else 'Fail'}")

# 4. FILTERING & AUDIT 2
# Capture counts before the Residential filter
listings_pre_filter = len(all_listings)
sold_pre_filter = len(all_sold)
print(f"Listings Concatenated: {listings_pre_filter}")
print(f"Sold Concatenated: {sold_pre_filter}")

if not all_listings.empty and 'PropertyType' in all_listings.columns:
    residential_listings = all_listings[
        all_listings['PropertyType'].astype(str).str.contains('Residential', case=False, na=False)
    ]
    print(f"Residential Listings after filter: {len(residential_listings)}")
else:
    print("PropertyType column not found in listing data; skipping Residential filter for listings.")

if not all_sold.empty and 'PropertyType' in all_sold.columns:
    residential_sold = all_sold[
        all_sold['PropertyType'].astype(str).str.contains('Residential', case=False, na=False)
    ]   
    print(f"Residential Sold after filter: {len(residential_sold)}")
else:
    print("PropertyType column not found in sold data; skipping Residential filter for sold.")

# 5. EXPORT
all_listings.to_csv('combined_listings.csv', index=False)
all_sold.to_csv('combined_sold.csv', index=False)

print("Processing complete. Files saved.")