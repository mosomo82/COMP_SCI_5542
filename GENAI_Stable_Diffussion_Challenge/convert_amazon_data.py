import json
import re
from pathlib import Path

# Paths
INPUT_JSONL = r"c:\Users\mtuan\Downloads\meta_Electronics.json\meta_Electronics.json"
OUTPUT_JSON = r"data\products.json"

def extract_attributes(item):
    """Attempt to find color, material, and style from the text."""
    text = " ".join(item.get("feature", [])) + " " + " ".join(item.get("description", [])) + " " + item.get("title", "")
    text = text.lower()
    
    # Common colors and materials to look for
    colors = ['black', 'white', 'red', 'blue', 'green', 'silver', 'grey', 'gray', 'gold', 'pink', 'yellow']
    materials = ['plastic', 'aluminum', 'metal', 'leather', 'silicone', 'steel', 'glass', 'copper']
    
    found_color = "Black" # Default
    for c in colors:
        if c in text:
            found_color = c.capitalize()
            break
            
    found_material = "Mixed Materials" # Default
    for m in materials:
        if m in text:
            found_material = m.capitalize()
            break
            
    # Simple heuristic for style
    style = "modern, functional, electronics"
    if "premium" in text:
        style = "premium, sleek, minimal"
    elif "compact" in text:
        style = "compact, portable, electronics"
        
    return found_color, found_material, style

def main():
    print(f"Reading from: {INPUT_JSONL}")
    
    selected_products = []
    
    try:
        with open(INPUT_JSONL, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue
                    
                item = json.loads(line)
                
                # Filter out books, software, video courses (Amazon electronics has some junk)
                title = item.get("title", "")
                main_cat = item.get("main_cat", "")
                
                is_junk = any(x in title.lower() for x in ['cd rom', 'hardcover', 'software', 'download', 'warranty', 'dvd', 'novel', 'story'])
                if is_junk or main_cat in ["Software", "Books"] or item.get("category", []) and "Books" in item.get("category", []):
                    continue
                
                # Only grab items that have an ASIN and aren't completely empty
                if "asin" in item and len(title) > 10:
                    color, material, style = extract_attributes(item)
                    
                    # Some tricky entries don't have main_cat set properly, check category list
                    fallback_cat = "Electronics"
                    if isinstance(item.get("category"), list) and item["category"]:
                        fallback_cat = item["category"][-1] # Usually the most specific subcategory
                    
                    category = main_cat if len(main_cat) > 2 else fallback_cat
                    if category == "":
                        category = "Electronics"
                        
                    product = {
                        "id": item["asin"],
                        "title": title[:60], # Keep title reasonable length for prompts
                        "category": category.replace("&amp;", "&"),
                        "color": color,
                        "material": material,
                        "style": style
                    }
                    
                    selected_products.append(product)
                    
                if len(selected_products) >= 10:
                    break
    except FileNotFoundError:
        print("Dataset not found!")
        return

    # Write output
    Path(OUTPUT_JSON).parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_JSON, "w", encoding="utf-8") as out_f:
        json.dump(selected_products, out_f, indent=2)
        
    print(f"\n✅ Created {OUTPUT_JSON} with 10 real Amazon products:")
    for p in selected_products:
        print(f" - [{p['id']}] {p['title'][:40]} | {p['color']} | {p['material']}")

if __name__ == "__main__":
    main()
