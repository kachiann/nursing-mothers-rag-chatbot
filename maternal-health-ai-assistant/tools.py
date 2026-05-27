import requests
from langchain.tools import tool

USDA_BASE = "https://api.nal.usda.gov/fdc/v1"
# Register free at https://fdc.nal.usda.gov/api-key-signup.html
# DEMO_KEY works but is rate-limited to 30 req/hr per IP
API_KEY = "DEMO_KEY"

# Nutrients most relevant to nursing mothers and infants
NURSING_NUTRIENTS = {
    "Energy",
    "Protein",
    "Calcium, Ca",
    "Iron, Fe",
    "Zinc, Zn",
    "Iodine, I",
    "Vitamin D (D2 + D3)",
    "Vitamin B-12",
    "Folate, total",
    "Choline, total",
    "Fatty acids, total omega-3",
}


def _search_food(food_name: str) -> dict | None:
    """
    Step 1 — POST /foods/search
    Returns the top match: {"fdcId": ..., "description": ..., "dataType": ...}
    Uses POST so the query goes in the body (avoids URL-encoding issues).
    """
    url = f"{USDA_BASE}/foods/search"
    payload = {
        "query": food_name,
        "pageSize": 1,
        # Prefer Foundation > SR Legacy > Survey > Branded for nutrient completeness
        "dataType": ["Foundation", "SR Legacy", "Survey (FNDDS)", "Branded"],
    }
    resp = requests.post(url, json=payload, params={"api_key": API_KEY}, timeout=6)
    resp.raise_for_status()
    foods = resp.json().get("foods", [])
    return foods[0] if foods else None


def _get_food_detail(fdc_id: int) -> dict | None:
    """
    Step 2 — GET /food/{fdcId}
    Returns full nutrient list with fields: nutrientName, unitName, value.
    """
    url = f"{USDA_BASE}/food/{fdc_id}"
    resp = requests.get(url, params={"api_key": API_KEY}, timeout=6)
    resp.raise_for_status()
    return resp.json()


@tool
def lookup_nutrient_info(food_name: str) -> str:
    """Look up nutritional information for a specific food from the USDA
    FoodData Central database. Use this when the user asks about the
    nutritional content, vitamins, minerals, or calories in a specific food —
    e.g. 'how much calcium is in spinach?' or 'is salmon high in omega-3?'
    Input should be a plain food name like 'spinach' or 'salmon fillet'."""
    try:
        # Step 1: search for the food to get its fdcId
        match = _search_food(food_name)
        if not match:
            return f"No food matching '{food_name}' found in the USDA database."

        fdc_id = match["fdcId"]
        food_label = match.get("description", food_name)
        data_type = match.get("dataType", "")

        # Step 2: fetch full detail record for complete nutrient list
        detail = _get_food_detail(fdc_id)
        if not detail:
            return f"Found '{food_label}' (fdcId {fdc_id}) but could not retrieve nutrient details."

        # /food/{fdcId} returns foodNutrients with nested nutrient object:
        # { "nutrient": { "name": "...", "unitName": "..." }, "amount": 0.0 }
        raw_nutrients = detail.get("foodNutrients", [])

        lines = [f"Nutritional info for **{food_label}** (per 100g) [{data_type}]:"]
        found = []
        for entry in raw_nutrients:
            nutrient = entry.get("nutrient", {})
            name = nutrient.get("name", "")
            unit = nutrient.get("unitName", "")
            amount = entry.get("amount")
            if name in NURSING_NUTRIENTS and amount is not None:
                found.append(f"  • {name}: {round(amount, 2)} {unit}")

        if not found:
            return (
                f"Found '{food_label}' but none of the key nursing-related nutrients "
                f"were present in this database entry (data type: {data_type}). "
                f"Try a more specific food name."
            )

        lines.extend(sorted(found))  # alphabetical for readability
        lines.append(f"\nSource: USDA FoodData Central (fdcId {fdc_id})")
        return "\n".join(lines)

    except requests.Timeout:
        return "The USDA API timed out. Please try again in a moment."
    except requests.HTTPError as e:
        return f"USDA API error ({e.response.status_code}): {e.response.text[:200]}"
    except requests.RequestException as e:
        return f"Could not reach the USDA API: {str(e)}"


@tool
def check_breastfeeding_guideline(topic: str) -> str:
    """Look up evidence-based breastfeeding or infant care guidelines on a
    specific topic. Use this when the user asks about safety, duration,
    storage, medications, or official recommendations — e.g. 'is it safe to
    drink alcohol while breastfeeding?' or 'how long should I breastfeed?'
    Input should be a short topic keyword like 'alcohol', 'duration', or
    'vitamin d'."""
    guidelines = {
        "duration": (
            "WHO recommends exclusive breastfeeding for the first 6 months, "
            "then continued breastfeeding alongside complementary foods up to 2 years or beyond. "
            "The AAP (updated 2022) recommends breastfeeding for at least 2 years."
        ),
        "alcohol": (
            "CDC: Alcohol passes into breast milk at levels similar to blood alcohol. "
            "If you drink, wait at least 2 hours per standard drink before nursing. "
            "Pumping and dumping does NOT speed clearance — only time does. "
            "Occasional moderate drinking (1 standard drink) is generally considered acceptable."
        ),
        "caffeine": (
            "Up to 300mg of caffeine per day (~2–3 cups of coffee) is considered safe "
            "while breastfeeding (NHS, CDC). Excess caffeine may cause infant irritability "
            "or disrupted sleep. Caffeine does pass into breast milk but at low levels."
        ),
        "vitamin d": (
            "AAP recommends 400 IU of Vitamin D daily for all breastfed infants, "
            "starting within the first few days of life. Breast milk alone typically "
            "provides insufficient Vitamin D regardless of maternal intake."
        ),
        "storage": (
            "CDC breast milk storage guidelines: "
            "room temperature (≤77°F/25°C) — up to 4 hours; "
            "refrigerator (40°F/4°C) — up to 4 days; "
            "freezer (0°F/−18°C) — 6 months ideal, up to 12 months acceptable. "
            "Use insulated cooler with ice packs for up to 24 hours when travelling. "
            "Never refreeze thawed milk."
        ),
        "medication": (
            "Most medications are compatible with breastfeeding, but always verify "
            "with a healthcare provider. The NIH LactMed database "
            "(https://www.ncbi.nlm.nih.gov/books/NBK501922/) is the authoritative free "
            "resource for drug-specific safety data during lactation."
        ),
        "latch": (
            "A good latch: baby's mouth opens wide (≥120° angle), covers most of the "
            "areola (not just the nipple), lower lip flanged outward. "
            "Nursing should not be painful after the first few seconds. "
            "If pain or poor weight gain persists, seek support from an IBCLC "
            "(International Board Certified Lactation Consultant)."
        ),
        "supply": (
            "Milk supply is demand-driven: the more frequently and effectively milk is "
            "removed, the more is produced. Aim for 8–12 nursing or pumping sessions per "
            "24 hours in the early weeks to establish supply. "
            "Skin-to-skin contact, adequate hydration, and avoiding formula supplementation "
            "unless medically necessary all support supply."
        ),
        "position": (
            "Common effective positions: cradle hold, cross-cradle hold, football/clutch "
            "hold, and side-lying. The key is that the baby's body faces the breast "
            "(tummy-to-tummy), the head is not turned, and the mouth covers most of the "
            "areola for a deep latch."
        ),
        "iodine": (
            "Iodine is critical for infant brain development and is secreted in breast milk. "
            "WHO recommends 250 mcg/day for breastfeeding women (vs 150 mcg normally). "
            "Many prenatal vitamins contain insufficient iodine — check the label. "
            "Good dietary sources: dairy, seafood, iodised salt."
        ),
    }

    topic_lower = topic.lower()
    for key, guidance in guidelines.items():
        if key in topic_lower:
            return f"[{key.title()} — evidence-based guidance]\n{guidance}"

    return (
        f"No pre-loaded guideline found for '{topic}'. "
        "Recommended resources: "
        "LactMed/NIH (https://www.ncbi.nlm.nih.gov/books/NBK501922/), "
        "La Leche League (llli.org), "
        "or AAP policy statements at healthychildren.org."
    )
