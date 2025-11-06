# ============================================================
#  Gemini Ingredient Detection + Supabase (.env Version)
# ============================================================

import os, json, re, datetime
from PIL import Image
import matplotlib.pyplot as plt
import google.generativeai as genai
from supabase import create_client, Client
from dotenv import load_dotenv

# ============================================================
# STEP 1 — Load Environment Variables
# ============================================================
load_dotenv()  #  loads .env automatically

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
SUPABASE_URL   = os.getenv("SUPABASE_URL")
SUPABASE_KEY   = os.getenv("SUPABASE_KEY")
USER_ID        = os.getenv("USER_ID")

# ============================================================
# STEP 2 — Configure Clients
# ============================================================
genai.configure(api_key=GEMINI_API_KEY)
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# ============================================================
# STEP 3 — Load Local Image
# ============================================================
image_path = os.path.join("images", "fridge.jpg")
img = Image.open(image_path)

plt.imshow(img)
plt.axis("off")
plt.title("Uploaded Image")
plt.show()

# ============================================================
# STEP 4 — Gemini Detection
# ============================================================
prompt = """
You are an expert in food and ingredient detection.
Analyze this image of a fridge, pantry, or kitchen counter and identify all visible ingredients or food items.

For each detected item, estimate how confident you are that the identification is correct (0.0 to 1.0).
Return strictly JSON:
{
  "ingredients": [
    {"name": "ingredient_name", "confidence": confidence_value}
  ]
}
"""

model = genai.GenerativeModel("models/gemini-2.5-flash")
response = model.generate_content([prompt, img])

print("\n🔍 Raw Gemini Response:\n")
print(response.text)

# ============================================================
# STEP 5 — Parse JSON
# ============================================================
text = response.text
match = re.search(r"\{[\s\S]*\}", text)
data = {"ingredients": []}
if match:
    try:
        data = json.loads(match.group(0))
        print("\n Parsed Ingredients:\n")
        for i in data.get("ingredients", []):
            print(f"- {i['name']}: {i['confidence']:.2f}")
    except json.JSONDecodeError:
        print("\n⚠️ Could not parse JSON, showing raw text.")
        print(response.text)

# ============================================================
# STEP 6 — Supabase Sync
# ============================================================
def sync_to_supabase(detections):
    today = datetime.date.today().isoformat()
    for d in detections:
        name = d.get("name", "").lower()
        conf = round(d.get("confidence", 0), 2)

        supabase.table("pantry").insert({
            "user_id": USER_ID,
            "item_name": name,
            "date_added": today
        }).execute()

        supabase.table("substitutions").insert({
            "ingredient": name,
            "alt_name": None,
            "nutrition_delta": {"confidence": conf},
            "source": "Gemini AI"
        }).execute()

    print(f" Synced {len(detections)} items to Supabase.")

sync_to_supabase(data.get("ingredients", []))

# ============================================================
# STEP 7 — Save output
# ============================================================
os.makedirs("outputs", exist_ok=True)
with open("outputs/detections.json", "w") as f:
    json.dump(data, f, indent=2)
print(" Saved results to outputs/detections.json")
