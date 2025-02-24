import pandas as pd
import re
from rapidfuzz import process, fuzz
from Levenshtein import distance

# Load and Clean Medication Database
def load_medication_database():
    file_path = "data/liste_amm.xls"  # Update path if needed
    df = pd.read_excel(file_path, dtype=str)  # Ensure all values are strings

    # Ensure required columns exist
    required_columns = ["Nom", "Dosage", "Forme", "DCI"]
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"ERROR: Column '{col}' is missing from medications.xlsx!")

    # Create normalized columns for matching
    df["Short Name"] = df["Nom"].str.lower().str.strip()
    df["Full Name"] = (df["Nom"] + " " + df["Dosage"] + " " + df["Forme"] + " (" + df["DCI"] + ")").str.lower().str.strip()

    return df

# Preprocess OCR Text to Remove Noise
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)  # Remove special characters
    text = re.sub(r"\b(tablet|pill|dose|take|prescribe|mg|ml|spray|syrup|g|p|b)\b", "", text)  # Remove extra words
    return text.strip()

# Correct OCR Medication Name Using Fuzzy Matching
def correct_medication(ocr_text, df):
    cleaned_text = preprocess_text(ocr_text)

    print(f"🔍 Cleaned OCR Text: {cleaned_text}")

    # First, check if exact match exists in "Short Name"
    exact_match_row = df[df["Short Name"] == cleaned_text]
    if not exact_match_row.empty:
        print(f"✅ Exact Match Found: {exact_match_row.iloc[0]['Nom']} (Dosage: {exact_match_row.iloc[0]['Dosage']})")
        return exact_match_row.iloc[0]['Nom'], exact_match_row.iloc[0]['Dosage']

    # If no exact match, use fuzzy matching on "Full Name"
    matches = process.extract(cleaned_text, df["Full Name"], scorer=fuzz.partial_ratio, limit=5)
    
    # remove matches with low score
    matches = [(name, score, index) for name, score, index in matches if score > 85]

    # Print filtered matches
    print(f"🔍 Filtered Matches for '{cleaned_text}': {matches}")

    if not matches:
        return "No close match found", "N/A"

     # Select the best match
    best_match, score, index = matches[0]
    matched_row = df.iloc[index]
    return matched_row["Nom"], matched_row["Dosage"]

# Test the Correction System
if __name__ == "__main__":
    short_name_list, full_name_list = load_medication_database()  # Unpack both lists

    # Example OCR errors
    ocr_results = [
        "Angiplant",
        "Paracetamol",
    ]

    for ocr_text in ocr_results:
        corrected_med, dosage = correct_medication(ocr_text, short_name_list, full_name_list)  # Pass both lists
        print(f"OCR Output: {ocr_text} → Corrected: {corrected_med} ({dosage})")

