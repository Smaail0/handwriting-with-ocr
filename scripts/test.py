input_file = "data/labels.txt"
output_file = "data/labels_fixed.txt"

with open(input_file, "r", encoding="utf-8") as f:
    lines = f.readlines()

# Convert multiple spaces into a single tab
with open(output_file, "w", encoding="utf-8") as f:
    for line in lines:
        parts = line.strip().split()  # Split on any whitespace
        if len(parts) >= 2:
            corrected_line = parts[0] + "\t" + " ".join(parts[1:])  # Ensure a single tab between filename and label
            f.write(corrected_line + "\n")
        else:
            print(f"Skipping invalid line: {repr(line)}")

print("✅ Labels file has been fixed and saved as 'labels_fixed.txt'.")