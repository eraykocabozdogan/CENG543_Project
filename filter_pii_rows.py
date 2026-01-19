import csv
import os

# Configuration

# Configuration
TARGET_ROWS = 250

# Define file groups. 'baseline' and 'placeholder' are used to detect PII rows.
# 'others' are additional files to filter using the same indices.
# Define file groups. 'baseline' and 'placeholder' are used to detect PII rows.
# 'others' are additional files to filter using the same indices.
DATA_GROUPS = [
    {
        'baseline': 'data/results_01_baseline.csv',
        'placeholder': 'data/results_02_placeholder.csv',
        'others': [
            'data/results_03_faker.csv',
            'data/results_04_context_aware.csv'
        ]
    },
    {
        'baseline': 'faiss_data/results_baseline_faiss.csv',
        'placeholder': 'faiss_data/results_placeholder_faiss.csv',
        'others': [
            'faiss_data/results_faker_faiss.csv',
            'faiss_data/results_context_aware_faiss.csv'
        ]
    }
]

def load_csv(path):
    rows = []
    if not os.path.exists(path):
        print(f"Warning: File not found: {path}")
        return [], []
        
    with open(path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows, reader.fieldnames

def write_csv(path, fieldnames, rows):
    # Determine output path (append _filtered before extension)
    base, ext = os.path.splitext(path)
    if "_filtered" in base:
        output_path = path # Already filtered name?
    else:
        output_path = f"{base}_filtered{ext}"
        
    with open(output_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Saved: {os.path.basename(output_path)}")

def process_group(group_config):
    print(f"Processing group based on: {os.path.basename(group_config['baseline'])}")
    
    # Load primary files for detection
    base_rows, base_fields = load_csv(group_config['baseline'])
    place_rows, place_fields = load_csv(group_config['placeholder'])
    
    if len(base_rows) != len(place_rows):
        print("Error: Row count mismatch between Baseline and Placeholder!")
        return

    # Load other files
    other_files_data = []
    for path in group_config['others']:
        rows, fields = load_csv(path)
        other_files_data.append({
            'path': path,
            'rows': rows,
            'fields': fields
        })
        if len(rows) != len(base_rows):
             print(f"Warning: Row count mismatch for {os.path.basename(path)}")
    
    # Identify indices
    pii_indices = []
    for i in range(len(base_rows)):
        b_row = base_rows[i]
        p_row = place_rows[i]
        
        # Check if context snippet changed
        if b_row.get('retrieved_context_snippet') != p_row.get('retrieved_context_snippet'):
            pii_indices.append(i)
            
        if len(pii_indices) >= TARGET_ROWS:
            break
            
    print(f"  Found {len(pii_indices)} PII rows.")
    
    # Filter and Save Baseline & Placeholder
    filtered_base = [base_rows[i] for i in pii_indices]
    filtered_place = [place_rows[i] for i in pii_indices]
    
    write_csv(group_config['baseline'], base_fields, filtered_base)
    write_csv(group_config['placeholder'], place_fields, filtered_place)
    
    # Filter and Save Others
    for item in other_files_data:
        # Only if we have data suitable for these indices
        if len(item['rows']) >= max(pii_indices):
            filtered_rows = [item['rows'][i] for i in pii_indices]
            write_csv(item['path'], item['fields'], filtered_rows)
        else:
            print(f"  Skipping {os.path.basename(item['path'])} due to size mismatch.")
    print("")

def main():
    for group in DATA_GROUPS:
        process_group(group)
    print("Done.")

if __name__ == "__main__":
    main()

