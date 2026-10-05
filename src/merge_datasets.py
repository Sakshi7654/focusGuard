import os
import glob
import pandas as pd

processed_dir = "data/processed"
output_file = "data/master_custom_dataset.csv"

# Locate all CSV files recorded by team members
csv_files = glob.glob(os.path.join(processed_dir, "*.csv"))

if not csv_files:
    print(f"[!] No CSV files found in '{processed_dir}'.")
    print("Ensure all members place their session files inside 'data/processed/'.")
else:
    dfs = []
    for file in csv_files:
        temp_df = pd.read_csv(file)
        # Record source file name to keep track of sessions
        temp_df["source_session"] = os.path.basename(file)
        dfs.append(temp_df)

    master_df = pd.concat(dfs, ignore_index=True)
    
    # Save the combined dataset
    master_df.to_csv(output_file, index=False)
    print("==========================================")
    print(f"[SUCCESS] Merged {len(csv_files)} session files.")
    print(f"[SUCCESS] Total aggregated samples: {len(master_df)} rows")
    print(f"[SUCCESS] Master dataset saved to: '{output_file}'")
    print("==========================================")