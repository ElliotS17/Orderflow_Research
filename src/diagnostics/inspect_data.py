from pathlib import Path
import databento as db

# Locate the first day's dataset
data_path = Path("data/raw/glbx-mdp3-20261012.trades.dbn.zst")

if not data_path.exists():
    raise FileNotFoundError(f"Could not find {data_path}")

print("Reading:", data_path.name)

# Decode the Databento DBN file
store = db.DBNStore.from_file(data_path)

# Display metadata
print("\n--- Dataset Metadata ---")
print(store.metadata)

# Convert records to a DataFrame
df = store.to_df()

print("\n--- First 10 Records ---")
print(df.head(10).to_string())

print("\n--- Dataset Information ---")
print(df.info())

print("\n--- Summary Statistics ---")
print(df.describe(include="all"))

print("\n--- Available Columns ---")
print(df.columns.tolist())

print("\nTotal records:", len(df))