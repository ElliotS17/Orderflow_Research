from pathlib import Path
import inspect
import databento as db

RAW_FILE = Path("data/raw/glbx-mdp3-20261012.trades.dbn.zst")

print("Databento version:", getattr(db, "__version__", "Unknown"))
print("\nDatabento module classes and functions:")
for name in dir(db):
    if any(term in name.lower() for term in ("decoder", "record", "store", "trade")):
        obj = getattr(db, name)
        print(f"\n{name}: {obj}")
        if inspect.isclass(obj) or inspect.isfunction(obj):
            print("  Signature:", end=" ")
            try:
                print(inspect.signature(obj))
            except (TypeError, ValueError):
                print("Unavailable")
            doc = inspect.getdoc(obj)
            if doc:
                print("  Documentation:", doc[:1200])

print("\nLoading one DBN file...")
store = db.DBNStore.from_file(RAW_FILE)

print("\nStore type:", type(store))
print("Store methods and attributes:")
for name in dir(store):
    if not name.startswith("_"):
        obj = getattr(store, name)
        if callable(obj):
            try:
                signature = inspect.signature(obj)
            except (TypeError, ValueError):
                signature = "(signature unavailable)"
            print(f"  {name}{signature}")
        else:
            print(f"  {name}: {type(obj)}")

print("\nDoes the store support iteration?")
try:
    iterator = iter(store)
    first_record = next(iterator)
    print("Yes.")
    print("First record type:", type(first_record))
    print("First record:", repr(first_record))
    print("Record attributes:", [
        name for name in dir(first_record) if not name.startswith("_")
    ])
except Exception as exc:
    print("Could not iterate directly:", type(exc).__name__, str(exc))