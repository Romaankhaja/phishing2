import asyncio
import os
import pandas as pd
from phishing_pipeline.pipeline import process_urls

# Create dummy CSV
csv_path = "dummy_test.csv"
out_path = "dummy_output.csv"

data = {
    "Identified Phishing/Suspected Domain Name": ["google.com", "example.com"],
    "Cooresponding CSE": ["Google", "IANA"],
    "Legitimate Domains": ["google.com", "example.com"]
}
df = pd.DataFrame(data)
df.to_csv(csv_path, index=False)

async def main():
    print("Running optimization test...")
    try:
        await process_urls(csv_path, out_path)
        print("Success! output generated.")
        if os.path.exists(out_path):
            res = pd.read_csv(out_path)
            print(f"Output rows: {len(res)}")
            print(res.head())
    except Exception as e:
        print(f"Test Failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    if os.name == 'nt':
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    asyncio.run(main())
