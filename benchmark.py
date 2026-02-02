import sys
import os
import asyncio
import sys
import pandas as pd
import time
from phishing_pipeline import pipeline, visual_features, utils

# setup paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# mimic the structure pipeline expects
HOLDOUT_CSV = "benchmark_holdout.csv"
WHITELIST_XLS = "benchmark_whitelist.xlsx"
OUTPUT_CSV = "benchmark_output.csv"

# 1. Load real dataset from PS-02_hold-out_Set_2_Part_1
def load_real_dataset():
    print("Loading real dataset from PS-02_hold-out_Set_2_Part_1.xlsx...")
    # Read the first dataset
    df = pd.read_excel("PS-02_hold-out_Set_2/PS-02_hold-out_Set_2_Part_1.xlsx")
    
    # Convert domain_name column to expected format
    df = df.rename(columns={"domain_name": "Identified Phishing/Suspected Domain Name"})
    df["Cooresponding CSE"] = "Phishing"
    df["Legitimate Domains"] = ""
    
    # Save as CSV for pipeline processing
    df.to_csv(HOLDOUT_CSV, index=False)
    
    # create dummy whitelist (pipeline checks this)
    df_wl = pd.DataFrame({"Legitimate Domains": [""]})
    df_wl.to_excel(WHITELIST_XLS, index=False)
    
    return HOLDOUT_CSV, len(df)

# 2. Run pipeline
async def run_benchmark():
    holdout_file, num_domains = load_real_dataset()
    import psutil
    import torch
    import gc
    batch_size = 100
    min_batch_size = 10
    max_domains = num_domains
    print(f"starting benchmark on {holdout_file} ({num_domains} domains)...")
    start_time = time.time()
    df = pd.read_csv(holdout_file)
    total = len(df)
    processed = 0
    batch_num = 0
    import tempfile
    import shutil
    first_batch = True
    temp_files = []
    while processed < total:
        batch_num += 1
        end_idx = min(processed + batch_size, total)
        batch = df.iloc[processed:end_idx]
        print(f"\n[Batch {batch_num}] Processing domains {processed+1}-{end_idx} of {total} (batch size: {batch_size})")
        # Monitor memory before batch
        cpu_mem = psutil.virtual_memory().percent
        gpu_mem = torch.cuda.memory_allocated() / torch.cuda.get_device_properties(0).total_memory if torch.cuda.is_available() else 0
        print(f"  [Memory] CPU: {cpu_mem:.1f}%  GPU: {gpu_mem*100:.1f}%")
        # Write each batch to a temp file, then append to OUTPUT_CSV
        with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as tmp:
            temp_file = tmp.name
        temp_files.append(temp_file)
        # Save batch to temp CSV for process_urls
        batch.to_csv(temp_file, index=False)
        try:
            await pipeline.process_urls(temp_file, temp_file)
            # Append to OUTPUT_CSV
            if first_batch:
                shutil.copyfile(temp_file, OUTPUT_CSV)
                first_batch = False
            else:
                # Append without header
                with open(temp_file, 'r', encoding='utf-8') as src, open(OUTPUT_CSV, 'a', encoding='utf-8') as dst:
                    next(src)  # skip header
                    shutil.copyfileobj(src, dst)
        except Exception as e:
            print(f"  [Batch {batch_num}] Error: {e}")
        processed = end_idx
        # Cleanup after batch
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
        gc.collect()
        # Monitor memory after batch
        cpu_mem = psutil.virtual_memory().percent
        gpu_mem = torch.cuda.memory_allocated() / torch.cuda.get_device_properties(0).total_memory if torch.cuda.is_available() else 0
        print(f"  [After Batch] CPU: {cpu_mem:.1f}%  GPU: {gpu_mem*100:.1f}%")
        # Adaptive batch size
        if cpu_mem > 85 or gpu_mem > 0.8:
            batch_size = max(min_batch_size, batch_size // 2)
            print(f"  [Batch {batch_num}] High memory usage detected. Reducing batch size to {batch_size}.")
        elif cpu_mem < 60 and gpu_mem < 0.5 and batch_size < 100:
            batch_size = min(100, batch_size * 2)
    # Cleanup temp files
    for f in temp_files:
        try:
            os.remove(f)
        except Exception:
            pass
    end_time = time.time()
    duration = end_time - start_time
    print(f"\n==========================================")
    print(f"BENCHMARK COMPLETED")
    print(f"Time taken: {duration:.2f} seconds")
    print(f"Domains processed: {num_domains}")
    print(f"Speed: {duration/num_domains:.2f} seconds/domain")
    print(f"Output saved to: {OUTPUT_CSV}")
    print(f"==========================================")
    try:
        await visual_features.close_browser_async()
        await asyncio.sleep(2)
    except Exception as e:
        print(f"Error during cleanup: {e}")
    gc.collect()

if __name__ == "__main__":
    # Windows loop policy
    if sys.platform.startswith("win"):
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
        
    asyncio.run(run_benchmark())
