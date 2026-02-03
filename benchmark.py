
# =================== BENCHMARK: FULL MAIN PIPELINE ===================
import sys
import os
import asyncio
import pandas as pd
import time
import psutil
import torch
import gc
import logging
from datetime import datetime
from phishing_pipeline import pipeline, shortlisting, visual_features
from phishing_pipeline.config import FINAL_OUTPUT

DATASET_PATH = "PS-02_hold-out_Set_2/PS-02_hold-out_Set_2_Part_1.xlsx"
WHITELIST_PATH = "uploads/PS-02_hold-out_Set1_Legitimate_Domains_for_10_CSEs.xlsx"
LIMIT_SAMPLES = None  # Set to None for full run, or e.g. 100 for quick test

def get_stats():
    cpu_mem = psutil.virtual_memory().percent
    gpu_mem = torch.cuda.memory_allocated() / torch.cuda.get_device_properties(0).total_memory if torch.cuda.is_available() else 0
    return cpu_mem, gpu_mem

async def run_full_pipeline_benchmark():
    print("\n" + "="*70)
    print("🚀 PHISHING PIPELINE: FULL END-TO-END BENCHMARK")
    print("="*70)


    # 1. Shortlisting (load/process only the specified file)
    print(f"\n[STEP 1] Shortlisting with Legitimate CSE Domains (Single File)...")
    if not os.path.exists(DATASET_PATH):
        print(f"❌ Dataset not found: {DATASET_PATH}")
        return
    if not os.path.exists(WHITELIST_PATH):
        print(f"❌ Whitelist not found: {WHITELIST_PATH}")
        return
    import pandas as pd
    start_short = time.time()
    try:
        # Load only the specified Excel file
        df = pd.read_excel(DATASET_PATH)
        print(f"Loaded {len(df)} rows from {DATASET_PATH}")
        # Use all domains for benchmarking (skip whitelist filtering)
        if 'domain_name' not in df.columns:
            print("❌ Could not find 'domain_name' column in dataset.")
            return
        df['domain'] = df['domain_name'].astype(str).str.lower()
        filtered_df = df.copy()
        # Add 'Legitimate Domains' column to match pipeline expectations
        filtered_df['Legitimate Domains'] = filtered_df['domain']
        n_candidates = len(filtered_df)
        print(f"✅ Shortlisting complete: {n_candidates} candidates from {os.path.basename(DATASET_PATH)} (no whitelist filtering).")
        print(f"DataFrame shape: {filtered_df.shape}")
        if n_candidates == 0:
            print("❌ No data to process. Exiting benchmark before pipeline step.")
            return
        import os
        holdout_path = os.path.abspath('holdout.csv')
        filtered_df.to_csv(holdout_path, index=False)
        print(f"holdout.csv written to: {holdout_path}")
        print(f"holdout.csv columns: {list(filtered_df.columns)}")
        print("Sample of holdout.csv:")
        print(filtered_df.head(5))
    except Exception as e:
        print(f"❌ Shortlisting failed: {e}")
        return
    t_short = time.time() - start_short
    print(f"⏱  Time: {t_short:.2f}s")

    # 2. Main Pipeline (run on filtered holdout.csv)
    print(f"\n[STEP 2] Main Pipeline: Feature Extraction, IP, Model (Single File)...")
    cpu_start, gpu_start = get_stats()
    start_pipe = time.time()
    try:
        df_out = await pipeline.run_pipeline(
            holdout_folder=os.path.dirname(DATASET_PATH),
            ps02_whitelist_file=WHITELIST_PATH,
            limit_whitelisted=None,
            use_existing_holdout=True
        )
        n_processed = len(df_out) if df_out is not None else 0
        print(f"✅ Pipeline complete: {n_processed} records processed.")
    except Exception as e:
        print(f"❌ Pipeline failed: {e}")
        return
    t_pipe = time.time() - start_pipe
    cpu_end, gpu_end = get_stats()
    print(f"⏱  Time: {t_pipe:.2f}s")
    print(f"🧠 Memory: CPU {cpu_end:.1f}% (Δ{cpu_end-cpu_start:+.1f}%), GPU {gpu_end*100:.1f}% (Δ{(gpu_end-gpu_start)*100:+.1f}%)")

    # 3. Packaging
    print(f"\n[STEP 3] Packaging Results...")
    start_pack = time.time()
    try:
        zip_path = pipeline.package_results()
        print(f"✅ Packaged results: {zip_path}")
    except Exception as e:
        print(f"❌ Packaging failed: {e}")
        zip_path = None
    t_pack = time.time() - start_pack
    print(f"⏱  Time: {t_pack:.2f}s")

    # 4. Final Summary
    t_total = t_short + t_pipe + t_pack
    print("\n" + "="*70)
    print("🏆 BENCHMARK SUMMARY")
    print("-"*70)
    print(f"Total Time:      {t_total:.2f} seconds")
    print(f"Shortlisting:    {t_short:.2f} s")
    print(f"Pipeline:        {t_pipe:.2f} s")
    print(f"Packaging:       {t_pack:.2f} s")
    print(f"Total Samples:   {n_processed}")
    print(f"Overall Speed:   {t_total/(n_processed if n_processed>0 else 1):.2f} s/domain")
    print(f"Timestamp:       {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)

    # 5. Show sample output
    try:
        if df_out is not None and hasattr(df_out, 'head'):
            print("\nSample Output (first 5 rows):")
            print(df_out.head(5))
    except Exception:
        pass

    # 6. Cleanup
    try:
        await visual_features.close_browser_async()
    except Exception:
        pass

if __name__ == "__main__":
    if sys.platform.startswith("win"):
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    try:
        asyncio.run(run_full_pipeline_benchmark())
    except KeyboardInterrupt:
        print("\nBenchmark stopped by user.")
    except Exception as e:
        print(f"\nFatal Error: {e}")
