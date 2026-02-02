# Memory Optimization & Resource Management Plan

## Current Problem Analysis

### Memory Bottlenecks Identified
1. **EasyOCR on GPU** - Loads ~1.5GB VRAM on first use, accumulates tensor cache
   - RTX 2050 has ~2GB VRAM total, easily exhausted with concurrent operations
   - Error: `CUDA error: out of memory`
   
2. **Playwright Screenshot Capture** - Each screenshot ~5-10MB in memory
   - 316,524 domains × concurrent tasks = hundreds of MB rapidly
   - Shares browser context globally (good) but no cleanup between tasks
   
3. **OpenCV Image Processing** - Large images consume significant CPU RAM
   - Error: `OpenCV: Failed to allocate 79MB` (happens with high-res screenshots)
   - Laplacian variance + KMeans clustering without downsampling
   
4. **Concurrent Task Overflow**
   - No limit on concurrent async tasks = unlimited pending operations
   - All 316K domains loaded into memory at once = crushing the system
   
5. **Memory Leaks**
   - Loaded images not explicitly freed
   - CUDA tensors held in memory between batches
   - Playwright page objects not properly closed in error cases

---

## Root Cause: Unbounded Concurrency

Current code uses `asyncio.gather()` with NO semaphore limits:
```python
results = await asyncio.gather(
    t_ip, t_ssl, t_url_feats, ..., t_ocr, t_brand, t_lap, t_fav,
    return_exceptions=True  # All tasks run immediately!
)
```

With 316K domains being processed asynchronously, thousands of tasks queue instantly.

---

## Solution Architecture

### Level 1: Semaphore-Based Concurrency Control

**GPU-intensive operations** need stricter limits:
- **OCR (EasyOCR)**: Limit to 1-2 concurrent tasks (shares single GPU memory)
- **Screenshot Capture**: Limit to 5-10 concurrent (Playwright can handle it)
- **Image Processing (KMeans, Laplacian)**: Limit to 5 (CPU-heavy)
- **URL Features**: Limit to 20-30 (lightweight CPU operations)

**Implementation:**
```python
# In utils.py
ocr_semaphore = asyncio.Semaphore(2)      # Max 2 concurrent OCR
screenshot_semaphore = asyncio.Semaphore(8)  # Max 8 concurrent screenshots
image_sem = asyncio.Semaphore(5)           # Max 5 image processing
cpu_semaphore = asyncio.Semaphore(20)      # Max 20 URL feature extraction
```

### Level 2: Batch Processing with Memory Monitoring

**Process domains in chunks** instead of all at once:
- **Batch Size**: Start with 100 domains per batch
- **Memory Check**: Monitor GPU/CPU before each batch
- **Adaptive Reduction**: If memory usage > 80%, reduce batch size by half
- **Cleanup After Each Batch**: Clear CUDA cache, force garbage collection

### Level 3: GPU Memory Explicit Management

**EasyOCR cleanup:**
```python
import gc
import torch

# After processing batch
torch.cuda.empty_cache()      # Clear GPU cache
gc.collect()                  # Free Python objects
```

**Lazy OCR Reader Loading:**
- Load model only once per batch, not per domain
- Reload periodically (every 500 domains) to free accumulated cache

### Level 4: Image Memory Optimization

**For OpenCV operations:**
- Downsample large images before KMeans (current: 150×150, OK)
- Explicitly close PIL Image objects: `img.close()`
- Use context managers for image operations
- Reduce full_page screenshots to viewport only

### Level 5: Playwright Browser Cleanup

**Proper cleanup on errors:**
```python
try:
    await page.goto(url, timeout=5000)
    await page.screenshot(path=out_file)
finally:
    await page.close()  # Always close, even on error
```

---

## Implementation Strategy

### Phase 1: Add Resource Limiting (High Priority)
**File: `phishing_pipeline/utils.py`**
- Add 4 semaphores (ocr, screenshot, image, cpu)
- Wrap OCR/screenshot/image calls with semaphore context
- Add batch size configuration constant

### Phase 2: Add Memory Monitoring (High Priority)
**File: `benchmark.py`**
- Get GPU memory usage before/after each batch
- Get CPU memory usage
- Adaptive batch size reduction on high memory usage
- Print memory stats

### Phase 3: GPU Cache Cleanup (Medium Priority)
**File: `phishing_pipeline/visual_features.py`**
- Add explicit `torch.cuda.empty_cache()` after OCR processing
- Add batch reload for EasyOCR reader
- Proper page closure in async screenshot function

### Phase 4: Safe Resource Testing (Medium Priority)
**File: `benchmark.py`**
- Add `--sample-size` parameter (test with 100/500/1000 first)
- Add `--batch-size` parameter for customization
- Add resource monitoring logging
- Add graceful error recovery (skip domain, continue)

---

## Memory Targets

**RTX 2050 Specs:**
- GPU VRAM: ~2GB
- Target GPU Usage: 60-70% during batch (safe margin 20-30%)
- CPU Target: 75% max
- Per-batch OCR models: ~1.2GB
- Per-batch screenshots: ~500MB (100 domains × 5MB)

**Estimated Processing Rate:**
- CPU-only features: 50-100 domains/sec (10-20 sec for 1000)
- With GPU OCR: 2-5 domains/sec (~200-500 sec for 1000)
- Batch overhead: ~5 sec per batch (CUDA cleanup, GC)

---

## Files to Modify

1. **phishing_pipeline/utils.py** - Add semaphores, memory-aware batch loops
2. **phishing_pipeline/visual_features.py** - GPU cache cleanup, safe resource closure
3. **benchmark.py** - Batch processing, memory monitoring, adaptive parameters

---

## Testing Plan

1. **Small Sample Test** (10 domains) - Verify no crashes
2. **Medium Sample Test** (500 domains) - Check memory stability
3. **Large Sample Test** (5,000 domains) - Full stress test
4. **Full Dataset Test** (all 316K) - Production readiness

**Success Criteria:**
- ✅ GPU memory never exceeds 1.8GB during processing
- ✅ CPU memory never exceeds 6GB
- ✅ No CUDA out-of-memory errors
- ✅ No OpenCV allocation failures
- ✅ Graceful handling of problematic domains
- ✅ Completion of all batches without crashes

---

## Code Changes Summary

### New Constants
```python
# Resource limits
MAX_CONCURRENT_OCR = 2
MAX_CONCURRENT_SCREENSHOTS = 8
MAX_CONCURRENT_IMAGE_PROCESSING = 5
MAX_CONCURRENT_CPU_TASKS = 20
BATCH_SIZE = 100
GPU_MEMORY_THRESHOLD = 0.8  # 80% usage triggers reduction
CPU_MEMORY_THRESHOLD = 0.85
OCR_RELOAD_INTERVAL = 500  # Reload model every 500 domains
```

### New Functions
```python
# Memory monitoring
get_gpu_memory_usage() -> float
get_cpu_memory_usage() -> float
should_reduce_batch_size() -> bool

# Resource cleanup
cleanup_gpu_cache() -> None
cleanup_python_objects() -> None

# Batch processing
process_urls_in_batches(urls, batch_size, callbacks) -> results
```

---

## Expected Outcomes

**Before Optimization:**
- Benchmark crashes after ~20% of dataset
- Memory errors: GPU out of memory, OpenCV allocation failures
- Cannot process 316K domains

**After Optimization:**
- ✅ Benchmark completes full dataset
- ✅ Stable GPU/CPU memory usage
- ✅ Processing speed: ~2-5 domains/sec with GPU OCR
- ✅ Estimated time: 18-26 hours for 316K domains (parallel batches can reduce)
- ✅ Graceful error handling for problematic domains
