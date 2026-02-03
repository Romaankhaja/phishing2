# OCR Optimization & Fine-Tuning Plan

This document outlines strategies to optimize the `extract_ocr_text` method in the phishing detection pipeline. The goal is to reduce computation time and VRAM usage on the RTX 2050 while maintaining high detection accuracy.

## 1. Multi-Stage OCR Filtering (Early Exit)
Currently, every screenshot is processed by the heavy EasyOCR model.
*   **Strategy:** Use a lightweight "Text Presence Detection" stage.
*   **Implementation:** Before running OCR, use OpenCV's **Edge Detection (Canny)** or **MSER (Maximally Stable Extremal Regions)** to check if the image even contains text-like patterns.
*   **Benefit:** Skips OCR entirely for pages with only graphics or empty layouts, saving CPU/GPU cycles.

## 2. Region of Interest (ROI) Cropping
Processing a full 1280x900 image is inefficient as phishers usually place brand names and login fields in predictable locations.
*   **Strategy:** Crop the image to critical zones before OCR.
*   **Proposed Zones:**
    1.  **Header (Top 200px):** For brand names and logos.
    2.  **Center-Center (Vertical/Horizontal slice):** For login forms and "Verify Now" buttons.
*   **Benefit:** Reduing the pixel count by 60-70% drastically speeds up inference time.

## 3. Hardware & Model Fine-Tuning
Leverage the RTX 2050 architecture more effectively.
*   **Strategy A: Half-Precision (FP16):** Ensure EasyOCR is running in FP16 mode. RTX series "Tensor Cores" are optimized for 16-bit math, which can double throughput compared to FP32.
*   **Strategy B: Batch Inference:** Modify `extract_all_features_async` to collect multiple images and pass them as a list to the OCR reader. Processing 4 images at once is faster than 4 sequential calls.
*   **Strategy C: Model Selection:** EasyOCR allows choosing between different detection (CRAFT) and recognition models. Switching to a "slim" model can reduce VRAM footprint.

## 4. Image Pre-processing for Accuracy
High accuracy at lower resolutions saves time.
*   **Technique:** Apply **Bilateral Filtering** to remove noise followed by **Adaptive Thresholding** (converting to binary B&W). 
*   **Benefit:** Simplifies the image background, making it easier for the neural network to "see" characters without needing high-complexity sampling.

## 5. Parallel Pipeline Restructuring
*   **Strategy:** Move OCR to a separate dedicated worker process or a "low-priority" queue. 
*   **Logic:** URL features (instant) and GeoIP (fast) can be processed first to provide an "Immediate Risk Score." The visual/OCR data can then be used to confirm or upgrade the risk level.

---

### **Estimated Impact**
| Optimization | Est. Speedup | Complexity | 
| :--- | :--- | :--- | 
| ROI Cropping | 2x - 3x | Medium | 
| FP16 Inference | 1.5x | Low | 
| Early Exit Filter | 1.2x (Avg) | Low | 
| Batching | 2x | High | 

## 6. Screenshot Capture Optimization (RAM & Network)
The following strategies focus on making the Playwright-based screenshot capture faster and less memory-intensive.

### A. Resource Interception & Filtering
Phishing detection mainly needs the visual layout of the page, not heavy media or tracking scripts.
*   **Strategy:** Block requests for ads, trackers, videos, and heavy fonts.
*   **Implementation:** Use `page.route("**/*")` to abort requests for irrelevant MIME types (e.g., `application/font-woff`, `video/*`, `image/gif`).
*   **Benefit:** Reduces memory usage by 40-50% and speeds up page loading significantly.

### B. Intelligent Viewport Scaling
*   **Strategy:** Capture at a lower `deviceScaleFactor` (e.g., 1.0 instead of 2.0). 
*   **Implementation:** Set `viewport={"width": 1280, "height": 800}` but use a lower DPI setting.
*   **Benefit:** Dramatically reduces the RAM required to store the pixel buffer and the resulting file size.

### C. Chromium Flags for Headless Efficiency
*   **Strategy:** Pass specific startup flags to the Chromium instance.
*   **Proposed Flags:**
    *   `--disable-dev-shm-usage`: Prevents `/dev/shm` memory exhaustion in container/small environments.
    *   `--js-flags="--max-old-space-size=512"`: Limits the JavaScript engine's memory usage.
    *   `--disable-gpu` (Conditional): If the RTX 2050 is busy with OCR, running Chromium on the CPU might prevent VRAM bottlenecks.

### D. Navigation Wait Optimizations
*   **Strategy:** Switch from `networkidle` to `domcontentloaded`.
*   **Implementation:** Phishers often use slow-loading external scripts to hide. Waiting for the full network to be idle is a waste of time.
*   **Benefit:** Capture occurs as soon as the HTML structure is ready, often saving 2-3 seconds per URL.

### E. Global Browser Context Pooling
*   **Strategy:** Reuse the same browser context for groups of 10-20 URLs before "refreshing."
*   **Benefit:** Balancing memory leaks (which happen in long-running browsers) with the overhead of creating new sessions.
