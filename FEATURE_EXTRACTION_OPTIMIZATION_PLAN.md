# Feature Extraction Fine-Tuning Plan

## Current State Analysis

### Feature Categories Identified:
1. **URL Structure Features** (`features.py`)
   - URL Length, character counts (dots, hyphens, slashes, special chars)
   - Domain analysis (length, hyphens, special characters)
   - Subdomain features (count, length, hyphen/digit patterns)
   - Path features (length, query, fragment, anchor presence)

2. **Cryptographic Features** (`features.py`)
   - Entropy calculation (URL and domain level)
   - SSL/TLS validation (certificate presence, validity, expiry days, issuer)
   - IP address extraction

3. **Visual Features** (`visual_features.py`)
   - Screenshot capture (using Playwright, async + sync)
   - Brand color extraction (KMeans clustering)
   - Logo/Favicon detection and matching
   - OCR text extraction (using EasyOCR)
   - Image quality metrics (Laplacian variance)

4. **Geolocation Features** (`geoip_utils.py`)
   - ASN (Autonomous System Number)
   - Country, region, city mapping

---

## Optimization Plan (5 Phases)

### **Phase 1: Feature Engineering & Selection** (Priority: HIGH)
**Goal:** Improve feature quality and reduce noise

#### 1.1 URL Structure Features Refinement
- [ ] **Add character distribution entropy** for URL path and query parameters
  - Current: Basic character counting
  - Enhancement: Shannon entropy per URL component (protocol, domain, path, query)
  - Impact: Better detection of obfuscated/encoded parameters

- [ ] **Implement domain age detection**
  - Use WHOIS data to extract registration date
  - Calculate domain age (days since registration)
  - Flag suspicious patterns: very new domains, recent registration changes

- [ ] **Add homograph attack detection**
  - Detect similar-looking characters (e.g., '0' vs 'O', 'l' vs '1')
  - Calculate "visual similarity score" to legitimate domain variants
  - Check Levenshtein distance to known legitimate domains

- [ ] **Normalize and categorize special character patterns**
  - Instead of just counting special chars, categorize them:
    - URL encoding patterns (%, ?, &)
    - Escape sequences (\, ^, ~)
    - Math/logic symbols (@, $, !, =)
  - Track sequences and positions (beginning, middle, end)

#### 1.2 Subdomain Analysis Enhancement
- [ ] **Implement subdomain depth analysis**
  - Current: Only counts subdomains
  - Add: Subdomain position patterns (valid depth for domain type)
  - Flag unusual nesting (10+ levels)

- [ ] **Add subdomain character analysis**
  - Detect random/gibberish subdomains vs. semantic ones
  - Check for number patterns in subdomains (e.g., "sub123.sub456")

#### 1.3 SSL/TLS Certificate Analysis
- [ ] **Expand certificate validation checks**
  - Current: Basic validity check
  - Add:
    - Certificate transparency log verification
    - Issuer reputation scoring (CA trust list)
    - Self-signed certificate detection
    - Wildcard certificate detection
    - SAN (Subject Alternative Name) count and anomalies

- [ ] **Add SSL/TLS version detection**
  - Identify deprecated versions (SSLv3, TLS 1.0)
  - Flag weak cipher suites

- [ ] **Certificate issuer blacklist/whitelist**
  - Known fraudulent CAs
  - Recently compromised CAs

---

### **Phase 2: Visual Feature Enhancement** (Priority: HIGH)
**Goal:** Improve visual analysis accuracy and efficiency

#### 2.1 Branding & Logo Analysis
- [ ] **Implement perceptual hashing refinement**
  - Current: Basic logo matching with imagehash
  - Add: Multiple hash algorithms (pHash, dHash, wHash)
  - Calculate similarity thresholds based on legitimate logo variants
  - Detect partial/cropped logo matches

- [ ] **Color palette analysis enhancement**
  - Current: KMeans clustering (3 colors)
  - Improve:
    - Use Delta-E 2000 (already imported but check usage)
    - Add dominant color percentage calculation
    - Detect color inversion patterns
    - Flag color combinations associated with phishing

- [ ] **Brand guideline compliance scoring**
  - Create brand profile: acceptable colors, fonts, logo placement
  - Calculate compliance score (0-100)
  - Flag significant deviations

#### 2.2 OCR & Text Analysis
- [ ] **Implement text content classification**
  - Current: Raw OCR text extraction
  - Enhance:
    - Extract and analyze call-to-action (CTA) phrases
    - Detect urgency language patterns (urgent, verify, confirm, act now)
    - Identify credential request indicators
    - Language detection and inconsistency flagging

- [ ] **Add font analysis**
  - Detect mixed/unusual fonts
  - Flag suspicious font choices for legitimate brands
  - Analyze text size hierarchy

#### 2.3 Layout & Structure Analysis
- [ ] **Implement page structure analysis**
  - Detect form elements (input fields, buttons)
  - Count form fields and analyze field types
  - Detect suspicious field ordering (e.g., asking for SSN before account number)

- [ ] **Add visual hierarchy analysis**
  - Implement contrast ratio calculation
  - Detect readability issues
  - Analyze element positioning and spacing

- [ ] **Image quality metrics expansion**
  - Current: Laplacian variance only
  - Add:
    - Gaussian blur detection
    - Compression artifacts detection
    - Resolution analysis (unusually low = suspicious)
    - Aspect ratio analysis

---

### **Phase 3: Performance & Async Optimization** (Priority: MEDIUM)
**Goal:** Reduce extraction time and resource usage

#### 3.1 Parallelization Improvements
- [ ] **Optimize screenshot capture**
  - Current: 5s timeout, domcontentloaded
  - Review: Can we reduce to 3s without missing content?
  - Add: Parallel navigation attempts (IPv4 vs IPv6, etc.)

- [ ] **Implement feature extraction batching**
  - Group independent CPU-bound tasks
  - Add: Multiprocessing for visual feature extraction
  - Maintain async I/O for network operations

#### 3.2 Caching & Memoization
- [ ] **Add screenshot cache**
  - Cache screenshots by domain hash
  - Implement cache invalidation (age-based: 30 days?)
  - Save disk space with compression

- [ ] **Cache OCR results**
  - Store OCR text output with screenshot hash
  - Avoid re-running EasyOCR on cached images

- [ ] **Cache favicon downloads**
  - Store favicons by domain with TTL
  - Implement favicon cache pruning

#### 3.3 GPU Optimization
- [ ] **Optimize EasyOCR GPU usage**
  - Current: Auto-detect GPU
  - Review: Batch OCR processing if processing multiple images
  - Monitor GPU memory usage
  - Add fallback for low-VRAM GPUs

- [ ] **Add GPU memory management**
  - Clear GPU cache between batches
  - Monitor GPU temperature

---

### **Phase 4: Robustness & Error Handling** (Priority: MEDIUM)
**Goal:** Improve reliability and reduce failed feature extractions

#### 4.1 Network Resilience
- [ ] **Improve SSL certificate fetching**
  - Current: Silent failures
  - Add: Retry logic with exponential backoff
  - Add: SNI (Server Name Indication) support
  - Detect certificate pinning

- [ ] **Enhance IP resolution**
  - Current: Simple gethostbyname
  - Add:
    - IPv6 support
    - Multiple DNS resolver fallback
    - DNS rebinding detection

- [ ] **Improve screenshot capture reliability**
  - Add: Retry logic for timeouts
  - Implement: Wait for dynamic content (JavaScript execution)
  - Add: Cookie/session handling if needed

#### 4.2 Validation & Sanity Checks
- [ ] **Add feature validation layer**
  - Validate feature ranges and types
  - Detect anomalies (e.g., negative values where impossible)
  - Log warnings for suspicious feature values

- [ ] **Implement fallback values**
  - Define sensible defaults for failed extractions
  - Use weighted imputation based on feature importance
  - Track extraction failure rates per feature

#### 4.3 Logging & Monitoring
- [ ] **Enhance feature extraction logging**
  - Log extraction time per feature
  - Track failure reasons
  - Monitor resource usage (CPU, GPU memory)

- [ ] **Add metrics collection**
  - Extraction success rate by feature
  - Average extraction time
  - Timeout frequency

---

### **Phase 5: Feature Integration & Testing** (Priority: MEDIUM)
**Goal:** Ensure features work well with model and maintain quality

#### 5.1 Feature Correlation Analysis
- [ ] **Analyze feature redundancy**
  - Calculate correlation matrix of extracted features
  - Identify highly correlated features
  - Evaluate if some features can be combined or removed

- [ ] **Feature importance analysis**
  - Use current XGBoost models to measure feature importance
  - Prioritize optimization of high-impact features
  - Consider dropping low-importance features

#### 5.2 Quality Assurance
- [ ] **Create feature extraction test suite**
  - Benchmark on known phishing/legitimate URLs
  - Verify feature stability (same URL → consistent features)
  - Test edge cases (IDN domains, special ports, etc.)

- [ ] **Create feature profiles**
  - Document expected value ranges per feature
  - Create anomaly detection for feature extraction issues

#### 5.3 Feature Documentation
- [ ] **Document each feature extraction step**
  - Purpose and motivation
  - Expected value ranges
  - Known limitations
  - Update frequency strategy

- [ ] **Maintain feature changelog**
  - Track when features are added/modified
  - Document impact on model performance

---

## Implementation Priority Matrix

| Phase | Priority | Effort | Impact | Timeline |
|-------|----------|--------|--------|----------|
| 1.1 - URL Features | HIGH | Medium | High | Week 1-2 |
| 1.2 - Subdomain Analysis | MEDIUM | Low | Medium | Week 1 |
| 1.3 - SSL Enhancement | HIGH | Medium | High | Week 2-3 |
| 2.1 - Branding & Logo | HIGH | High | High | Week 2-4 |
| 2.2 - OCR & Text | HIGH | Medium | High | Week 3-4 |
| 2.3 - Layout Analysis | MEDIUM | High | Medium | Week 4-5 |
| 3.1 - Parallelization | MEDIUM | Medium | Medium | Week 5-6 |
| 3.2 - Caching | MEDIUM | Low | Medium | Week 6 |
| 3.3 - GPU Optimization | LOW | Medium | Low | Week 6-7 |
| 4.1 - Network Resilience | MEDIUM | Medium | Medium | Week 5-6 |
| 4.2 - Validation | MEDIUM | Low | Medium | Week 5 |
| 4.3 - Monitoring | MEDIUM | Low | Medium | Week 7 |
| 5.1 - Correlation Analysis | MEDIUM | Low | High | Week 7-8 |
| 5.2 - QA Testing | HIGH | Medium | High | Week 8 |
| 5.3 - Documentation | LOW | Low | Low | Ongoing |

---

## Quick Win Recommendations

**Start with these (immediate impact, low effort):**
1. ✅ Add domain age feature (WHOIS)
2. ✅ Expand SSL certificate checks (issuer reputation)
3. ✅ Implement OCR text classification (urgency patterns)
4. ✅ Add form field detection from screenshots
5. ✅ Improve retry logic for network operations

**Then move to:**
6. Color palette analysis refinement
7. Homograph attack detection
8. Caching layer implementation
9. Feature correlation analysis
10. Comprehensive testing suite

---

## Expected Outcomes

- **Detection Accuracy:** +3-7% improvement in classification
- **False Positive Rate:** Reduction through better feature quality
- **Extraction Speed:** 15-25% faster with caching + parallelization
- **Robustness:** 20-30% fewer feature extraction failures
- **Maintainability:** Better documented, easier to troubleshoot

---

## Dependencies & Resources

### New Libraries/Tools Needed:
- `python-whois` - Domain age extraction
- Additional WHOIS service API (optional: whoisxmlapi)
- Enhanced SSL/TLS library (cryptography module)

### Existing but Under-utilized:
- `colormath` - Already imported, use delta_e_cie2000
- `easyocr` - Already in use, optimize batching
- `imagehash` - Enhance with multiple algorithms
- `geoip2` - Already in use for location

### GPU Optimization:
- Monitor `torch.cuda` usage
- Batch EasyOCR requests

---

## Rollback Strategy

Each enhancement should:
1. Include feature flags (enable/disable per feature)
2. Store previous feature versions
3. Support A/B testing with model
4. Include simple disable option if performance degrades

