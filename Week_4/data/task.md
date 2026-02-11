# Extension Menu Implementation

## 1. Evaluation Dashboard
- [x] Add a second Streamlit tab/page that reads `logs/query_metrics.csv`
- [x] Show P@5 and R@10 charts per retrieval mode
- [x] Show latency distribution
- [x] Show per-query breakdown table

## 2. Batch Evaluation Pipeline
- [x] Add `batch_evaluate()` to `pipeline.py` that runs all MINI_GOLD queries across all modes
- [x] Wire into Streamlit with a "Run Batch Evaluation" button
- [x] Output comparison table with metrics per mode

## 3. Metadata Filtering
- [x] Add metadata fields (source, type) to evidence items in `load_corpus`
- [x] Add sidebar filters for source/file type
- [x] Filter evidence before retrieval

## 4. Response Caching & Query Dedup
- [x] Add `@st.cache_data` response cache with 1hr TTL
- [x] Add sidebar "Clear Cache" button
- [x] Wire query tab to use cached function
