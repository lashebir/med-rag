# Benchmark Analysis - Quick Start Guide

Generate presentation-ready tables showing latency and precision metrics across different query counts.

## 🚀 Fastest Way

```bash
# 1. Navigate to this directory
cd app/search_strategies

# 2. Activate virtual environment
source ../../.venv/bin/activate

# 3. Install dependencies (one-time)
pip install -r requirements_analysis.txt

# 4. Run the generator
python generate_benchmark_tables.py
```

**Output:**
- Formatted tables printed to console
- CSV files in `benchmark_tables/` directory
- Excel file with all tables in separate sheets

## 📊 What You Get

### Tables Generated:
1. **Average Latency** - Mean response time by strategy
2. **Median Latency** - Typical response time
3. **P95 Latency** - 95th percentile (performance SLA)
4. **Precision** - Accuracy of retrieved results
5. **Recall** - Completeness of retrieved results
6. **F1 Score** - Balanced precision/recall metric
7. **Comprehensive Summary** - All metrics combined

### Strategies Compared:
- IVFFlat Only (baseline)
- IVFFlat + NER Boost
- IVFFlat + tsvector
- IVFFlat + NER + tsvector (full hybrid)

### Query Counts Analyzed:
- 30 queries
- 50 queries
- 100 queries

## 📓 Alternative: Jupyter Notebook

For interactive analysis with visualizations:

```bash
# Launch notebook
jupyter notebook benchmark_analysis.ipynb

# Or use JupyterLab
jupyter lab benchmark_analysis.ipynb
```

See [README_NOTEBOOK.md](README_NOTEBOOK.md) for full notebook documentation.

## 📁 Output Location

All generated files are saved to:
```
app/search_strategies/benchmark_tables/
├── avg_latency_comparison.csv
├── median_latency_comparison.csv
├── p95_latency_comparison.csv
├── precision_comparison.csv
├── recall_comparison.csv
├── f1_score_comparison.csv
├── comprehensive_summary.csv
└── benchmark_analysis_YYYYMMDD_HHMMSS.xlsx
```

## 💡 Usage Tips

### For Presentations
Use the Excel file - it has all tables in separate sheets for easy copy/paste into slides.

### For Analysis
Use the Jupyter notebook for interactive exploration and custom visualizations.

### For Automation
Use the Python script in CI/CD pipelines or scheduled reports.

## 🔧 Troubleshooting

**"ModuleNotFoundError: No module named 'pandas'"**
```bash
pip install -r requirements_analysis.txt
```

**"FileNotFoundError"**
Make sure you're running from `app/search_strategies/` directory:
```bash
pwd  # Should show: .../med-rag/app/search_strategies
```

**"No benchmark files found"**
Check that these files exist:
```bash
ls -la "recall analysis/"/*.json
```

## 📚 Related Docs

- [Full Notebook Guide](README_NOTEBOOK.md) - Detailed documentation
- [Search Strategies](SEARCH_STRATEGIES.md) - Strategy explanations
- [A/B Testing](../AB_TESTING_GUIDE.md) - Production testing

---

**Quick Command Reference:**
```bash
# Install dependencies
pip install -r requirements_analysis.txt

# Generate tables
python generate_benchmark_tables.py

# Open notebook
jupyter notebook benchmark_analysis.ipynb

# View results
open benchmark_tables/  # macOS
```
