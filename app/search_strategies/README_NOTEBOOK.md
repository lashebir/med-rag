# Benchmark Analysis Notebook

This notebook provides comprehensive analysis and visualization of search strategy benchmarks across different query counts.

## Quick Start

### Install Required Packages First

```bash
# From project root
cd app/search_strategies

# Activate virtual environment
source ../../.venv/bin/activate

# Install analysis dependencies
pip install -r requirements_analysis.txt
```

### Option 1: Python Script (Fastest)

```bash
# Generate all tables and export to CSV/Excel
python generate_benchmark_tables.py

# View output in console and check benchmark_tables/ directory
```

### Option 2: Jupyter Notebook

```bash
# Launch notebook
jupyter notebook benchmark_analysis.ipynb
```

### Option 3: JupyterLab (Recommended)

```bash
# Launch JupyterLab
jupyter lab benchmark_analysis.ipynb
```

### Option 4: VS Code

1. Open the notebook in VS Code
2. Install the "Jupyter" extension
3. Select Python kernel from `.venv`
4. Click "Run All" or run cells individually

## What the Notebook Does

### 📊 Tables Generated

1. **Average Latency** - Mean response time by strategy and query count
2. **Median Latency** - Median response time (less affected by outliers)
3. **P95 Latency** - 95th percentile latency (performance under load)
4. **Precision** - Proportion of retrieved documents that are relevant
5. **Recall** - Proportion of relevant documents that are retrieved
6. **F1 Score** - Harmonic mean of precision and recall
7. **Comprehensive Summary** - All metrics in one table

### 📈 Visualizations

1. **Grouped Bar Chart** - Average latency comparison
2. **Heatmaps** - Precision, Recall, and F1 scores
3. **Box Plots** - Latency distribution across strategies
4. **Key Findings** - Automated summary of best performers

### 💾 Exports

The notebook automatically exports:
- Individual CSV files for each metric
- Excel workbook with all tables in separate sheets
- Files saved to `benchmark_tables/` directory

## Benchmark Data Files

The notebook analyzes these files:

```
recall analysis/
├── benchmark_results_30queries_recall_metrics.json    # 30 queries
├── benchmark_50queries_recall_metrics.json             # 50 queries
└── benchmark_100queries_recall_metrics.json            # 100 queries
```

## Search Strategies Compared

1. **IVFFlat Only** - Pure vector similarity (baseline)
2. **IVFFlat + NER Boost** - Vector + medical entity boost
3. **IVFFlat + tsvector** - Vector + full-text keyword search
4. **IVFFlat + NER + tsvector** - Full hybrid (all signals combined)

## Metrics Explained

### Latency Metrics
- **Avg Latency**: Mean response time across all queries
- **Median Latency**: Middle value (50th percentile)
- **P95 Latency**: 95% of queries complete within this time
- **P99 Latency**: 99% of queries complete within this time
- **Std Dev**: Standard deviation (variability measure)

### Quality Metrics
- **Precision**: TP / (TP + FP) - How many retrieved docs are relevant
- **Recall**: TP / (TP + FN) - How many relevant docs were retrieved
- **F1 Score**: 2 × (Precision × Recall) / (Precision + Recall)

## Sample Output

```
TABLE 1: Average Latency (ms) by Strategy and Query Count
================================================================================
                           30 queries  50 queries  100 queries
IVFFlat Only                   45.23       52.18        78.08
IVFFlat + NER Boost            28.91       30.45        31.82
IVFFlat + tsvector             32.67       33.89        34.68
IVFFlat + NER + tsvector       34.12       35.21        35.79
```

## Troubleshooting

### "No module named 'pandas'"
```bash
pip install pandas matplotlib seaborn openpyxl
```

### "Kernel not found"
```bash
# Make sure you're in the virtual environment
source ../../.venv/bin/activate
python -m ipykernel install --user --name=medrag
```

### "File not found" errors
```bash
# Run the notebook from app/search_strategies/ directory
cd app/search_strategies
jupyter notebook benchmark_analysis.ipynb
```

### Plots not showing
```bash
# Add this to the first cell if using terminal
%matplotlib inline
```

## Customization

### Change Color Schemes
```python
# Modify these lines in the notebook
cmap='RdYlGn'    # Red-Yellow-Green
# Try: 'viridis', 'plasma', 'Blues', 'RdBu'
```

### Export to Different Formats
```python
# Add to the export cell
table.to_html('output.html')
table.to_latex('output.tex')
fig.savefig('plot.png', dpi=300, bbox_inches='tight')
```

### Filter by Specific Benchmark
```python
# Filter to show only 100 queries
filtered_df = latency_df[latency_df['Benchmark'] == '100 queries']
```

## Next Steps

1. ✅ Run the notebook
2. 📊 Review the tables and charts
3. 💾 Export tables to `benchmark_tables/`
4. 📈 Use Excel file for presentations
5. 🔧 Customize visualizations as needed

## Related Documentation

- [Search Strategies Guide](SEARCH_STRATEGIES.md)
- [IVFFlat Tuning](IVFFLAT_TUNING.md)
- [A/B Testing Guide](../AB_TESTING_GUIDE.md)

## Questions?

See the main [LAUNCH_GUIDE.md](../../LAUNCH_GUIDE.md) for application setup and usage.
