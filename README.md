# genai-pyspark-pipeline

## File format benchmark

Install the project dependencies, then run the benchmark with its default
500,000-row dataset:

```powershell
.\venv\Scripts\python.exe -m src.format_benchmark
```

The benchmark writes one file for each format under
`output\format_benchmark` and prints file size, write/read wall-clock time,
peak Python allocation tracked by `tracemalloc`, process CPU time, estimated
energy usage, and percentage reductions versus CSV. Energy is estimated with a
65 W CPU TDP assumption:

```text
energy (Wh) = CPU time (seconds) * 65 / 3600
```

Use `--rows` and `--output-dir` for a smaller smoke run or a different output
location:

```powershell
.\venv\Scripts\python.exe -m src.format_benchmark --rows 1000 --output-dir output\format_smoke
```

## Sales analytics dashboard

Launch the Streamlit dashboard after generating the data:

```powershell
.\venv\Scripts\python.exe -m streamlit run dashboard.py
```

The dashboard includes overview metrics, top customers, category sales,
monthly trends, and an optional Pandas-vs-PySpark performance benchmark.