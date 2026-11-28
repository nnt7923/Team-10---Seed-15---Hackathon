# Pipeline dữ liệu (thư mục `data/`)

Tài liệu đầy đủ (luồng v2 bản chốt, nguồn dữ liệu, model, kết quả) nằm ở [README của repo](../README.md).

Chạy nhanh trong thư mục này:

```powershell
python scripts/run_pipeline.py --config configs/pipeline_v2.yaml --stages resolve,clean,balance,report,export
```

```powershell
python scripts/analyze_raw.py --config configs/pipeline_v2.yaml
```
