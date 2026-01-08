# Grafana Dashboard Setup for VLM Metrics

This directory contains a Grafana dashboard JSON file with custom VLM metrics added to Ray's Serve Deployment Dashboard.

## VLM Metrics Included:
1. **VLM Batch Size per GPU** - Shows how many images are batched together
2. **VLM GPU Throughput** - Images processed per second (different from QPS)

---

## Import Dashboard to Grafana

### Steps:
1. Open Grafana at http://localhost:8265 → Click **"View in Grafana"**
2. **Sign in** to Grafana (default: `admin`/`admin`)
3. Click **"+"** or **"Dashboards"** → **"Import"**
4. Click **"Upload JSON file"**
5. Select: `/home/rp218/projects/offline-rl/ray_based_architecture/vlm_grafana_dashboard.json`
6. Click **"Load"** → **"Import"**

✅ **Done!** The dashboard is now saved in Grafana's database and will persist across Ray restarts.

---

## Quick Reference

### VLM Metrics (Ray already provides QPS, latency, errors, queue size):
- **Batch Size** (`ray_vlm_batch_size`) → Batching efficiency (target: ~128)
- **Throughput** (`ray_vlm_throughput`) → GPU utilization (images/sec)

### Verify Metrics are Exporting:
```bash
curl http://localhost:8080/metrics | grep vlm
```

### If Metrics Missing:
Restart Ray with updated code:
```bash
ray stop
python ray_based_architecture/test_labeling.py
```

