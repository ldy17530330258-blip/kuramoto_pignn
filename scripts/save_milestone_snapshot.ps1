$ErrorActionPreference = "Stop"

# ===== 1) 基本信息 =====
$ProjectRoot = Get-Location
$MilestoneName = "2026-03-25_kuramoto_sync_robustness_v2_random_degree"

$SnapshotRoot = Join-Path $ProjectRoot "archive\snapshots\$MilestoneName"
$CodeDir      = Join-Path $SnapshotRoot "code"
$OutputsDir   = Join-Path $SnapshotRoot "outputs"
$DocsDir      = Join-Path $SnapshotRoot "docs"
$EnvDir       = Join-Path $SnapshotRoot "env"
$ZipPath      = Join-Path $ProjectRoot "archive\$MilestoneName.zip"

New-Item -ItemType Directory -Force -Path $CodeDir, $OutputsDir, $DocsDir, $EnvDir | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $OutputsDir "logs"), (Join-Path $OutputsDir "figures"), (Join-Path $OutputsDir "checkpoints") | Out-Null

Write-Host "[1/7] Copying code ..."

# ===== 2) 复制代码 =====
$CodeItems = @(
    "configs",
    "models",
    "physics",
    "training",
    "scripts",
    "README.md",
    "requirements.txt"
)

foreach ($item in $CodeItems) {
    if (Test-Path $item) {
        Copy-Item $item -Destination $CodeDir -Recurse -Force
    }
}

Write-Host "[2/7] Copying key logs ..."

# ===== 3) 复制关键日志与结果 =====
$LogPatterns = @(
    "outputs\logs\robustness_compare_all_random_test_*",
    "outputs\logs\robustness_compare_all_degree_test_*",
    "outputs\logs\robustness_v2_edge_random_test_*",
    "outputs\logs\rollout_compare_all_*",
    "outputs\logs\rollout_benchmark_v2_edge.json",
    "outputs\logs\kuramoto_pignn_*_results.json"
)

foreach ($pattern in $LogPatterns) {
    Get-ChildItem $pattern -ErrorAction SilentlyContinue | ForEach-Object {
        Copy-Item $_.FullName -Destination (Join-Path $OutputsDir "logs") -Force
    }
}

Write-Host "[3/7] Copying key figures ..."

# ===== 4) 复制图 =====
if (Test-Path "outputs\figures\paper_ready") {
    Copy-Item "outputs\figures\paper_ready" -Destination (Join-Path $OutputsDir "figures") -Recurse -Force
}

$FigurePatterns = @(
    "outputs\figures\robustness_compare_all_random_test_*",
    "outputs\figures\robustness_compare_all_degree_test_*"
)

foreach ($pattern in $FigurePatterns) {
    Get-ChildItem $pattern -ErrorAction SilentlyContinue | ForEach-Object {
        Copy-Item $_.FullName -Destination (Join-Path $OutputsDir "figures") -Force
    }
}

Write-Host "[4/7] Copying checkpoints ..."

Write-Host "[4.5/7] Copying PyG dataset and splits ..."

# ===== 4.5) 复制数据集 =====
$DataArchiveDir = Join-Path $SnapshotRoot "data\pyg_dataset"
New-Item -ItemType Directory -Force -Path $DataArchiveDir | Out-Null

$DatasetItems = @(
    "data\pyg_dataset\kuramoto_dataset.pt",
    "data\pyg_dataset\graph_split.pkl",
    "data\pyg_dataset\dataset_meta.json"
)

foreach ($item in $DatasetItems) {
    if (Test-Path $item) {
        Copy-Item $item -Destination $DataArchiveDir -Force
    }
}

# ===== 5) 复制 best checkpoints =====
$CheckpointPatterns = @(
    "outputs\checkpoints\kuramoto_pignn_pure_data_best.pt",
    "outputs\checkpoints\kuramoto_pignn_R_guided_best.pt",
    "outputs\checkpoints\kuramoto_pignn_v1_best.pt",
    "outputs\checkpoints\kuramoto_pignn_v2_edge_best.pt"
)

foreach ($pattern in $CheckpointPatterns) {
    Get-ChildItem $pattern -ErrorAction SilentlyContinue | ForEach-Object {
        Copy-Item $_.FullName -Destination (Join-Path $OutputsDir "checkpoints") -Force
    }
}

Write-Host "[5/7] Saving environment info ..."

# ===== 6) 保存环境信息 =====
"ProjectRoot: $ProjectRoot" | Out-File (Join-Path $EnvDir "project_root.txt") -Encoding utf8
"Milestone: $MilestoneName" | Out-File (Join-Path $EnvDir "milestone.txt") -Encoding utf8

python --version 2>&1 | Out-File (Join-Path $EnvDir "python_version.txt") -Encoding utf8
pip freeze 2>&1 | Out-File (Join-Path $EnvDir "pip_freeze.txt") -Encoding utf8

try {
    conda env export --no-builds 2>&1 | Out-File (Join-Path $EnvDir "conda_env.yml") -Encoding utf8
} catch {
    "conda env export failed" | Out-File (Join-Path $EnvDir "conda_env_export_error.txt") -Encoding utf8
}

try {
    nvidia-smi 2>&1 | Out-File (Join-Path $EnvDir "nvidia_smi.txt") -Encoding utf8
} catch {
    "nvidia-smi not available" | Out-File (Join-Path $EnvDir "nvidia_smi.txt") -Encoding utf8
}

Write-Host "[6/7] Writing docs ..."

# ===== 7) 写一个简要说明 =====
$ReadmeText = @"
# Milestone snapshot

Milestone:
$MilestoneName

Purpose:
Kuramoto-PIGNN synchronization robustness milestone.
Contains:
- code snapshot
- random/degree attack robustness results
- paper-ready figures
- best checkpoints
- environment info

Recommended key files:
- outputs/logs/robustness_compare_all_random_test_summary.csv
- outputs/logs/robustness_compare_all_degree_test_summary.csv
- outputs/figures/paper_ready/
- outputs/checkpoints/kuramoto_pignn_v2_edge_best.pt
"@
$ReadmeText | Out-File (Join-Path $DocsDir "snapshot_readme.md") -Encoding utf8

$RunCmdText = @"
# Put the exact commands you used here

python -u scripts/evaluate_robustness.py --tags "pure_data,R_guided,v1,v2_edge" --split test --attack_mode random --q_values "0,0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50" --repeats 5 --rollout_steps 50 --tail_window 10 --device cuda --out_prefix outputs/logs/robustness_compare_all_random_test

python -u scripts/evaluate_robustness.py --tags "pure_data,R_guided,v1,v2_edge" --split test --attack_mode degree --q_values "0,0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50" --repeats 1 --rollout_steps 50 --tail_window 10 --device cuda --out_prefix outputs/logs/robustness_compare_all_degree_test

python -u scripts/plot_publication_robustness.py --summary_paths outputs/logs/robustness_compare_all_random_test_summary.csv outputs/logs/robustness_compare_all_degree_test_summary.csv --detailed_paths outputs/logs/robustness_compare_all_random_test_detailed.csv outputs/logs/robustness_compare_all_degree_test_detailed.csv --attack_names random degree --topology_attack degree --save_dir outputs/figures/paper_ready
"@
$RunCmdText | Out-File (Join-Path $DocsDir "run_commands.txt") -Encoding utf8

Write-Host "[7/7] Compressing snapshot ..."

# ===== 8) 压缩 =====
if (Test-Path $ZipPath) {
    Remove-Item $ZipPath -Force
}
Compress-Archive -Path $SnapshotRoot -DestinationPath $ZipPath -Force

Write-Host ""
Write-Host "Done."
Write-Host "Snapshot folder: $SnapshotRoot"
Write-Host "Zip file       : $ZipPath"