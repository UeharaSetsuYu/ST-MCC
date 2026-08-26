# ST-MCC

ST-MCC: Structure-Teacher-Guided Multi-Level Consistency Learning and Cross-View Latent Completion for Incomplete Multi-View Clustering
## 文件职责

| File | Responsibility |
|---|---|
| `train.py` | Sole training entry point, 20+10 two-stage training, final evaluation and model saving |
| `model.py` | Encoders, Common/Specific decomposition, decoders, GRL, shared prototypes, and cross-view predictors |
| `config.py` | A small set of commonly used experimental parameters and command-line argument parsing |
| `unit.py` | Structure teacher, loss functions, EMA prototype updates, and clustering metrics |
| `data_load.py` | BDGP data loading, L2 normalization, and label-free missing-view mask |
## Trainning 

1. Construct a fixed structure teacher and pseudo-labels from the observed data.
2. First 20 epochs: use only Masked Reconstruction, where both Common and Specific representations participate in reconstruction.
3. Last 10 epochs: jointly optimize Reconstruction, GRL, Pair, PAM, Structure, and Cross-view Loss.
4. Use the Cross Predictor to complete the missing views with 128-dimensional latent representations.
5. Average and normalize the two completed Common representations, then perform KMeans clustering.

The current version has removed the Orthogonality, Cluster Head CE, and Balance branches, which were never activated in the final stabilized training schedule.
## Running 

```powershell
python train.py --seed 61
```



```powershell
powershell -ExecutionPolicy Bypass -File .\run_train.ps1 -Seed 61 -MaskSeed 61
```

Specify the dataset and output directory:

```powershell
powershell -ExecutionPolicy Bypass -File .\run_train.ps1 `
  -Data "./Datasets/BDGP/BDGP.mat" `
  -Output "outputs\bdgp_result"
```
`run_train.ps1` reuses the verified `lycenv` dependency environment of the current project. If this environment has already been properly activated in your terminal, you can also directly run `python train.py`.

For standard experiments, only five command-line arguments are exposed: `data/output/seed/mask-seed/missing-rate`. The network dimensions, loss weights, and teacher-related details belong to the validated stable configuration and are fixed close to the modules where they are actually used.
## Output
- `outputs/bdgp_clean/seed_61.json`: Configuration, per-epoch losses, teacher diagnostics, and final ACC/NMI/ARI/PUR results.
- `outputs/bdgp_clean/seed_61.pt`: Model parameters, input dimensions, and configuration.
Ground-truth labels are not used in the structure teacher, loss functions, or checkpoint selection, and are used only for final metric reporting.