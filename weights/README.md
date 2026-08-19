# Weights

Model checkpoints are **not** stored in git (see `.gitignore`). Put your trained
classification checkpoint here, for example:

```
weights/best.pt
```

and reference it with `--weights weights/best.pt`.

If you want to publish the checkpoint alongside the code, attach it to a GitHub
Release or deposit it on Zenodo/HuggingFace and link it from the root `README.md`
instead of committing the binary.

The class order stored inside the checkpoint (`model.names`) must match the class
order documented in `configs/material.example.yaml`.
