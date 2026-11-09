# DDPM image synthesis

Train and apply a diffusion model to HDF5 image patches.

```bash
python -m pip install -r requirements.txt
python train_script_multiple_gpu.py --help
python apply_script_diffusion.py --help
```

Supply HDF5 roots and semicolon-delimited CSV input lists through the dataloader arguments. Set training output/log directories and the application checkpoint/output paths. Tiled application writes TIFF volumes.

[Apache-2.0 license](LICENSE) · [Attribution](THIRD_PARTY_NOTICES.md) · [Citation](CITATION.cff)
