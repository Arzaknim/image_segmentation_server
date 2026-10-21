# Agent Directives & Performance Guidelines

## Fast Verification Policy
- **Fast compiler checks over release builds:** Use `cargo check` and `make -C src_cpp` to verify syntax and typing. Never run `cargo build --release` during routine development verification unless latency benchmarking is requested.
- **Synthetic/unit logic verification over model forward passes:** Verify mathematical logic, IoU, loss functions, and data transformations using fast unit tests or small synthetic arrays instead of loading full weights into GPU/CPU memory and executing sliding-window tiled inference.
- **No unbounded system searches:** Never run broad root searches (`find /`). Use direct environment lookups (`which python3`, `conda info --envs`, or project-local paths).
- **Skip live server spin-up & curl loops:** Do not launch background server processes or curl loops unless network/protocol endpoints themselves are being redesigned.
- **Default Workspace Python Environment:** Use `/home/arzakuser/miniconda3/envs/image_seg/bin/python`.

