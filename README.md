# WarpDB

A from-scratch GPU SQL engine (C++/CUDA, no libcudf) with JIT-fused pipelines and a
residency-aware CPU/GPU placement optimizer. See [docs/PRD.md](docs/PRD.md).

**Status:** Phase 0 — toolchain + first kernel running on sm_120.

## Build (native Windows)

Prerequisites: Visual Studio 2022/2026 (C++ workload), Windows SDK, CUDA Toolkit 13.x, Python 3.10+.

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements.txt
.\build.ps1 -Test          # configure + build + run tests
.\build\release\warpdb.exe # prints GPU info and runs the hello kernel
```

Notes: CMake is pinned to 3.31 (4.4.5 mis-generated Ninja rules in our setup). `-Clean` wipes the build dir.
