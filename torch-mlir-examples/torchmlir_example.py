#!/usr/bin/env python3
import shutil
import sys
from pathlib import Path
import torch  
import torch.nn as nn  
import torch_mlir 

# Try both frontends: compiler_utils and fx. Use whichever is available.
compile_module = None
fx_export_and_import = None
try:
    from torch_mlir.compiler_utils import compile_module as _cm  # newer API
    compile_module = _cm
except Exception:
    pass

try:
    from torch_mlir import fx as _fx  # FX importer
    fx_export_and_import = _fx.export_and_import
except Exception:
    pass

if not compile_module and not fx_export_and_import:
    print("ERROR: Neither torch_mlir.compiler_utils nor torch_mlir.fx is available in this install.")
    sys.exit(1)

# --- Define a tiny model ------------------------------------------------------
class SimpleModel(nn.Module):
    def forward(self, x):
        if isinstance(x, (tuple, list)): 
            x = x[0]
        return torch.relu(x @ x.T + 1.0)

m = SimpleModel().eval()
example = (torch.ones(4, 4),)  


out_dir = Path("build_artifacts")
out_dir.mkdir(exist_ok=True)

# --- Stage 1: Torch dialect MLIR ---------------------------------------------
torch_mlir_txt = None
if compile_module:
    # Produce Torch dialect (keeps ops like torch.aten.*)
    mod = compile_module(m, example, output_type="torch")
    torch_mlir_txt = mod.operation.get_asm()
else:
    # Fallback: FX importer gives you a module too
    mod = fx_export_and_import(m, example)
    torch_mlir_txt = str(mod)

(torch_file := out_dir / "model.torch.mlir").write_text(torch_mlir_txt)
print(f"Wrote Torch dialect MLIR → {torch_file}")

print("\nAll done! Artifact:")
for p in [torch_file]:
    print("  -", p)
