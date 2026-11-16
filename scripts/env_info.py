"""
Prints the Python version, the platform and the versions of the installed packages as JSON,
plus the GPU that PyTorch sees, if PyTorch is installed. Runs on Python 3.8 and later.

    python scripts/env_info.py > env.json
"""

import json
import platform
import sys
from importlib import metadata


def main():
    info = {
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "executable": sys.executable,
        "packages": {},
    }
    for dist in metadata.distributions():
        name = dist.metadata["Name"]
        if name:
            info["packages"][name.lower()] = dist.version
    info["packages"] = dict(sorted(info["packages"].items()))
    if "torch" in info["packages"]:
        import torch

        info["torch_cuda"] = torch.version.cuda
        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            info["gpu"] = {
                "name": props.name,
                "memory_gib": round(props.total_memory / 2**30, 2),
                "compute_capability": f"{props.major}.{props.minor}",
                "cudnn": torch.backends.cudnn.version(),
            }
    json.dump(info, sys.stdout, indent=1)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
