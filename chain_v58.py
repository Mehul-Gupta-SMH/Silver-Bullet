import subprocess, sys, os
root = os.path.dirname(os.path.abspath(__file__))
python = os.path.join(root, ".sbvenv", "Scripts", "python.exe")

modes = [
    ("reference-vs-generated",  "train_rvg_v58.log"),
    ("model-vs-model",          "train_mvm_v58.log"),
]

for mode, log_name in modes:
    log_path = os.path.join(root, log_name)
    print(f"Starting {mode} -> {log_name}", flush=True)
    with open(log_path, "w") as fh:
        proc = subprocess.run(
            [python, "-m", "backend.train", "--mode", mode],
            stdout=fh, stderr=fh, cwd=root
        )
    print(f"Done {mode}: exit {proc.returncode}", flush=True)

print("RVG+MVM COMPLETE", flush=True)
