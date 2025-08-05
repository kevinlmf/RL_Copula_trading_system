import subprocess

strategies = [
    "scripts/train_cppo.py",
    "scripts/train_sac.py",
    "learning/strategy/baseline/buy_and_hold.py",
    "learning/strategy/baseline/equal_weight.py",
    "learning/strategy/baseline/copula_cvar_opt.py",
]

for script in strategies:
    print(f"\n▶️ Running {script}...")
    try:
        subprocess.run(["python", script], check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Script failed: {script}")
        print(e)

