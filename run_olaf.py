import os
import sys
import subprocess
import venv
from pathlib import Path
import platform

# Project root (where run_olaf.py lives)
ROOT_DIR = Path(__file__).resolve().parent
VENV_DIR = ROOT_DIR / ".venv"

# Packages that DO NOT include torch, torchaudio, demucs or openai-whisper
BASE_PACKAGES = [
    "PyQt6",
    "librosa",
    "matplotlib",
    "soundfile",
    "phonemizer",
]

# Additional packages required for visualization plugins (3D / neon ribbons, etc.)
VISUAL_PACKAGES = [
    "numpy",          # usually already pulled by other deps but kept explicit
    "opencv-python",  # for glow / post-processing
    "skia-python",    # for high-quality 2D drawing
    "vispy",          # for 3D GPU-accelerated, audio-reactive visualizations
    "pyqtgraph",      # for 3D GPU-accelerated, audio-reactive visualizations
    "pyopenGL",       # for 3D GPU-accelerated, audio-reactive visualizations  
    "PyOpenGL_accelerate",       # for 3D GPU-accelerated, audio-reactive visualizations    
]


# Additional packages required for the auto_clean audio script
AUTO_CLEAN_PACKAGES = [
    "pedalboard",
    "noisereduce",
    "pyloudnorm",
]

DEMUCS_PACKAGE = "demucs==4.0.1"
WHISPER_PACKAGE = "openai-whisper"
# On Windows, use the woct0rdho fork that ships Triton wheels for Windows.
# For torch 2.5.x, the recommended mapping is Triton 3.1.x,
# so we pin to <3.2 to stay in the compatible range.
TRITON_WINDOWS_PACKAGE = 'triton-windows<3.2'

# Add near other package constants
WHISPER_EXTRA_PACKAGES = [
    "tiktoken",
    "tqdm",
    "more-itertools",
]

def ensure_venv() -> Path:
    """Create .venv if needed and return the venv python executable."""
    if not VENV_DIR.exists():
        print("[olaf] Creating local virtual environment (.venv)...")
        venv.create(VENV_DIR, with_pip=True)

    if os.name == "nt":
        python_path = VENV_DIR / "Scripts" / "python.exe"
    else:
        python_path = VENV_DIR / "bin" / "python"

    if not python_path.exists():
        raise RuntimeError(f"Python in venv not found: {python_path}")

    return python_path


def run_pip(venv_python: Path, args: list[str]) -> None:
    """Run pip inside the venv, raise if it fails."""
    cmd = [str(venv_python), "-m", "pip"] + args
    print("[olaf] Running:", " ".join(cmd))
    subprocess.check_call(cmd)


def install_torch_and_torchaudio(venv_python: Path) -> None:
    """
    Install torch + torchaudio, trying CUDA builds first (Windows/Linux, cu121),
    then falling back to CPU-only if needed.
    """
    system = platform.system().lower()

    # Upgrade pip first
    run_pip(venv_python, ["install", "--upgrade", "pip"])

    # 1) Try CUDA wheels (cu121) on Windows/Linux
    if system in ("windows", "linux"):
        index_url = "https://download.pytorch.org/whl/cu121"
        try:
            print("[olaf] Trying to install torch + torchaudio with CUDA (cu121)…")
            run_pip(venv_python, ["install", "--upgrade", "torch", "torchaudio", "--index-url", index_url])
            print("[olaf] torch + torchaudio (CUDA) installed successfully.")
            return
        except subprocess.CalledProcessError as e:
            print("[olaf] Failed to install CUDA wheels:", e)
            print("[olaf] Falling back to CPU-only torch/torchaudio.")

    # 2) CPU-only fallback (including macOS)
    try:
        run_pip(venv_python, ["install", "--upgrade", "torch", "torchaudio"])
        print("[olaf] CPU-only torch + torchaudio installed successfully.")
    except subprocess.CalledProcessError as e:
        print("[olaf] ERROR: Could not install torch/torchaudio at all:", e)
        print("[olaf] Olaf will run without GPU and some features may be broken.")


def install_other_packages(venv_python: Path) -> None:
    """Install all other required packages without touching torch/torchaudio."""
    print("[olaf] Installing base packages (PyQt, librosa, matplotlib, etc.)…")
    if BASE_PACKAGES:
        run_pip(venv_python, ["install", "--upgrade", *BASE_PACKAGES])

    # Visualization-specific packages (skia, opencv, numpy, etc.)
    if VISUAL_PACKAGES:
        print("[olaf] Installing visualization packages (skia, opencv, etc.)…")
        try:
            run_pip(venv_python, ["install", "--upgrade", *VISUAL_PACKAGES])
        except subprocess.CalledProcessError as e:
            print("[olaf] WARNING: Could not install visualization packages:", e)
            print("[olaf] Some visualization plugins may not work.")

    # Auto-clean audio packages (pedalboard, noisereduce, pyloudnorm)
    if AUTO_CLEAN_PACKAGES:
        print("[olaf] Installing auto-clean audio packages (pedalboard, noisereduce, pyloudnorm)…")
        try:
            run_pip(venv_python, ["install", "--upgrade", *AUTO_CLEAN_PACKAGES])
        except subprocess.CalledProcessError as e:
            print("[olaf] WARNING: Could not install auto-clean packages:", e)
            print("[olaf] The automatic audio clean-up feature may be disabled.")

    # Install demucs WITHOUT dependencies (torch/torchaudio already installed above)
    print("[olaf] Installing demucs (without dependencies)…")
    try:
        run_pip(
            venv_python,
            ["install", "--upgrade", "--no-deps", DEMUCS_PACKAGE],
        )
    except subprocess.CalledProcessError as e:
        print("[olaf] WARNING: Could not install demucs:", e)
        print("[olaf] Stem separation may not work.")

    # Install openai-whisper WITHOUT pulling torch
    print("[olaf] Installing openai-whisper (without dependencies)…")
    try:
        run_pip(
            venv_python,
            ["install", "--upgrade", "--no-deps", WHISPER_PACKAGE],
        )
    except subprocess.CalledProcessError as e:
        print("[olaf] WARNING: Could not install openai-whisper:", e)
        print("[olaf] Vocal alignment features may not work.")

    # Install openai-whisper WITHOUT pulling torch
    print("[olaf] Installing openai-whisper (without dependencies)…")
    try:
        run_pip(venv_python, ["install", "--upgrade", "--no-deps", WHISPER_PACKAGE])
    except subprocess.CalledProcessError as e:
        print("[olaf] WARNING: Could not install openai-whisper:", e)
        print("[olaf] Vocal alignment features may not work.")

    # NEW: install Whisper runtime deps explicitly (because we used --no-deps)
    print("[olaf] Installing Whisper runtime dependencies…")
    try:
        run_pip(venv_python, ["install", "--upgrade", *WHISPER_EXTRA_PACKAGES])
    except subprocess.CalledProcessError as e:
        print("[olaf] WARNING: Could not install Whisper dependencies:", e)
        print("[olaf] Whisper may fail to import (e.g., missing tiktoken).")

    # Optional: install Triton for faster Whisper timing kernels.
    # - On Windows: use the 'triton-windows' fork that ships wheels.
    # - On Linux: try the official 'triton' package.
    system = platform.system().lower()

    if system == "windows":
        print("[olaf] Installing Triton (Windows fork: triton-windows)…")
        try:
            run_pip(
                venv_python,
                ["install", "--upgrade", TRITON_WINDOWS_PACKAGE],
            )
        except subprocess.CalledProcessError as e:
            print("[olaf] WARNING: Could not install triton-windows:", e)
            print("[olaf] Whisper will fall back to slower CPU timing kernels on Windows.")
    elif system == "linux":
        print("[olaf] Installing Triton (official Linux package)…")
        try:
            run_pip(
                venv_python,
                ["install", "--upgrade", "triton"],
            )
        except subprocess.CalledProcessError as e:
            print("[olaf] WARNING: Could not install Triton:", e)
            print("[olaf] Whisper will fall back to slower CPU timing kernels.")
    else:
        print("[olaf] Triton is not configured for this OS; skipping Triton installation.")


def install_requirements(venv_python: Path) -> None:
    print("[olaf] Updating dependencies in venv...")
    # 1) torch + torchaudio (CUDA if possible)
    install_torch_and_torchaudio(venv_python)
    # 2) everything else (without touching torch/torchaudio)
    install_other_packages(venv_python)


def main():
    venv_python = ensure_venv()
    install_requirements(venv_python)

    # Prepare environment for subprocess (so it can import olaf_app properly)
    env = os.environ.copy()
    old_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(ROOT_DIR) + (os.pathsep + old_pythonpath if old_pythonpath else "")

    args = sys.argv[1:]

    if not args or args[0] == "gui":
        # GUI mode (default if no argument)
        module = "olaf_app.gui"
        extra = args[1:] if args else []
    else:
        # CLI mode (new, list, etc.)
        module = "olaf_app"
        extra = args

    cmd = [str(venv_python), "-m", module, *extra]
    sys.exit(subprocess.call(cmd, env=env))


if __name__ == "__main__":
    main()
