"""
Environment detection and model/data path resolution.

This is the ONLY place in the codebase that knows whether we're running
on Kaggle or locally. Everything else (api/, core/) just calls
resolve_path() and gets a working file path back, regardless of where
the code is actually running.

To add a new environment later (e.g. a cloud VM, Hugging Face Spaces),
add a branch here — nothing else needs to change.
"""

import os


def detect_environment() -> str:
    """Returns 'kaggle' or 'local' based on what's actually present
    on disk, not on any environment variable someone has to remember
    to set."""
    if os.path.isdir("/kaggle/input"):
        return "kaggle"
    return "local"


ENVIRONMENT = detect_environment()

# Where to look for model/data files when running locally.
# Override with the MODEL_DIR environment variable if you keep
# your files somewhere else.
LOCAL_MODEL_DIR = os.environ.get(
    "MODEL_DIR",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models"),
)


def resolve_path(filename: str) -> str:
    """Find filename wherever it actually lives for this environment.

    Kaggle: searches every attached input dataset (handles the fact
    that Kaggle's exact input folder naming can vary between sessions).

    Local: looks directly in LOCAL_MODEL_DIR.

    Raises FileNotFoundError with an actionable message either way —
    never fails silently.
    """
    if ENVIRONMENT == "kaggle":
        for root, _dirs, files in os.walk("/kaggle/input"):
            if filename in files:
                return os.path.join(root, filename)
        raise FileNotFoundError(
            f"'{filename}' not found under /kaggle/input. "
            f"Check that the dataset containing it is attached via Add Input."
        )

    # local
    candidate = os.path.join(LOCAL_MODEL_DIR, filename)
    if os.path.exists(candidate):
        return candidate
    raise FileNotFoundError(
        f"'{filename}' not found in '{LOCAL_MODEL_DIR}'. "
        f"Place the model/data files there, or set the MODEL_DIR "
        f"environment variable to point at the right folder."
    )


def get_device():
    """Returns the torch device to use — GPU if available, CPU otherwise.
    Import is local to this function so env.py has no hard torch
    dependency for callers that only need path resolution."""
    import torch

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


if __name__ == "__main__":
    print(f"Detected environment: {ENVIRONMENT}")
    if ENVIRONMENT == "local":
        print(f"Looking for model files in: {LOCAL_MODEL_DIR}")
