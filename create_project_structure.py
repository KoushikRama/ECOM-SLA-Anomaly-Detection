from pathlib import Path

PROJECT_NAME = "."

STRUCTURE = {
    "README.md": None,
    "requirements.txt": None,
    "data": {
        "raw": {},
        "processed": {},
        "artifacts": {},
    },
    "notebooks": {},
    "models": {
        "vae.py": None,
        "saved": {},
    },
    "src": {
        "data_loader.py": None,
        "feature_engineering.py": None,
        "preprocess.py": None,
        "train.py": None,
        "infer.py": None,
        "evaluate.py": None,
        "utils.py": None,
    },
    "outputs": {
        "figures": {},
        "predictions": {},
        "logs": {},
    },
    "scripts": {
        "run_train.py": None,
        "run_infer.py": None,
    },
    "tests": {},
}


def create_structure(base_path: Path, structure: dict) -> None:
    for name, content in structure.items():
        current_path = base_path / name

        if content is None:
            current_path.parent.mkdir(parents=True, exist_ok=True)
            current_path.touch(exist_ok=True)
            print(f"Created file: {current_path}")
        elif isinstance(content, dict):
            current_path.mkdir(parents=True, exist_ok=True)
            print(f"Created directory: {current_path}")
            create_structure(current_path, content)


def add_basic_file_content(project_root: Path) -> None:
    readme_path = project_root / "README.md"
    requirements_path = project_root / "requirements.txt"

    if readme_path.exists() and readme_path.stat().st_size == 0:
        readme_path.write_text(
            "# CRUD SLA Anomaly Detection\n\n"
            "Project for anomaly detection on CRUD SLA metrics using VAE.\n",
            encoding="utf-8",
        )

    if requirements_path.exists() and requirements_path.stat().st_size == 0:
        requirements_path.write_text(
            "pandas\n"
            "numpy\n"
            "scikit-learn\n"
            "torch\n"
            "matplotlib\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    project_root = Path(PROJECT_NAME).resolve()
    project_root.mkdir(parents=True, exist_ok=True)

    create_structure(project_root, STRUCTURE)
    add_basic_file_content(project_root)

    print(f"\nProject structure created successfully in: {project_root}")