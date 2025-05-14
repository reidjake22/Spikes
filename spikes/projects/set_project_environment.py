import os
import pathlib
import sys
import os
import pathlib

def set_project_environment(base_dir: str, name: str = None) -> str:
    """
    Sets up a full project environment with directories, placeholder files,
    and a .gitignore entry to ignore data and configs folders.

    Args:
        base_dir (str): The base directory where the project folder will be created
        name (str, optional): The name of the project folder (defaults to 'default_project')
    """
    project_name = name or "default_project"
    project_base = os.path.join(base_dir, project_name)

    # Directory structure to create
    project_dirs = [
        "configs",
        "configs/network",
        "configs/input",
        "configs/input/filters",
        "visualisations",
        "data",
        "results",
        "misc",
        "code",
        "code/scripts",
        "code/functions",
        "code/output",
    ]

    # Create directories
    for directory in project_dirs:
        path = os.path.join(project_base, directory)
        os.makedirs(path, exist_ok=True)
        print(f"📁 Created directory: {path}")

    # Create placeholder files
    placeholder_files = {
        os.path.join(project_base, "README.md"):
            f"# Project: {project_name}\n\nProject description goes here.",
        os.path.join(project_base, "configs/network_state.md"):
            "# Add dependencies here\n",
        os.path.join(project_base, "code/scripts/experiments.ipynb"):
            "# Placeholder for notebook"
    }

    for file_path, content in placeholder_files.items():
        if not os.path.exists(file_path):
            pathlib.Path(os.path.dirname(file_path)).mkdir(parents=True, exist_ok=True)
            with open(file_path, 'w') as f:
                f.write(content)
            print(f"📝 Created file: {file_path}")

    # Prepare .gitignore entries (relative to base_dir)
    gitignore_path = os.path.join(base_dir, ".gitignore")
    relative_paths_to_ignore = [
        f"{project_name}/configs/*",
        f"{project_name}/data/*",
        f"{project_name}/code/output/*"
    ]

    entries_to_add = relative_paths_to_ignore.copy()

    if os.path.exists(gitignore_path):
        with open(gitignore_path, "r") as f:
            existing = f.read().splitlines()
        entries_to_add = [entry for entry in entries_to_add if entry not in existing]

    if entries_to_add:
        with open(gitignore_path, "a") as f:
            if os.path.getsize(gitignore_path) > 0:
                f.write("\n")
            f.write("\n".join(entries_to_add))
            f.write("\n")
        print(f"🔒 Updated .gitignore with: {', '.join(entries_to_add)}")
    else:
        print("ℹ️ .gitignore already contains all necessary entries.")

    print(f"✅ Project environment set up at: {project_base}")
    return project_base
