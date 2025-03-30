import os
import pathlib
import sys

def set_project_environment(base_dir: str, name: str = None):
    """
    Creates necessary folders and files for a project environment.
    
    Args:
        base_dir (str): Directory relative to which project should be created.
        name (str): Name of the project folder.
    """    
    original_sys_path = sys.path.copy()
    
    try:
        sys.path = [base_dir] + sys.path
        
        project_name = name or "default_project"
        project_base = os.path.join(base_dir, project_name)
        project_dirs = [
            "configs",
            "configs/untrained",
            "configs/trained",
            "configs/input",
            "visualisations",
            "data",
            "results",
            "misc",
            "code",
            "code/scripts",
            "code/output",
        ]

        for directory in project_dirs:
            path = os.path.join(project_base, directory)
            os.makedirs(path, exist_ok=True)
            print(f"Created directory: {path}")

        placeholder_files = {
            os.path.join(project_base, "README.md"): f"# Project: {project_name}\n\nProject description goes here.",
            os.path.join(project_base, "configs/network_state.md"): "# Add dependencies here\n",
            os.path.join(project_base, "code/scripts/experiments.ipynb"): "# Placeholder for notebook"
        }

        for file_path, content in placeholder_files.items():
            if not os.path.exists(file_path):
                pathlib.Path(os.path.dirname(file_path)).mkdir(parents=True, exist_ok=True)
                with open(file_path, 'w') as f:
                    f.write(content)
                print(f"Created file: {file_path}")

        print(f"✅ Project environment set up at: {os.path.abspath(project_base)}")
    
    finally:
        sys.path = original_sys_path
