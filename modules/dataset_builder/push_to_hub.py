#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Push complete dataset to HuggingFace Hub

Push locally generated complete dataset (tasks, scenarios, goals, prompts) to HuggingFace Hub.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional, Dict, Any, Union

# Project root path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

try:
    from huggingface_hub import HfApi, upload_folder, get_token
    from huggingface_hub.utils import HfHubHTTPError

    HF_HUB_AVAILABLE = True
except ImportError:
    HF_HUB_AVAILABLE = False
    print("Warning: huggingface_hub not installed. Please run: pip install huggingface_hub")


# Default paths
DATASET_DIR_DEFAULT = os.path.join(
    os.path.dirname(__file__),
    "../../dataset",
)


def validate_dataset_directory(dataset_dir: Path, platform: str = "semantic") -> Dict[str, Any]:
    """
    Validate whether the dataset directory contains required components

    Args:
        dataset_dir: Dataset root directory path
        platform: Platform type (semantic, unreal, all)

    Returns:
        Validation result dict, containing validity and statistics
    """
    result = {
        "valid": True,
        "missing": [],
        "stats": {
            "types": [],
            "scenarios": 0,
            "goals": 0,
            "tasks": 0,
            "prompts": 0,
        },
    }

    # Determine base directory and required directories based on platform type
    if platform == "semantic":
        base_dir = dataset_dir / "semantic"
        required_dirs = ["tasks", "scenarios", "goals"]
    elif platform == "unreal":
        base_dir = dataset_dir / "unreal"
        required_dirs = ["goals"]
    elif platform == "all":
        # Validate both platforms
        semantic_result = validate_dataset_directory(dataset_dir, "semantic")
        unreal_result = validate_dataset_directory(dataset_dir, "unreal")
        
        # Merge results
        result["valid"] = semantic_result["valid"] and unreal_result["valid"]
        result["missing"] = semantic_result["missing"] + unreal_result["missing"]
        result["stats"]["types"] = list(set(semantic_result["stats"]["types"] + unreal_result["stats"]["types"]))
        result["stats"]["scenarios"] = semantic_result["stats"]["scenarios"]
        result["stats"]["goals"] = semantic_result["stats"]["goals"] + unreal_result["stats"]["goals"]
        result["stats"]["tasks"] = semantic_result["stats"]["tasks"]
        result["stats"]["prompts"] = semantic_result["stats"]["prompts"]
        return result
    else:
        raise ValueError(f"Invalid platform type: {platform}. Must be one of: semantic, unreal, all")

    # Check if base directory exists
    if not base_dir.is_dir():
        result["missing"].append(f"{base_dir.name}/")
        result["valid"] = False
        return result

    # Check required directories
    for dir_name in required_dirs:
        if not (base_dir / dir_name).is_dir():
            result["missing"].append(f"{base_dir.name}/{dir_name}/")
            result["valid"] = False

    # Check prompts directory (only needed for semantic platform, optional)
    if platform == "semantic":
        prompts_dir = base_dir / "prompts"
        has_prompts = prompts_dir.is_dir()
    else:
        has_prompts = False

    if not result["valid"]:
        return result

    # Collect dataset statistics
    # 1. Detect types (from goals directory)
    goals_dir = base_dir / "goals"
    if goals_dir.exists():
        result["stats"]["types"] = [d.name for d in goals_dir.iterdir() if d.is_dir()]

        # Count data for each type
        for type_name in result["stats"]["types"]:
            # Goals
            goals_file = goals_dir / type_name / "goals.jsonl"
            if not goals_file.exists():
                goals_file = goals_dir / type_name / "goals.json"
            if goals_file.exists():
                if goals_file.suffix == ".jsonl":
                    with open(goals_file, "r", encoding="utf-8") as f:
                        result["stats"]["goals"] += sum(1 for line in f if line.strip())
                else:
                    with open(goals_file, "r", encoding="utf-8") as f:
                        goals_data = json.load(f)
                        result["stats"]["goals"] += (
                            len(goals_data) if isinstance(goals_data, list) else 1
                        )

            # Scenarios (semantic platform only)
            if platform == "semantic":
                scenarios_dir = base_dir / "scenarios" / type_name
                if scenarios_dir.exists():
                    result["stats"]["scenarios"] += len(
                        [d for d in scenarios_dir.iterdir() if d.is_dir()]
                    )

                # Tasks (semantic platform only)
                tasks_dir = base_dir / "tasks" / type_name
                if tasks_dir.exists():
                    tasks_file = tasks_dir / "tasks.jsonl"
                    if tasks_file.exists():
                        with open(tasks_file, "r", encoding="utf-8") as f:
                            result["stats"]["tasks"] += sum(1 for line in f if line.strip())

    # Count Prompts (semantic platform only)
    if platform == "semantic" and has_prompts:
        prompts_dir = base_dir / "prompts"
        for type_dir in prompts_dir.iterdir():
            if type_dir.is_dir():
                prompts_file = type_dir / "prompts.jsonl"
                if prompts_file.exists():
                    with open(prompts_file, "r", encoding="utf-8") as f:
                        result["stats"]["prompts"] += sum(
                            1 for line in f if line.strip()
                        )

    return result


def push_dataset(
    dataset_dir: str,
    repo_id: str,
    token: Optional[str] = None,
    dataset_name: Optional[str] = None,
    description: Optional[str] = None,
    version: str = "1.0.0",
    private: bool = False,
    commit_message: Optional[str] = None,
    create_repo: bool = True,
    force: bool = False,
    revision: str = "main",
    base_revision: str = "main",
    create_branch: bool = True,
    platform: str = "semantic",
) -> str:
    """
    Push complete dataset to HuggingFace Hub

    Args:
        dataset_dir: Local dataset root directory path (containing semantic/ and/or unreal/ subdirectories)
        repo_id: HuggingFace repository ID (format: username/dataset-name)
        token: HuggingFace access token (if None, obtained from HF_TOKEN environment variable)
        dataset_name: Dataset name (for registration, if None, uses the last part of repo_id)
        description: Dataset description
        version: Version number
        private: Whether the repository is private
        commit_message: Commit message
        create_repo: Whether to create the repository if it does not exist
        revision: Target branch/Revision (default: main)
        base_revision: Base branch/commit for creating new branches (default: main)
        create_branch: Whether to create the branch if it does not exist (default: True)
        platform: Platform data to push (semantic, unreal, all)

    Returns:
        Repository ID
    """
    if not HF_HUB_AVAILABLE:
        raise ImportError("huggingface_hub not installed. Please run: pip install huggingface_hub")

    dataset_path = Path(dataset_dir)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_path}")

    # Validate dataset directory
    validation_result = validate_dataset_directory(dataset_path, platform)
    if not validation_result["valid"]:
        raise ValueError(
            f"Invalid dataset directory: {dataset_path}\n"
            f"Missing components: {', '.join(validation_result['missing'])}"
        )

    stats = validation_result["stats"]
    print(f"📊 Dataset statistics (platform: {platform}):")
    print(f"   Types: {', '.join(stats['types']) if stats['types'] else 'None'}")
    if platform in ("semantic", "all"):
        print(f"   Scenarios: {stats['scenarios']}")
    print(f"   Goals: {stats['goals']}")
    if platform in ("semantic", "all"):
        print(f"   Tasks: {stats['tasks']}")
        print(f"   Prompts: {stats['prompts']}")

    # Determine source directory based on platform type
    if platform == "semantic":
        source_path = dataset_path / "semantic"
    elif platform == "unreal":
        source_path = dataset_path / "unreal"
    else:  # all
        source_path = dataset_path

    # Determine dataset name
    if dataset_name is None:
        dataset_name = repo_id.split("/")[-1]

    # Create HfApi instance
    api = HfApi(token=token)

    # Check if repository exists
    repo_exists = False
    try:
        repo_info = api.repo_info(repo_id=repo_id, repo_type="dataset", token=token)
        repo_exists = True
        print(f"ℹ️  Repository already exists: {repo_id}")
        print(f"   Last updated: {repo_info.last_modified}")
        if hasattr(repo_info, "siblings") and repo_info.siblings:
            print(f"   File count: {len(repo_info.siblings)}")
    except Exception:
        repo_exists = False

    # If repository does not exist, create it
    if not repo_exists:
        if create_repo:
            try:
                api.create_repo(
                    repo_id=repo_id,
                    repo_type="dataset",
                    private=private,
                    exist_ok=False,
                )
                print(f"✅ Repository created: {repo_id}")
            except Exception as e:
                raise ValueError(f"Cannot create repository {repo_id}: {e}")
        else:
            raise ValueError(
                f"Repository {repo_id} does not exist, and --no-create-repo is set."
                " Please create the repository first or remove the --no-create-repo option."
            )
    else:
        # Repository already exists, notify user
        if not force:
            print(f"ℹ️  Repository {repo_id} already exists, will update existing content")
            print(f"   If content differs, a new commit will be created")
            print(f"   Use --force flag to skip this prompt")
        else:
            print(f"⚠️  Force push mode: will overwrite/update repository {repo_id} content")

    # Branch existence check + auto-create branch. Note: if revision == "main", branch creation is generally not needed.
    if revision is None or revision.strip() == "":
        revision = "main"

    # Track whether this is a freshly created branch (needs force-clean to simulate orphan)
    _freshly_created_branch = False

    # Avoid "create branch" action on main (main always exists)
    if revision != "main":
        try:
            refs = api.list_repo_refs(repo_id=repo_id, repo_type="dataset")
            existing_branches = set()
            if hasattr(refs, "branches") and refs.branches:
                existing_branches = {b.name for b in refs.branches}
            branch_exists = revision in existing_branches
        except Exception:
            # When list_repo_refs fails, handle conservatively: do not assume branch exists
            branch_exists = False

        if not branch_exists:
            if create_branch:
                try:
                    api.create_branch(
                        repo_id=repo_id,
                        repo_type="dataset",
                        branch=revision,
                        revision=base_revision,
                        exist_ok=True,
                    )
                    _freshly_created_branch = True
                    print(f"✅ Branch created: {repo_id}@{revision} (will be cleaned to orphan state)")
                except Exception as e:
                    raise ValueError(
                        f"Cannot create branch {repo_id}@{revision} (base: {base_revision}): {e}"
                    )
            else:
                raise ValueError(
                    f"Target branch {repo_id}@{revision} does not exist, and --no-create-branch is set."
                )
        else:
            print(f"ℹ️  Target branch already exists: {repo_id}@{revision}")

    # Generate description (if not provided)
    if description is None:
        if platform == "semantic":
            description = (
                f"GSI Dataset (Semantic Platform): Multi-robot planning dataset with "
                f"{stats['scenarios']} scenarios, {stats['goals']} goals, "
                f"{stats['tasks']} tasks, and {stats['prompts']} prompts"
            )
        elif platform == "unreal":
            description = (
                f"GSI Dataset (Unreal Platform): Goal definitions with "
                f"{stats['goals']} goals for Unreal Engine integration"
            )
        else:  # all
            description = (
                f"GSI Dataset: Complete multi-robot planning dataset with "
                f"{stats['scenarios']} scenarios, {stats['goals']} goals, "
                f"{stats['tasks']} tasks, and {stats['prompts']} prompts"
            )

    # Prepare README.md (if it does not exist)
    readme_file = source_path / "README.md"
    if not readme_file.exists():
        # Read prompts config info (if exists, semantic platform only)
        prompts_config = {}
        if platform in ("semantic", "all"):
            prompts_dir = source_path / "prompts" if platform == "semantic" else dataset_path / "semantic" / "prompts"
            if prompts_dir.exists():
                for type_dir in prompts_dir.iterdir():
                    if type_dir.is_dir():
                        config_file = type_dir / "config.json"
                        if config_file.exists():
                            with open(config_file, "r", encoding="utf-8") as f:
                                prompts_config = json.load(f)
                            break

        # Generate different README content based on platform type
        if platform == "unreal":
            readme_content = f"""---
license: mit
task_categories:
- text-generation
- planning
tags:
- robotics
- multi-agent
- planning
- unreal-engine
- task-planning
size_categories:
- 1K<n<10K
---

# {dataset_name}

{description or 'GSI dataset for Unreal Engine platform'}

## Dataset Description

This dataset contains goal definitions for multi-robot task planning on the Unreal Engine platform.
Scene graphs are obtained in real-time from UE5, so only goal data is included.

## Dataset Structure

```
unreal/
└── goals/
    └── {{type}}/
        └── goals.jsonl
```

## Dataset Statistics

- **Types**: {', '.join(stats['types']) if stats['types'] else 'N/A'}
- **Goals**: {stats['goals']}

## Usage

### Loading Goals

```python
from pathlib import Path
import json

# Load goals
goals_file = Path('goals/cybertown/goals.jsonl')
with open(goals_file, 'r') as f:
    goals = [json.loads(line) for line in f]

# Each goal contains:
# - id: Goal identifier (e.g., "g_48")
# - instruction: Natural language instruction
# - goal_details: Structured goal information
# - meta: Metadata about the goal
```

## Citation

If you use this dataset, please cite:

```bibtex
@dataset{{{dataset_name.lower().replace('-', '_')},
  title = {{{dataset_name}}},
  author = {{Windy Lab}},
  year = {{2025}},
  license = {{MIT}},
}}
```
"""
        else:
            readme_content = f"""---
license: mit
task_categories:
- text-generation
- planning
tags:
- robotics
- multi-agent
- planning
- multi-robot-planning
- task-planning
size_categories:
- 1K<n<10K
---

# {dataset_name}

{description or 'Complete GSI dataset for multi-robot task planning'}

## Dataset Description

This dataset contains a complete multi-robot planning dataset with scenarios, goals, tasks, and prompts.

## Dataset Structure

```
{"dataset/" if platform == "all" else ""}{"semantic/" if platform in ("semantic", "all") else ""}
├── tasks/              # Task definitions
│   └── {{type}}/
│       └── tasks.jsonl
├── scenarios/          # Scenario configurations
│   └── {{type}}/
│       └── {{scenario_id}}/
│           ├── scene_graph.json
│           ├── plans.json
│           └── scene.png
├── goals/             # Goal definitions
│   └── {{type}}/
│       └── goals.jsonl
└── prompts/           # Generated prompts (deduplicated)
    └── {{type}}/
        ├── prompts.jsonl
        ├── pool_*.json
        └── config.json
```

## Dataset Statistics

- **Types**: {', '.join(stats['types']) if stats['types'] else 'N/A'}
- **Scenarios**: {stats['scenarios']}
- **Goals**: {stats['goals']}
- **Tasks**: {stats['tasks']}
- **Prompts**: {stats['prompts']}

## Usage

### Loading the Dataset

```python
from pathlib import Path
import json

# Load tasks
tasks_file = Path('tasks/cybertown/tasks.jsonl')
with open(tasks_file, 'r') as f:
    tasks = [json.loads(line) for line in f]

# Load scenarios
scenarios_dir = Path('scenarios/cybertown')
scenarios = {{}}
for scenario_dir in scenarios_dir.iterdir():
    if scenario_dir.is_dir():
        with open(scenario_dir / 'scene_graph.json', 'r') as f:
            scenarios[scenario_dir.name] = json.load(f)

# Load goals
goals_file = Path('goals/cybertown/goals.jsonl')
with open(goals_file, 'r') as f:
    goals = [json.loads(line) for line in f]

# Load prompts (deduplicated format)
prompts_dir = Path('prompts/cybertown')
# Load config
with open(prompts_dir / 'config.json', 'r') as f:
    config = json.load(f)

# Load text pools
pools = {{}}
for pool_file in prompts_dir.glob('pool_*.json'):
    pool_name = pool_file.stem.replace('pool_', '')
    with open(pool_file, 'r') as f:
        pools[pool_name] = json.load(f)

# Load main prompts data
with open(prompts_dir / 'prompts.jsonl', 'r') as f:
    for line in f:
        record = json.loads(line)
        # Reconstruct full prompt using indices and pools
        # record contains indices like skill_set_idx, env_desc_idx, etc.
        ...
```

## Prompt Configuration

{f"- Planner Mode: {prompts_config.get('planner_mode', 'unknown')}" if prompts_config else ""}
{f"- Use Environment Model: {prompts_config.get('use_environment_model', 'unknown')}" if prompts_config else ""}

## Citation

If you use this dataset, please cite:

```bibtex
@dataset{{{dataset_name.lower().replace('-', '_')},
  title = {{{dataset_name}}},
  author = {{Windy Lab}},
  year = {{2025}},
  license = {{MIT}},
}}
```
"""
        with open(readme_file, "w", encoding="utf-8") as f:
            f.write(readme_content)
        print(f"✅ README.md created")

    # Push to Hub
    if commit_message is None:
        if repo_exists:
            commit_message = f"Update dataset v{version}"
        else:
            commit_message = f"Upload dataset v{version}"

    print(f"📤 Pushing dataset to Hub: {repo_id} ...")
    print(f"   Commit message: {commit_message}")
    delete_patterns_val = "*" if (force or _freshly_created_branch) else None

    if force:
        print("⚠️  Force sync mode enabled")

    print(f"   Target branch/Revision: {revision}")
    if revision != "main":
        print(f"   Branch creation base: {base_revision} (if branch does not exist and creation is allowed)")

    try:
        # Use upload_folder to upload the entire directory
        # HuggingFace Hub handles automatically:
        # - If files are identical, upload is skipped
        # - If files differ, they are updated
        # - Remote-only files are kept (unless delete_patterns is used)
        print(f"   Source directory: {source_path}")
        upload_folder(
            folder_path=str(source_path),
            repo_id=repo_id,
            repo_type="dataset",
            token=token,
            commit_message=commit_message,
            ignore_patterns=[".git", "__pycache__", "*.pyc", ".DS_Store"],
            delete_patterns=delete_patterns_val,
            revision=revision,
        )

        if repo_exists:
            print(f"✅ Dataset successfully updated to: https://huggingface.co/datasets/{repo_id}")
            print(f"   Note: If content has changed, Hub will create a new commit record")
        else:
            print(f"✅ Dataset successfully pushed to: https://huggingface.co/datasets/{repo_id}")

        return repo_id
    except HfHubHTTPError as e:
        print(f"❌ Push failed: {e}")
        if "403" in str(e) or "Forbidden" in str(e):
            print(f"   Hint: Please check token permissions, ensure write access")
        raise
    except Exception as e:
        print(f"❌ Push failed: {e}")
        raise


def main():
    parser = argparse.ArgumentParser(
        description="Push complete dataset to HuggingFace Hub",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
    Examples:
    # Push entire dataset using default paths
    python push_to_hub.py --repo-id username/gsi-dataset

    # Specify dataset root directory and repository ID
    python push_to_hub.py --dataset-dir dataset --repo-id username/cybertown-dataset

    # Use token from environment variable
    export HF_TOKEN=your_token_here
    python push_to_hub.py --repo-id username/gsi-dataset

    # Create private repository
    python push_to_hub.py --repo-id username/private-dataset --private

    # Push to a new branch (e.g. branch name: small)
    python push_to_hub.py --repo-id username/gsi-dataset --revision small
            """,
    )

    # Input paths
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default=DATASET_DIR_DEFAULT,
        help=f"Dataset root directory path (containing semantic/ and/or unreal/ subdirectories, default: {DATASET_DIR_DEFAULT})",
    )

    # Hub configuration
    parser.add_argument(
        "--repo-id",
        type=str,
        required=True,
        help="HuggingFace repository ID (format: username/dataset-name)",
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="HuggingFace access token (lower priority than HF_TOKEN environment variable)",
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default=None,
        help="Dataset name (if not provided, uses the last part of repo_id)",
    )
    parser.add_argument(
        "--description",
        type=str,
        default=None,
        help="Dataset description",
    )
    parser.add_argument(
        "--version",
        type=str,
        default="2.0.0",
        help="Version number",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Create private repository",
    )
    parser.add_argument(
        "--commit-message",
        type=str,
        default=None,
        help="Commit message",
    )
    parser.add_argument(
        "--no-create-repo",
        action="store_true",
        help="Do not auto-create repository if it does not exist (default will auto-create)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force push, even if repository already exists (default will update existing content)",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default="main",
        help="Target branch/Revision (default: main; e.g.: small)",
    )
    parser.add_argument(
        "--base-revision",
        type=str,
        default="main",
        help="Base branch/Revision for creating new branches (default: main)",
    )
    parser.add_argument(
        "--no-create-branch",
        action="store_true",
        help="Do not auto-create branch if target branch does not exist (default will auto-create)",
    )
    parser.add_argument(
        "--platform",
        type=str,
        choices=["semantic", "unreal", "all"],
        default="semantic",
        help="Platform data to push (default: semantic)",
    )

    args = parser.parse_args()

    try:
        # Prefer token from environment variable, fall back to command-line argument
        # Following hub_manager.py approach, supporting two environment variables
        token = args.token

        # 1. If not passed via command line, try to get from environment variable
        if token is None:
            token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")

        if token is None:
            print("ℹ️  Attempting to read local HuggingFace cached Token...")
            token = get_token()

        # 3. If still None, raise error
        if token is None:
            raise ValueError(
                "HuggingFace token not found. Please run 'huggingface-cli login' or set the HF_TOKEN environment variable."
            )

        repo_id = push_dataset(
            dataset_dir=args.dataset_dir,
            repo_id=args.repo_id,
            token=token,
            dataset_name=args.dataset_name,
            description=args.description,
            version=args.version,
            private=args.private,
            commit_message=args.commit_message,
            create_repo=not args.no_create_repo,
            force=args.force,
            revision=args.revision,
            base_revision=args.base_revision,
            create_branch=not args.no_create_branch,
            platform=args.platform,
        )

        print("\n" + "=" * 80)
        print("📊 Push complete!")
        print(f"   Repository ID: {repo_id}")
        print(f"   Hub URL: https://huggingface.co/datasets/{repo_id}")
        print("=" * 80)

        return 0

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
