#!/usr/bin/env python3
"""
CLI tool for managing experiments.

Usage:
    tsgen-experiments list                    # List all experiments
    tsgen-experiments info 1                  # Show info about experiment 0001
    tsgen-experiments create my_experiment    # Create new experiment folder
"""

import argparse
from pathlib import Path

from tsgen.experiments.manager import ExperimentManager


def list_experiments(args):
    """List all experiments."""
    manager = ExperimentManager()
    experiments = manager.list_experiments()

    if not experiments:
        print("No experiments found.")
        return

    print("\n" + "="*80)
    print(f"{'#':<6} {'Name':<30} {'Model':<20} {'Status':<10}")
    print("="*80)

    for exp in experiments:
        status = "✓ Done" if exp['completed'] else "⧗ Running"
        print(f"{exp['number']:<6} {exp['name']:<30} {exp['model']:<20} {status:<10}")

    print("="*80)
    print(f"\nTotal: {len(experiments)} experiments")
    print(f"Completed: {sum(1 for e in experiments if e['completed'])}")
    print(f"In progress: {sum(1 for e in experiments if not e['completed'])}\n")


def show_info(args):
    """Show detailed info about an experiment."""
    manager = ExperimentManager()
    exp_path = manager.get_experiment_path(str(args.experiment_id))

    if not exp_path:
        print(f"Error: Experiment '{args.experiment_id}' not found.")
        return

    # Read README
    readme_path = exp_path / "README.md"
    if readme_path.exists():
        with open(readme_path, 'r') as f:
            content = f.read()
        print("\n" + content)
    else:
        print(f"No README found at {readme_path}")

    # Show file structure
    print("\n" + "="*80)
    print("Files:")
    print("="*80)
    for item in sorted(exp_path.rglob("*")):
        if item.is_file():
            rel_path = item.relative_to(exp_path)
            size = item.stat().st_size
            print(f"  {rel_path} ({size:,} bytes)")


def create_experiment(args):
    """Create a new experiment folder."""
    import yaml

    manager = ExperimentManager()

    # Get next experiment number
    exp_num = manager.get_next_experiment_number()

    # Prepare minimal config
    config = {
        'experiment_name': args.name.replace('_', ' ').title(),
        'model_type': args.model or 'unknown',
        'experiment_number': exp_num
    }

    # Create experiment
    exp_path = manager.create_experiment(
        name=args.name,
        config=config,
        description=args.description or "No description provided."
    )

    print(f"\n✓ Created experiment {exp_num:04d}_{args.name}")
    print(f"  Location: {exp_path}")
    print(f"\nNext steps:")
    print(f"  1. Edit config: {exp_path / 'config.yaml'}")
    print(f"  2. Update README: {exp_path / 'README.md'}")
    print(f"  3. Run experiment: tsgen --config {exp_path / 'config.yaml'}\n")


def open_experiment(args):
    """Open experiment directory in file browser."""
    import subprocess
    import platform

    manager = ExperimentManager()
    exp_path = manager.get_experiment_path(str(args.experiment_id))

    if not exp_path:
        print(f"Error: Experiment '{args.experiment_id}' not found.")
        return

    # Open in file browser based on OS
    system = platform.system()
    try:
        if system == 'Darwin':  # macOS
            subprocess.run(['open', exp_path])
        elif system == 'Windows':
            subprocess.run(['explorer', exp_path])
        else:  # Linux
            subprocess.run(['xdg-open', exp_path])
        print(f"Opened {exp_path}")
    except Exception as e:
        print(f"Error opening directory: {e}")
        print(f"Path: {exp_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Manage experiments for tsgen project",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  tsgen-experiments list
  tsgen-experiments info 1
  tsgen-experiments create my_experiment --model timevae --description "Test experiment"
  tsgen-experiments open 1
        """
    )

    subparsers = parser.add_subparsers(dest='command', help='Command to run')

    # List command
    list_parser = subparsers.add_parser('list', help='List all experiments')
    list_parser.set_defaults(func=list_experiments)

    # Info command
    info_parser = subparsers.add_parser('info', help='Show experiment details')
    info_parser.add_argument('experiment_id', help='Experiment number or name')
    info_parser.set_defaults(func=show_info)

    # Create command
    create_parser = subparsers.add_parser('create', help='Create new experiment')
    create_parser.add_argument('name', help='Experiment name (2-3 words with underscores)')
    create_parser.add_argument('--model', help='Model type')
    create_parser.add_argument('--description', help='Experiment description')
    create_parser.set_defaults(func=create_experiment)

    # Open command
    open_parser = subparsers.add_parser('open', help='Open experiment directory')
    open_parser.add_argument('experiment_id', help='Experiment number or name')
    open_parser.set_defaults(func=open_experiment)

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    args.func(args)


if __name__ == '__main__':
    main()
