#!/usr/bin/env python3
"""
PLGAFormer Complete Paper Enhancement Pipeline
===============================================
Runs all new experiments and generates all improved paper artifacts.

Usage:
  python run_all_enhancements.py              # Run everything
  python run_all_enhancements.py --quick      # Quick mode (fewer epochs)
  python run_all_enhancements.py --artifacts  # Only generate artifacts
"""
import sys, os, subprocess, argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def run_script(script_path, description, quick=False):
    """Run a Python script with optional quick mode."""
    print(f'\n{"="*70}')
    print(f'RUNNING: {description}')
    print(f'{"="*70}')
    cmd = [sys.executable, '-u', str(script_path)]
    if quick:
        cmd.append('--quick')
    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
    if result.returncode != 0:
        print(f'  WARNING: {description} exited with code {result.returncode}')
    else:
        print(f'  COMPLETED: {description}')


def run_root_enhancements(quick=False, skip_gpu=False):
    """Run Exp5-Exp7 through the unified project entrypoint."""
    print(f'\n{"="*70}')
    print('RUNNING: Exp5-Exp7 via run.py --task enhancements')
    print(f'{"="*70}')
    if skip_gpu:
        print('  SKIPPED: --skip_gpu requested')
        return
    cmd = [sys.executable, '-u', str(PROJECT_ROOT / 'run.py'), '--task', 'enhancements']
    if quick:
        cmd.append('--quick')
    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
    if result.returncode != 0:
        print(f'  WARNING: enhancement experiments exited with code {result.returncode}')
    else:
        print('  COMPLETED: enhancement experiments')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--quick', action='store_true', help='Quick mode (reduced epochs)')
    parser.add_argument('--artifacts', action='store_true', help='Only generate paper artifacts')
    parser.add_argument('--skip_gpu', action='store_true', help='Skip GPU-intensive experiments')
    args = parser.parse_args()

    print('='*70)
    print('PLGAFormer Paper Enhancement Pipeline')
    print('='*70)
    print(f'  Quick mode: {args.quick}')
    print(f'  Skip GPU:   {args.skip_gpu}')
    print(f'  Artifacts only: {args.artifacts}')

    if args.artifacts:
        run_script(PROJECT_ROOT / 'experiments' / 'generate_paper_artifacts_v2.py',
                   'Generate paper artifacts (figures + tables)')
        return

    # ============ Phase 1: Paper Artifacts (no GPU needed) ============
    run_script(PROJECT_ROOT / 'experiments' / 'generate_paper_artifacts_v2.py',
               'Phase 1: Generate improved paper figures and LaTeX tables')

    # ============ Phase 2: Enhancement Experiments ============
    run_root_enhancements(quick=args.quick, skip_gpu=args.skip_gpu)

    # ============ Phase 5: Final Artifact Refresh ============
    run_script(PROJECT_ROOT / 'experiments' / 'generate_paper_artifacts_v2.py',
               'Phase 5: Refresh paper artifacts with new experiment data')

    print('\n' + '='*70)
    print('ALL ENHANCEMENTS COMPLETE')
    print('='*70)
    print(f'  Figures: experiments/paper_artifacts_v2/')
    print(f'  Tables:  experiments/paper_artifacts_v2/')
    print(f'  Results: experiments/exp5_missing_data/results/')
    print(f'           experiments/exp6_efficiency/results/')
    print(f'           experiments/exp7_longterm/results/')


if __name__ == '__main__':
    main()
