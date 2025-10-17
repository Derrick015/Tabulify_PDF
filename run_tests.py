#!/usr/bin/env python
"""
Test runner script for AI-Powered PDF Table Extractor.

This script provides a convenient way to run tests with different options.
"""
import sys
import subprocess
import argparse


def run_command(cmd):
    """Run a command and return the exit code."""
    print(f"\n{'='*70}")
    print(f"Running: {' '.join(cmd)}")
    print('='*70)
    result = subprocess.run(cmd)
    return result.returncode


def main():
    parser = argparse.ArgumentParser(description="Run tests for PDF Table Extractor")
    parser.add_argument(
        '--coverage',
        action='store_true',
        help='Run tests with coverage report'
    )
    parser.add_argument(
        '--html',
        action='store_true',
        help='Generate HTML coverage report (implies --coverage)'
    )
    parser.add_argument(
        '--unit',
        action='store_true',
        help='Run only unit tests'
    )
    parser.add_argument(
        '--integration',
        action='store_true',
        help='Run only integration tests'
    )
    parser.add_argument(
        '--file',
        type=str,
        help='Run specific test file'
    )
    parser.add_argument(
        '--verbose',
        '-v',
        action='store_true',
        help='Verbose output'
    )
    parser.add_argument(
        '--failed',
        action='store_true',
        help='Run only previously failed tests'
    )
    parser.add_argument(
        '--parallel',
        action='store_true',
        help='Run tests in parallel (requires pytest-xdist)'
    )
    
    args = parser.parse_args()
    
    # Build pytest command
    cmd = ['pytest']
    
    # Add verbosity
    if args.verbose:
        cmd.append('-v')
    
    # Add coverage
    if args.coverage or args.html:
        cmd.extend(['--cov=src', '--cov-report=term-missing'])
        if args.html:
            cmd.append('--cov-report=html')
    
    # Add markers
    if args.unit:
        cmd.extend(['-m', 'unit'])
    elif args.integration:
        cmd.extend(['-m', 'integration'])
    
    # Add specific file
    if args.file:
        cmd.append(args.file)
    else:
        cmd.append('tests/')
    
    # Add failed tests option
    if args.failed:
        cmd.append('--lf')
    
    # Add parallel option
    if args.parallel:
        cmd.extend(['-n', 'auto'])
    
    # Run tests
    exit_code = run_command(cmd)
    
    if exit_code == 0:
        print("\n✅ All tests passed!")
        if args.html:
            print("\n📊 Coverage report generated: htmlcov/index.html")
    else:
        print("\n❌ Some tests failed!")
        sys.exit(exit_code)


if __name__ == '__main__':
    main()

