#!/usr/bin/env python
"""
Standalone test runner (no pytest required).
Usage: python run_tests.py
"""
import sys, importlib.util

sys.path.insert(0, 'src')

def load_tests(path):
    spec = importlib.util.spec_from_file_location('module', path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return [(name, getattr(mod, name)) for name in dir(mod) if name.startswith('test_')]

if __name__ == '__main__':
    suites = ['tests/test_hcw_dynamics.py', 'tests/test_swarm_protocol.py']
    passed = failed = 0
    for suite in suites:
        print(f'\n{suite}')
        print('-' * 50)
        for name, fn in load_tests(suite):
            try:
                fn()
                print(f'  PASS  {name}')
                passed += 1
            except Exception as e:
                print(f'  FAIL  {name}: {e}')
                failed += 1
    print(f'\n{"="*50}')
    print(f'Result: {passed} passed, {failed} failed')
    sys.exit(failed)
