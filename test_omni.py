#!/usr/bin/env python3
"""Test if omni modules are available."""

try:
    import omni
    print('[OK] omni module available')
except ImportError as e:
    print(f'[ERROR] omni not available: {e}')

try:
    import omni.isaac
    print('[OK] omni.isaac available')
except ImportError as e:
    print(f'[WARNING] omni.isaac not available: {e}')

try:
    import omni.isaac.core
    print('[OK] omni.isaac.core available')
except ImportError as e:
    print(f'[WARNING] omni.isaac.core not available: {e}')

try:
    import carb
    print('[OK] carb available')
except ImportError as e:
    print(f'[WARNING] carb not available: {e}')

print('Python check complete')
