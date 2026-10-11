# Project Instructions

## Code Style
- Write class and function docstrings 
- Keep functions small, ideally less than 25 lines
- Avoid deeply nested conditions, exception handling, etc. Use helper functions 
- Use modern Python 3.12+ features whenever possible
- Use strong typing with dataclasses or pydantic models, avoid large untyped dicts
- Utilize protocols and ABCs to have layers of abstraction and extensibility 
- Avoid magic numbers and utilize config.py files or .env for modularity / easy changes


## Architecture
- Follow the repository pattern
- Follow test driven development:
    - define a test, verify it fails
    - write functionality
    - verify test succeeds
