# Development notes

## How to make releases

- Mark release in `CHANGELOG.md`
    - `git add CHANGELOG.md`
    - `git commit -m "Parepare for release X.Y.Z"`
- Modify version in `pyproject.toml` (skip this step if dynamic version is used)
- Make a new commit and tag it with `vX.Y.Z`
    - `git tag -a vX.Y.Z -m "Version X.Y.Z"`
    - `git push origin vX.Y.Z` if `origin` is the name of remote repo.
- Remove lingering `dist` (if `dist` exists)
    - `rm -rf dist`
- Run `python3 -m build`
    - first run `pip3 install build` if `build` not found
- Run `python3 -m twine upload dist/*`
    - first run `pip3 install twine` if `twine` not found
