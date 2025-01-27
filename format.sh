
set -e

# Get tools directory path relative to git root
GIT_ROOT=$(git rev-parse --show-toplevel)
TOOLS_RELATIVE=skythought/tools
TOOLS_DIR=$GIT_ROOT/$TOOLS_RELATIVE

if command -v uv >/dev/null 2>&1; then
    uv pip install -q pre-commit
else 
    pip install -q pre-commit
fi

# Hook file should be executable
HOOK_SCRIPT=$GIT_ROOT/.githooks/pre-commit
chmod +x $HOOK_SCRIPT

# pre-commit run --all-files always runs from the root directory. we run this only on tools/ for now. 
pre-commit run --files $GIT_ROOT/skythought/tools --config .pre-commit-config.yaml
