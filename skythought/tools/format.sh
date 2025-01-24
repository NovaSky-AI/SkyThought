
set -e

# Get tools directory path relative to git root
TOOLS_DIR=$(git rev-parse --show-toplevel)/skythought/tools

if command -v uv >/dev/null 2>&1; then
    uv pip install -q pre-commit
else 
    pip install -q pre-commit
fi

# Hook file should be executable
HOOK_SCRIPT=$TOOLS_DIR/.githooks/pre-commit
chmod +x $HOOK_SCRIPT

git config --local core.hooksPath "$TOOLS_DIR/.githooks"
# pre-commit run --all-files always runs from the root directory. we run this only on tools/ for now. 
cd $TOOLS_DIR;
pre-commit run --files $TOOLS_DIR/* --config $TOOLS_DIR/.pre-commit-config.yaml