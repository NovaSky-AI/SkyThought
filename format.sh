
set -e

# Get tools directory path relative to git root
GIT_ROOT=$(git rev-parse --show-toplevel)
TOOLS_RELATIVE=skythought/tools
TOOLS_DIR=$GIT_ROOT/$TOOLS_RELATIVE

if [ ! -d "$TOOLS_DIR" ]; then
    echo "Error: Tools directory not found at $TOOLS_DIR"
    exit 1
fi

if command -v uv >/dev/null 2>&1; then
    uv pip install -q pre-commit
else 
    pip install -q pre-commit
fi

# Hook file should be executable
HOOK_SCRIPT=$GIT_ROOT/.githooks/pre-commit
chmod +x $HOOK_SCRIPT

# git config --local core.hooksPath ".githooks"
# pre-commit run --all-files always runs from the root directory. we run this only on tools/ for now. 
<<<<<<< HEAD:format.sh
pre-commit run --files $GIT_ROOT/skythought/tools --config .pre-commit-config.yaml
=======
cd $TOOLS_DIR;
pre-commit run --files ./* --config .pre-commit-config.yaml
>>>>>>> 52a4ce7c6ff95b56188414d4228cb017069fa339:skythought/tools/format.sh
