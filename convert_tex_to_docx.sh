#!/bin/bash
# ─────────────────────────────────────────────────────────────
# Convert LaTeX papers to DOCX (for viewing in Pages/Word)
# Handles: formulas (as images), figures, tables, bibliography
# ─────────────────────────────────────────────────────────────

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
OUTPUT_DIR="$SCRIPT_DIR/output"
IBM_DIR="$SCRIPT_DIR/IBMquantum/output"

# Files to convert
TEX_FILES=(
    "$SCRIPT_DIR/paper3.tex"
)

# Check pandoc
if ! command -v pandoc &>/dev/null; then
    echo "❌ pandoc not found. Install with: brew install pandoc"
    exit 1
fi

echo "📝 Converting LaTeX → DOCX"
echo "   pandoc $(pandoc --version | head -1)"
echo ""

for TEX in "${TEX_FILES[@]}"; do
    if [ ! -f "$TEX" ]; then
        echo "⚠️  Skipping $TEX (not found)"
        continue
    fi

    BASENAME=$(basename "$TEX" .tex)
    DOCX="$SCRIPT_DIR/${BASENAME}.docx"

    echo "→ Converting: $BASENAME.tex"

    pandoc "$TEX" \
        -f latex \
        -t docx \
        --resource-path="$OUTPUT_DIR:$IBM_DIR:$SCRIPT_DIR" \
        --standalone \
        --table-of-contents \
        --number-sections \
        --citeproc 2>/dev/null || true

    # Main conversion (without citeproc if it fails)
    pandoc "$TEX" \
        -f latex \
        -t docx \
        --resource-path="$OUTPUT_DIR:$IBM_DIR:$SCRIPT_DIR" \
        --standalone \
        --table-of-contents \
        --number-sections \
        -o "$DOCX" \
        2>&1 | grep -v "^$" || true

    if [ -f "$DOCX" ]; then
        SIZE=$(du -h "$DOCX" | cut -f1)
        echo "   ✅ $DOCX ($SIZE)"
    else
        echo "   ❌ Failed to create $DOCX"
    fi
done

# Also convert paw-internal-docs copy if it exists
INTERNAL="$SCRIPT_DIR/../paw-internal-docs/paper3.tex"
if [ -f "$INTERNAL" ]; then
    echo ""
    echo "→ Converting: paw-internal-docs/paper3.tex"
    DOCX_INT="$SCRIPT_DIR/../paw-internal-docs/paper3.docx"

    pandoc "$INTERNAL" \
        -f latex \
        -t docx \
        --resource-path="$OUTPUT_DIR:$IBM_DIR:$SCRIPT_DIR" \
        --standalone \
        --table-of-contents \
        --number-sections \
        -o "$DOCX_INT" \
        2>&1 | grep -v "^$" || true

    if [ -f "$DOCX_INT" ]; then
        SIZE=$(du -h "$DOCX_INT" | cut -f1)
        echo "   ✅ $DOCX_INT ($SIZE)"
    else
        echo "   ❌ Failed to create $DOCX_INT"
    fi
fi

echo ""
echo "🎉 Done! Open .docx files with Pages or Word."
