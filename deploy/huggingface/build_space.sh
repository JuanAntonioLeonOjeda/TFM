#!/usr/bin/env bash
# Prepara en DESTINO todo lo que necesita el Space de Hugging Face: el código del último commit, el modelo
# y las radiografías de ejemplo. Después se sube con: hf upload <usuario>/<space> DESTINO . --repo-type space
#
# Uso: deploy/huggingface/build_space.sh DESTINO
#      MODEL_DIR y EXAMPLES_DIR cambian de dónde se cogen el modelo y los ejemplos (outputs/ y res/examples/)
#      CODE_ONLY=1 prepara solo el código, sin modelo ni ejemplos (lo usa el workflow deploy-space.yml, porque
#      no están en GitHub; en el Space se quedan los que ya había)
set -euo pipefail

DEST=${1:?Uso: $0 DESTINO}
ROOT=$(git -C "$(dirname "$0")" rev-parse --show-toplevel)
MODEL_DIR=${MODEL_DIR:-$ROOT/outputs}
EXAMPLES_DIR=${EXAMPLES_DIR:-$ROOT/res/examples}
MODEL_FILES=(best_phased_full_covars.pt best_phased_full_covars_covariates.json)
CODE=(api data evaluation models training infer.py requirements-api.txt frontend)

CODE_ONLY=${CODE_ONLY:-0}

if [[ $CODE_ONLY != 1 ]]; then
  for f in "${MODEL_FILES[@]}"; do
    [[ -f "$MODEL_DIR/$f" ]] || { echo "Falta $MODEL_DIR/$f" >&2; exit 1; }
  done
  [[ -f "$EXAMPLES_DIR/examples.json" ]] || { echo "Faltan los ejemplos: python -m api.examples" >&2; exit 1; }
fi
[[ -z $(git -C "$ROOT" status --porcelain -- "${CODE[@]}") ]] \
  || echo "Aviso: hay cambios sin commit en el código; se usa el último commit" >&2

mkdir -p "$DEST"
# Solo se borra lo que genera este script, para no tocar el .git de un clon del Space
for p in "${CODE[@]}" outputs res Dockerfile README.md; do rm -rf "${DEST:?}/$p"; done

git -C "$ROOT" archive HEAD "${CODE[@]}" | tar -x -C "$DEST"
cp "$ROOT/deploy/huggingface/Dockerfile" "$ROOT/deploy/huggingface/README.md" "$DEST/"
if [[ $CODE_ONLY == 1 ]]; then
  echo "Código listo en $DEST (sin modelo ni ejemplos)"
  exit 0
fi
mkdir -p "$DEST/outputs" "$DEST/res/examples"
for f in "${MODEL_FILES[@]}"; do cp "$MODEL_DIR/$f" "$DEST/outputs/"; done
cp "$EXAMPLES_DIR"/examples.json "$EXAMPLES_DIR"/*.[Dd][Cc][Mm] "$DEST/res/examples/"

echo "Space listo en $DEST ($(du -sh "$DEST" | cut -f1))"
