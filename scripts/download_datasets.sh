#!/usr/bin/env bash
# Baixa os datasets REAIS usados no projeto (apenas as 6 classes do PlantAI).
#   - PlantVillage (Hughes & Salathé, 2015; Mohanty et al., 2016) — licença CC-BY-SA 3.0
#     https://github.com/spMohanty/PlantVillage-Dataset
#   - PlantDoc (Singh et al., 2020) — imagens de campo, licença CC-BY 4.0
#     https://github.com/pratikkayal/PlantDoc-Dataset
# Usa clone parcial (sparse checkout) para não baixar os ~54 mil arquivos do PlantVillage.
# Uso: bash scripts/download_datasets.sh        (requer apenas `git`)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TMP="${TMPDIR:-/tmp}/plantai_datasets"
mkdir -p "$TMP" "$ROOT/data/plantvillage" "$ROOT/data/plantdoc"

PV_CLASSES=(
  "Tomato___Early_blight" "Tomato___Late_blight" "Tomato___healthy"
  "Potato___Early_blight" "Potato___Late_blight" "Corn_(maize)___Common_rust_"
)
# PlantDoc usa outros nomes de pasta; o mapa abaixo alinha às 6 classes do PlantVillage.
PD_FOLDERS=("Tomato Early blight leaf" "Tomato leaf late blight" "Tomato leaf"
            "Potato leaf early blight" "Potato leaf late blight" "Corn rust leaf")

if [ ! -d "$TMP/pv/.git" ]; then
  git clone --filter=blob:none --no-checkout --depth 1 \
      https://github.com/spMohanty/PlantVillage-Dataset "$TMP/pv"
fi
pushd "$TMP/pv" >/dev/null
git sparse-checkout init --no-cone
printf '/raw/color/%s/\n' "${PV_CLASSES[@]}" > .git/info/sparse-checkout
printf '%s\n' '/leaf_grouping/leaf-map.json' >> .git/info/sparse-checkout
git checkout
popd >/dev/null
for c in "${PV_CLASSES[@]}"; do
  mkdir -p "$ROOT/data/plantvillage/$c"
  cp -n "$TMP/pv/raw/color/$c/"* "$ROOT/data/plantvillage/$c/"
done
cp -f "$TMP/pv/leaf_grouping/leaf-map.json" "$ROOT/data/plantvillage/leaf-map.json"

if [ ! -d "$TMP/pd/.git" ]; then
  git clone --filter=blob:none --no-checkout --depth 1 \
      https://github.com/pratikkayal/PlantDoc-Dataset "$TMP/pd"
fi
pushd "$TMP/pd" >/dev/null
git sparse-checkout init --no-cone
: > .git/info/sparse-checkout
for f in "${PD_FOLDERS[@]}"; do
  printf '/train/%s/\n/test/%s/\n' "$f" "$f" >> .git/info/sparse-checkout
done
git checkout
popd >/dev/null
for i in "${!PV_CLASSES[@]}"; do
  dst="$ROOT/data/plantdoc/${PV_CLASSES[$i]}"
  mkdir -p "$dst"
  declare -A seen_hashes=()
  while IFS= read -r existing_hash; do
    [ -n "$existing_hash" ] && seen_hashes["$existing_hash"]=1
  done < <(find "$dst" -maxdepth 1 -type f -exec sha256sum {} + 2>/dev/null | cut -c1-16)
  for split in train test; do
    src="$TMP/pd/$split/${PD_FOLDERS[$i]}"
    [ -d "$src" ] || continue
    while IFS= read -r -d '' image; do
      filename="${image##*/}"
      safe_filename="$(printf '%s' "$filename" | tr -cd '[:alnum:]._-')"
      [ -n "$safe_filename" ] || safe_filename="image"
      source_hash="$(sha256sum "$image" | cut -c1-16)"
      if [ -n "${seen_hashes[$source_hash]+x}" ]; then
        continue
      fi
      target="$dst/$safe_filename"
      if [ -e "$target" ]; then
        extension="${safe_filename##*.}"
        stem="${safe_filename%.*}"
        target="$dst/${stem}_${source_hash}.${extension}"
      fi
      cp "$image" "$target"
      seen_hashes["$source_hash"]=1
    done < <(find "$src" -maxdepth 1 -type f -print0)
  done
  unset seen_hashes
done
echo "OK: dados em $ROOT/data/plantvillage e $ROOT/data/plantdoc"
