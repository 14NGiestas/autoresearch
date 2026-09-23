#!/bin/bash
#SBATCH -J wikisource
#SBATCH -c 8
#SBATCH -t 02:00:00
#SBATCH -o /home/pauli/autoresearch/logs/wikisource_%j.log
# Wikisource PT: download (86MB) -> extract (ns0 only) -> filter -> tokenize.
set -u
cd /home/pauli/autoresearch
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
W=/tmp/ptwikisource
mkdir -p $W
DUMP=$W/ptwikisource-latest-pages-articles.xml.bz2
[ -f "$DUMP" ] || curl -L --retry 3 -A "autoresearch-research-contact:local" \
  -o "$DUMP" https://dumps.wikimedia.org/ptwikisource/latest/ptwikisource-latest-pages-articles.xml.bz2
bunzip2 -t "$DUMP" || { echo "dump corrompido"; exit 1; }
echo "== extract $(date +%H:%M)"
cd tools/wikiextractor && /home/pauli/autoresearch/.venv-numpy/bin/python3 \
  -m wikiextractor.WikiExtractor --json --no-templates --namespaces 0 \
  --processes 8 -o $W/wx_full "$DUMP" > $W/extract.log 2>&1
cd /home/pauli/autoresearch
echo "== filter $(date +%H:%M)"
.venv-numpy/bin/python3 scripts/filter_wiki.py --in $W/wx_full \
  --out $W/ws_filt.jsonl --min-bytes 2048 > $W/filter.log 2>&1
echo "== tokenize 8-way $(date +%H:%M)"
split -n l/8 -d $W/ws_filt.jsonl $W/chunk_ && \
  for f in $W/chunk_0[0-7]; do mv $f $f.jsonl; done
for i in 00 01 02 03 04 05 06 07; do
  .venv-numpy/bin/python3 scripts/tokenize_wiki_bpe.py \
    --src $W/chunk_$i.jsonl --out $W/ws_bpe_$i.txt &
done
wait
cat $W/ws_bpe_*.txt > $W/ws_bpe_all.txt
wc -l $W/ws_bpe_all.txt | sed 's/^/  obras: /'
rm -f $W/chunk_*.jsonl $W/ws_bpe_0*.txt
echo "fim wikisource $(date +%H:%M)"
