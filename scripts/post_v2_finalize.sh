#!/bin/bash
# Post-v2 finalization: wait for generate_needs_v2.py to finish,
# then regenerate vectors for newly processed companies.
set -e
cd "$(dirname "$0")/.."

echo "Waiting for generate_needs_v2.py to complete..."
while ps aux | grep -q "[g]enerate_needs_v2"; do
    v2=$(sqlite3 data/matcher.db "SELECT SUM(CASE WHEN needs_generated_at != '2026-04-10T00:23:17' AND estimated_needs IS NOT NULL THEN 1 ELSE 0 END) FROM companies WHERE midterm_plan_text IS NOT NULL AND LENGTH(midterm_plan_text) > 100;" 2>/dev/null || echo "locked")
    echo "$(date '+%H:%M') v2_done: $v2"
    sleep 300
done

echo "=== V2 Generation Complete ==="
sqlite3 data/matcher.db "
SELECT
  COUNT(*) as total,
  SUM(CASE WHEN midterm_plan_text IS NOT NULL AND LENGTH(midterm_plan_text) > 100 THEN 1 ELSE 0 END) as has_strategy,
  SUM(CASE WHEN needs_generated_at != '2026-04-10T00:23:17' AND estimated_needs IS NOT NULL THEN 1 ELSE 0 END) as v2_needs,
  SUM(CASE WHEN needs_generated_at = '2026-04-10T00:23:17' THEN 1 ELSE 0 END) as v1_remaining
FROM companies;"

echo "=== Regenerating vectors for v2 needs ==="
python3 scripts/regenerate_vectors.py

echo "=== All done ==="
