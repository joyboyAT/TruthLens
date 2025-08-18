from __future__ import annotations

from phase6_output_ux.verdict_mapping import map_verdict

if __name__ == "__main__":
	for s in [0.2, 0.55, 0.8]:
		print(map_verdict(s))
