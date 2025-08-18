"""
De-duplication & clustering for TruthLens Phase 3.

- Near-duplicate detection with SimHash (64-bit) and Hamming threshold
- Cluster by (near-duplicate simhash) and allow URL canonical hints
- Keep best-scored representative per cluster; track provenance URLs
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
import hashlib
from urllib.parse import urlparse, urlunparse


@dataclass
class DedupConfig:
	bits: int = 64
	hamming_threshold: int = 3  # <=3 bits different considered near-dup
	bucket_prefix_bits: int = 12  # use top-k bits to bucket


@dataclass
class DedupItem:
	"""Minimal item for dedup.

	- score: higher is better for representative
	"""
	evidence_id: str
	chunk_id: str
	url: str
	text: str
	score: float
	published_at_iso: Optional[str] = None


@dataclass
class Cluster:
	representative: DedupItem
	provenance_urls: List[str] = field(default_factory=list)
	members: List[DedupItem] = field(default_factory=list)


# ---- URL canonicalization ----

def canonicalize_url(u: str) -> str:
	try:
		p = urlparse(u)
		# Lowercase netloc, strip default ports, drop query/fragment
		netloc = p.netloc.lower()
		if netloc.endswith(":80"):
			netloc = netloc[:-3]
		if netloc.endswith(":443"):
			netloc = netloc[:-4]
		path = p.path.rstrip("/") or "/"
		return urlunparse((p.scheme.lower(), netloc, path, "", "", ""))
	except Exception:
		return u


# ---- SimHash ----

def _hash_token(token: str, bits: int) -> int:
	# Use md5 for stability across runs
	d = hashlib.md5(token.encode("utf-8")).digest()
	val = int.from_bytes(d, byteorder="big")
	mask = (1 << bits) - 1
	return val & mask


def _shingles(text: str, k: int = 4) -> List[str]:
	# whitespace tokenization + k-shingles
	tokens = [t for t in text.split() if t]
	if len(tokens) <= k:
		return [" ".join(tokens)] if tokens else []
	sh = [" ".join(tokens[i : i + k]) for i in range(0, len(tokens) - k + 1)]
	return sh


def simhash(text: str, bits: int = 64, k_shingle: int = 4) -> int:
	if not text:
		return 0
	weights = [0] * bits
	for sh in _shingles(text, k_shingle):
		h = _hash_token(sh, bits)
		for i in range(bits):
			if (h >> i) & 1:
				weights[i] += 1
			else:
				weights[i] -= 1
	res = 0
	for i in range(bits):
		if weights[i] >= 0:
			res |= (1 << i)
	return res


def hamming_distance(a: int, b: int) -> int:
	x = a ^ b
	# count bits
	count = 0
	while x:
		x &= x - 1
		count += 1
	return count


# ---- Union-Find ----
class UnionFind:
	def __init__(self, n: int):
		self.parent = list(range(n))
		self.rank = [0] * n
	def find(self, x: int) -> int:
		if self.parent[x] != x:
			self.parent[x] = self.find(self.parent[x])
		return self.parent[x]
	def union(self, a: int, b: int):
		ra = self.find(a)
		rb = self.find(b)
		if ra == rb:
			return
		if self.rank[ra] < self.rank[rb]:
			self.parent[ra] = rb
		elif self.rank[ra] > self.rank[rb]:
			self.parent[rb] = ra
		else:
			self.parent[rb] = ra
			self.rank[ra] += 1


# ---- Clustering ----

def deduplicate(items: List[DedupItem], cfg: Optional[DedupConfig] = None) -> List[Cluster]:
	cfg = cfg or DedupConfig()
	if not items:
		return []
	# Precompute
	sigs: List[int] = []
	curls: List[str] = []
	for it in items:
		curls.append(canonicalize_url(it.url))
		sigs.append(simhash(it.text, bits=cfg.bits))
	# Bucket by prefix bits to reduce comparisons
	buckets: Dict[int, List[int]] = {}
	prefix_mask = ((1 << cfg.bucket_prefix_bits) - 1) << (cfg.bits - cfg.bucket_prefix_bits)
	for i, sig in enumerate(sigs):
		bucket_key = (sig & prefix_mask) >> (cfg.bits - cfg.bucket_prefix_bits)
		buckets.setdefault(bucket_key, []).append(i)
	uf = UnionFind(len(items))
	for _, idxs in buckets.items():
		L = len(idxs)
		for a in range(L):
			ia = idxs[a]
			for b in range(a + 1, L):
				ib = idxs[b]
				# If same canonical URL, merge immediately
				if curls[ia] == curls[ib]:
					uf.union(ia, ib)
					continue
				# Else near-duplicate based on Hamming distance
				d = hamming_distance(sigs[ia], sigs[ib])
				if d <= cfg.hamming_threshold:
					uf.union(ia, ib)
	# Build clusters
	root_to_members: Dict[int, List[int]] = {}
	for i in range(len(items)):
		root = uf.find(i)
		root_to_members.setdefault(root, []).append(i)
	clusters: List[Cluster] = []
	for root, idxs in root_to_members.items():
		members = [items[i] for i in idxs]
		# Choose best-scored representative
		rep = max(members, key=lambda it: it.score)
		prov = [items[i].url for i in idxs]
		clusters.append(Cluster(representative=rep, provenance_urls=list(dict.fromkeys(prov)), members=members))
	return clusters
