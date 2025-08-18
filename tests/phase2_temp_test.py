import os, re, sys, json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault('TRUTHLENS_DISABLE_ML','1')

report = []
failures = []

def check(name, cond, detail=""):
    if cond:
        report.append({"name": name, "ok": True})
    else:
        report.append({"name": name, "ok": False, "detail": detail})
        failures.append(name)

# 1) Preprocess
try:
    from extractor.preprocess import normalize_whitespace, split_sentences
    norm = normalize_whitespace('  A   B  ')
    sents = split_sentences('X. Y? Z!')
    check('preprocess.normalize', norm=='A B', norm)
    check('preprocess.split', sents==['X.', 'Y?', 'Z!'], str(sents))
except Exception as e:
    check('preprocess', False, repr(e))

# 2) Claim detection
try:
    from extractor.claim_detector import is_claim
    ok1, prob1 = is_claim('The earth is flat.')
    ok2, prob2 = is_claim('What a nice day!')
    check('claim_detect.pos', ok1 and prob1>0.8, f'ok1={ok1},prob1={prob1}')
    check('claim_detect.neg', (not ok2) and prob2<=0.6, f'ok2={ok2},prob2={prob2}')
except Exception as e:
    check('claim_detect', False, repr(e))

# 3) Span extraction
try:
    from extractor.claim_extractor import extract_claim_spans
    s = 'The company fired 100 employees last month.'
    spans = extract_claim_spans(s)
    cond = isinstance(spans, list) and any('fired 100 employees last month' in sp.get('text','') for sp in spans)
    check('span_extract', cond, str(spans))
except Exception as e:
    check('span_extract', False, repr(e))

# 4) Atomicize
try:
    from extractor.atomicizer import to_atomic
    claims = to_atomic('A bought B and C in 2020', None)
    texts = [c.get('text','').strip().lower() for c in claims]
    cond = ('a bought b in 2020' in texts) and ('a bought c in 2020' in texts)
    check('atomicize', cond, str(texts))
except Exception as e:
    check('atomicize', False, repr(e))

# 5) Context
try:
    from extractor.context import analyze_context
    ctx = analyze_context('vaccines cause X', 'According to an unverified tweet, vaccines might cause X.')
    check('context.modality', ctx.get('modality')=='speculative', str(ctx))
    check('context.attribution', ctx.get('attribution')=='an unverified tweet', str(ctx))
except Exception as e:
    check('context', False, repr(e))

# 6) Ranker
try:
    from extractor.ranker import score_claim
    s1 = score_claim('5G towers cause COVID-19.')
    s2 = score_claim('I love pizza.')
    check('rank.high', s1>0.8, f's1={s1}')
    check('rank.low', s2<0.2, f's2={s2}')
except Exception as e:
    check('rank', False, repr(e))

# 7) Pipeline
try:
    from extractor.pipeline import process_text
    out = process_text('5G towers cause COVID-19. I love pizza. According to an unverified tweet, vaccines might cause X.')
    cond = isinstance(out, list) and all('checkworthiness' in c and 'context' in c for c in out)
    check('pipeline', cond, json.dumps(out, ensure_ascii=False)[:300])
except Exception as e:
    check('pipeline', False, repr(e))

res = {"report": report, "failures": failures}
print(json.dumps(res, indent=2))
if failures:
    sys.exit(1)
