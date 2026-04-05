import sys, json
sys.path.insert(0, '.')
from pathlib import Path
from core_pipeline import load_pdf, run_query

questions = json.loads(Path('sample_docs/questions.json').read_text())
results = {}

for doc_name, qs in questions.items():
    print(f'\n=== {doc_name} ===', flush=True)
    pdf_path = Path('sample_docs') / doc_name
    pipeline = load_pdf(pdf_path, reuse_existing=True)

    correct = 0
    total = 0
    wrong = []
    for category in ['answerable', 'unanswerable']:
        for q in qs[category]:
            expected = (category == 'answerable')
            result = run_query(pipeline, q)
            got = result.get('answerable', False)
            ok = got == expected
            if ok:
                correct += 1
            else:
                wrong.append((category, q))
            total += 1
            mark = 'OK   ' if ok else 'WRONG'
            exp_s = 'ANS  ' if expected else 'UNANS'
            got_s = 'ANS  ' if got else 'UNANS'
            print(f'  [{mark}] exp={exp_s} got={got_s} | {q[:65]}', flush=True)

    pct = 100 * correct / total
    print(f'\n  SCORE: {correct}/{total} = {pct:.1f}%', flush=True)
    if wrong:
        print('  Wrong:', flush=True)
        for cat, q in wrong:
            print(f'    [{cat}] {q}', flush=True)
    results[doc_name] = {'correct': correct, 'total': total, 'pct': pct}

print('\n\n=== FINAL SUMMARY ===', flush=True)
for d, r in results.items():
    print(f'{d}: {r["correct"]}/{r["total"]} ({r["pct"]:.1f}%)', flush=True)
