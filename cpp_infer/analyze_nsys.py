"""Analyze CUDA activity only; this is not whole-SoC utilization."""
import collections
import json
from pathlib import Path
import sqlite3
import sys

folder=Path(sys.argv[1])
db=sqlite3.connect('file:'+str(folder/'four_headless.sqlite')+'?mode=ro',uri=True)
tables={r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
strings=dict(db.execute('SELECT id,value FROM StringIds'))
def union_ns(intervals):
    intervals=sorted(intervals)
    if not intervals:return 0
    start,end=intervals[0]; total=0
    for a,b in intervals[1:]:
        if a>end: total+=end-start; start,end=a,b
        else:end=max(end,b)
    return total+end-start

kernels=list(db.execute('SELECT start,end,demangledName FROM CUPTI_ACTIVITY_KIND_KERNEL'))
groups=collections.defaultdict(list)
for a,b,name in kernels:groups[strings.get(name,str(name))].append((a,b))
kernel_intervals=[(a,b) for a,b,_ in kernels]
copies=[]
if 'CUPTI_ACTIVITY_KIND_MEMCPY' in tables:
    copies=list(db.execute('SELECT start,end,bytes,copyKind FROM CUPTI_ACTIVITY_KIND_MEMCPY'))
all_intervals=kernel_intervals+[(a,b) for a,b,_,_ in copies]
begin=min(a for a,b in all_intervals); end=max(b for a,b in all_intervals)
window=end-begin
stats=[]
for name,values in groups.items():
    total=sum(b-a for a,b in values)
    stats.append({'name':name,'calls':len(values),'total_ms':total/1e6,'mean_us':total/len(values)/1e3})
stats.sort(key=lambda x:x['total_ms'],reverse=True)
silu=[x for x in stats if 'silu<' in x['name'].lower()]
memory=[]
for kind in sorted({r[3] for r in copies}):
    selected=[r for r in copies if r[3]==kind]
    memory.append({'copyKind':kind,'calls':len(selected),'total_ms':sum(b-a for a,b,_,_ in selected)/1e6,
                   'total_bytes':sum(size for _,_,size,_ in selected)})
api=[]
if 'CUPTI_ACTIVITY_KIND_RUNTIME' in tables:
    for name,count,total in db.execute('SELECT nameId,COUNT(*),SUM(end-start) FROM CUPTI_ACTIVITY_KIND_RUNTIME GROUP BY nameId ORDER BY SUM(end-start) DESC LIMIT 12'):
        api.append({'name':strings.get(name,str(name)),'calls':count,'total_ms':total/1e6,'mean_us':total/count/1e3})
result={'activity_window_ms':window/1e6,'kernel_calls':len(kernels),
        'kernel_union_ms':union_ns(kernel_intervals)/1e6,
        'cuda_activity_union_ms':union_ns(all_intervals)/1e6,
        'cuda_activity_fraction':union_ns(all_intervals)/window,
        'silu_calls':sum(x['calls'] for x in silu),'silu_total_ms':sum(x['total_ms'] for x in silu),
        'memory':memory,'top_kernels':stats[:20],'runtime_api':api}
(folder/'nsys_analysis.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))
