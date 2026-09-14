import json,subprocess,sys,time,threading
from pathlib import Path
root=Path(__file__).parent
stop=threading.Event()
def monitor():
 with (root/'quiet-host.jsonl').open('x') as log:
  while not stop.is_set():
   p=subprocess.run(['ps','-axo','pid,pcpu,comm'],capture_output=True,text=True,check=True)
   rows=[]
   for line in p.stdout.splitlines()[1:]:
    fields=line.strip().split(None,2)
    if len(fields)==3:rows.append({'pid':int(fields[0]),'cpu':float(fields[1]),'name':Path(fields[2]).name})
   log.write(json.dumps({'time_ns':time.perf_counter_ns(),'top':sorted(rows,key=lambda r:r['cpu'],reverse=True)[:12]})+'\n');log.flush()
   stop.wait(10)
t=threading.Thread(target=monitor);t.start()
try:
 for name,headless in [('quiet-A1',True),('quiet-B1',False),('quiet-B2',False),('quiet-A2',True)]:
  print('START',name,flush=True)
  cmd=[sys.executable,str(root/'retry_probe.py'),name]
  if headless:cmd+=['--headless']
  with (root/f'{name}.log').open('x') as log:
   p=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT)
  print('END',name,p.returncode,flush=True)
  if p.returncode:break
finally:stop.set();t.join()
