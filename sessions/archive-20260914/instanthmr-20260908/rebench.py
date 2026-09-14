"""Repeat unchanged full-source speed trials; observe host load, no image output."""
import json
import subprocess
import sys
import threading
import time
from pathlib import Path
import Foundation
root=Path(__file__).parent
stop=threading.Event()
process=Foundation.NSProcessInfo.processInfo()
def monitor():
 with (root/'rebench-20260909-host.jsonl').open('x') as log:
  while not stop.is_set():
   p=subprocess.run(['ps','-axo','pid,pcpu,comm'],capture_output=True,text=True,check=True)
   rows=[]
   for line in p.stdout.splitlines()[1:]:
    f=line.strip().split(None,2)
    if len(f)==3:rows.append({'pid':int(f[0]),'cpu':float(f[1]),'name':Path(f[2]).name})
   log.write(json.dumps({'time_ns':time.perf_counter_ns(),'thermal_state':int(process.thermalState()),'low_power_mode':bool(process.isLowPowerModeEnabled()),'top':sorted(rows,key=lambda r:r['cpu'],reverse=True)[:12]})+'\n');log.flush()
   stop.wait(10)
plan=[('phone-unpaced','phone-1080p60-20260906_030954.mp4',False),('phone-paced','phone-1080p60-20260906_030954.mp4',True),('macbook-paced','macbook-720p30-20260905T175716Z.mp4',True)]
t=threading.Thread(target=monitor);t.start()
results=[]
try:
 for name,source,paced in plan:
  name='rebench-20260909-'+name
  cmd=[sys.executable,str(root/'trial.py'),str(Path('benchmarks/inputs')/source),name]
  if paced:cmd+=['--paced']
  print('START',name,flush=True)
  begin=time.perf_counter_ns()
  with (root/f'{name}.log').open('x') as log:p=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT)
  results.append({'name':name,'start_ns':begin,'end_ns':time.perf_counter_ns(),'returncode':p.returncode})
  print('END',name,p.returncode,flush=True)
  if p.returncode:break
finally:
 stop.set();t.join()
 (root/'rebench-20260909-plan.json').write_text(json.dumps(results,indent=2))
