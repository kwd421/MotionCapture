import json, time
import numpy as np
from motioncapture.recorded_display import SDLDisplay
view=SDLDisplay('MotionCapture | SDL transfer and quit verification')
im=np.zeros((694,960,3),np.uint8); im[:, :320]=(0,0,255); im[:,320:640]=(0,255,0); im[:,640:]=(255,0,0)
try:
 costs=[]
 for _ in range(120):
  start=time.perf_counter_ns(); view.show(im); view.poll(); costs.append((time.perf_counter_ns()-start)/1e6)
 observed=view.pg.image.tobytes(view.surface,'RGB')
 assert observed == im[:,:,::-1].tobytes()
 view.pg.event.post(view.pg.event.Event(view.pg.KEYDOWN,key=view.pg.K_q))
 quit_ok=False
 try: view.poll()
 except KeyboardInterrupt: quit_ok=True
 assert quit_ok
 print(json.dumps({'display':view.metadata,'pixels_equal':True,'quit_event_interrupts':quit_ok,'pump_ms_mean':float(np.mean(costs[1:])),'pump_ms_p95':float(np.percentile(costs[1:],95)),'vsync_observed':view.pg.display.is_vsync() if hasattr(view.pg.display,'is_vsync') else None}),flush=True)
finally: view.close()
assert not view.pg.display.get_init()
print('display_owner_released',flush=True)
