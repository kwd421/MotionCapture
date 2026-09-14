import sys
from motioncapture.recorded_display import SDLDisplay
from motioncapture.wholebody_recorded_preview import main
original=SDLDisplay.show
count=0
def show(self,image):
 global count
 original(self,image);count+=1
 if count==3:self.pg.event.post(self.pg.event.Event(self.pg.KEYDOWN,key=self.pg.K_q))
SDLDisplay.show=show
sys.argv=['preview','benchmarks/inputs/pose-rpi-dance.mp4','--provider','coreml-all','--allow-cpu-partitions','--research-only','--max-frames','12','--opencv-threads','1','--expected-ort-version','1.29.0','--display-backend','sdl','--output','sessions/opt-20260908-sdl-quit-flow.json']
raise SystemExit(main())
