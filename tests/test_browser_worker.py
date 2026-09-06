"""Real Chromium worker + HTTP + RGB, but explicit synthetic SDK (no native AI)."""
import os
import shutil
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from motioncapture.browser_hands import BrowserHandError, BrowserHandLab, BrowserHandTask

pytestmark = pytest.mark.skipif(os.environ.get("MOCAP_BROWSER_E2E") != "1",
    reason="Explicit optional real-browser test; set MOCAP_BROWSER_E2E=1")

FAKE_SDK = b'''
export const FilesetResolver = {forVisionTasks: async () => ({})};
export class HandLandmarker {
 static async createFromOptions(_, o) {
  if(o.runningMode!=="VIDEO" || o.numHands!==2 || o.minHandPresenceConfidence!==.5)
   throw Error("wrong_options");
  let closed=false;
  return {
   detectForVideo(image, timestamp) {
    if(closed) throw Error("closed_task");
    if(image.data[3]!==255) throw Error("bad_alpha");
    const p={x:image.data[0]/255,y:image.data[1]/255,z:image.data[2]/255};
    return {landmarks:[Array.from({length:21},()=>({...p}))],
      worldLandmarks:[Array.from({length:21},()=>({...p}))],
      handedness:[[{categoryName:"Left",score:1}]]};
   }, close(){closed=true;}
  };
 }
}
'''


def assets():
    root = Path(__file__).parents[1] / "tools/browser_hands"
    files = {"/app/"+p.name: p.read_bytes() for p in root.iterdir()
             if p.suffix in {".html", ".mjs", ".css"} and ".test." not in p.name}
    files["/"] = files["/app/index.html"]
    files["/sdk/vision_bundle.mjs"] = FAKE_SDK
    return files


@pytest.fixture
def browser():
    playwright = pytest.importorskip("playwright.sync_api")
    executable = shutil.which("chromium")
    if executable is None:
        pytest.skip("Chromium is not installed")
    with playwright.sync_playwright() as p:
        instance = p.chromium.launch(executable_path=executable, headless=True,
            args=["--no-sandbox", "--use-angle=swiftshader", "--enable-unsafe-swiftshader"])
        yield instance
        instance.close()


def test_actual_worker_preserves_rgb_pts_and_closes(browser):
    with BrowserHandLab(assets(), timeout=8) as lab:
        page = browser.new_page()
        page.goto(lab.url)
        lab.wait_ready()
        task = BrowserHandTask(lab, "CPU")
        rgb = np.zeros((720, 1280, 3), np.uint8)
        rgb[:] = [23, 47, 89]
        image = SimpleNamespace(numpy_view=lambda: rgb)
        for timestamp in (0, 16, 33):
            result = task.detect_for_video(image, timestamp)
            first = result.hand_landmarks[0][0]
            assert first.x == 23/255 and first.y == 47/255 and first.z == 89/255
        task.close()
        assert len(task.timings) == 3 and task.state == "closed"
        page.close()


def test_actual_software_gl_is_rejected_not_counted_as_gpu(browser):
    with BrowserHandLab(assets(), timeout=8) as lab:
        page = browser.new_page()
        page.goto(lab.url)
        lab.wait_ready()
        with pytest.raises(
            BrowserHandError,
            match="software_gpu|browser_task_failed|webgl2_unavailable|gpu_renderer_unavailable",
        ):
            BrowserHandTask(lab, "GPU")
        assert lab.failure is not None
        page.close()
