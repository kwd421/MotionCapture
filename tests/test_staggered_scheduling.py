"""Task-admission tests; synthetic tasks do not measure native model speed."""
from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor

import pytest
from test_recorded_tracker_clock import recorded
from test_recorded_tracker_clock import tracker as tracker


@pytest.mark.parametrize("first", ["pose", "hands"])
def test_face_admitted_after_first_completion_and_at_most_two_calls(tracker, first):
    t, _ = tracker
    t.task_scheduling = "staggered"
    lock = threading.Lock()
    entered = {name: threading.Event() for name in ("pose", "hands", "face")}
    release = {name: threading.Event() for name in ("pose", "hands", "face")}
    active = maximum = 0
    calls = []
    errors = []
    outputs = []
    frame = recorded(90000)

    for name in entered:
        task = getattr(t, "_" + name)
        original = task.detect_for_video

        def detect(image, timestamp, *, name=name, original=original):
            nonlocal active, maximum
            with lock:
                active += 1
                maximum = max(maximum, active)
                calls.append((name, id(image), timestamp))
            entered[name].set()
            assert release[name].wait(3), f"test did not release {name}"
            try:
                return original(image, timestamp)
            finally:
                with lock:
                    active -= 1

        task.detect_for_video = detect

    def run():
        try:
            outputs.append(t.process_recorded(frame))
        except BaseException as exc:
            errors.append(exc)

    with ThreadPoolExecutor(max_workers=2) as executor:
        t._executor = executor
        runner = threading.Thread(target=run)
        runner.start()
        try:
            assert entered["hands"].wait(2) and entered["pose"].wait(2)
            assert not entered["face"].is_set()
            release[first].set()
            assert entered["face"].wait(2)
            assert maximum == 2
        finally:
            for event in release.values():
                event.set()
            runner.join(3)
    assert not runner.is_alive() and not errors
    assert len(outputs) == 1 and outputs[0].identity is frame.identity
    assert sorted(name for name, _, _ in calls) == ["face", "hands", "pose"]
    assert len({image for _, image, _ in calls}) == 1
    assert {stamp for _, _, stamp in calls} == {0}


def test_staggered_failure_drains_other_consumer_without_admitting_face(tracker):
    t, _ = tracker
    t.task_scheduling = "staggered"
    hands_started = threading.Event()
    failing = threading.Event()
    release = threading.Event()
    returned = threading.Event()
    face_called = threading.Event()
    errors = []
    original_hands = t._hands.detect_for_video

    def hands(image, timestamp):
        hands_started.set()
        assert release.wait(3)
        return original_hands(image, timestamp)

    def pose(*_):
        assert hands_started.wait(2)
        failing.set()
        raise RuntimeError("synthetic pose failure")

    def face(*_):
        face_called.set()
        raise AssertionError("face must not be admitted after known error")

    t._hands.detect_for_video = hands
    t._pose.detect_for_video = pose
    t._face.detect_for_video = face

    def run():
        try:
            t.process_recorded(recorded(0))
        except BaseException as exc:
            errors.append(exc)
        finally:
            returned.set()

    with ThreadPoolExecutor(max_workers=2) as executor:
        t._executor = executor
        runner = threading.Thread(target=run)
        runner.start()
        try:
            assert failing.wait(2)
            assert not returned.is_set()  # Other task still borrows input.
            assert not face_called.is_set()
        finally:
            release.set()
            runner.join(3)
    assert not runner.is_alive()
    assert len(errors) == 1 and "synthetic pose failure" in str(errors[0])
    assert not face_called.is_set()


@pytest.mark.parametrize("mode", ["serial", "parallel", "staggered"])
def test_each_mode_preserves_all_model_timestamps_and_missing_outputs(tracker, mode):
    t, clocks = tracker
    t.task_scheduling = mode
    with ThreadPoolExecutor(max_workers=2 if mode == "staggered" else 3) as executor:
        t._executor = executor
        for index, pts in enumerate((90000, 91517, 93033)):
            output = t.process_recorded(recorded(pts, index))
            assert output.result.detection_state == {
                "pose": "missing", "left_hand": "missing", "right_hand": "missing",
                "face": "missing", "face_blendshapes": "missing",
            }
    assert clocks == [0] * 3 + [16] * 3 + [33] * 3
