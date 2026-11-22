"""Observable ground-truth training launcher for Colab.

Start with:
  python -u -m jeveration.rgbd_gt_launch fit --root ... --save-dir ...

Prints immediately *before* importing torch or the training module. Emits an
import heartbeat and one stack trace when an import hangs, then runs the
regular CLI unchanged. No full Qwen model should load in fit mode.
"""
import faulthandler
import os
import sys
import threading
import time


def main():
    started=time.monotonic()
    print(f"[BOOT] Python={sys.version.split()[0]} PID={os.getpid()} mode={sys.argv[1:] if len(sys.argv)>1 else '<none>'}",flush=True)
    done=threading.Event()

    def heartbeat():
        while not done.wait(15):
            print(f"[BOOT] Import still active ({time.monotonic()-started:.0f}s). "
                  "Awaiting training module import...",flush=True)

    watcher=threading.Thread(target=heartbeat,daemon=True)
    watcher.start()
    faulthandler.enable()
    # Helpful Python stack snapshot when loading a dependency unexpectedly hangs.
    faulthandler.dump_traceback_later(90,repeat=False,file=sys.stderr)
    try:
        print("[BOOT] Importing lightweight ground-truth training module...",flush=True)
        from .rgbd_gt_train import main as run_training
        print(f"[BOOT] Training CLI imported in {time.monotonic()-started:.1f}s",flush=True)
    finally:
        done.set()
        faulthandler.cancel_dump_traceback_later()

    run_training()


if __name__=="__main__":
    main()
