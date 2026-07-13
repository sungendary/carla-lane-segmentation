"""
Batch runner (approach B: runner starts/stops the CARLA server per set).

Why per-set servers: leaving one server up across many load_world() calls
leaks memory and the server eventually dies mid-run. Starting a fresh
headless server for each set keeps every set on a clean process, so long
unattended collection no longer stalls when the server crashes.

Flow per set:
  start server (headless) -> wait until it accepts connections ->
  run collect_data.py as a child process -> kill the server (whole process
  group) -> next set.

Features: resume (.done markers), failure isolation, frame-count verify.

Usage:
  python src/run_collection.py        # no manual server needed
"""
import os
import sys
import time
import signal
import subprocess

import yaml
import carla
from config import repo_path, load_config

SETS_PATH = repo_path('configs', 'sets.yaml')
COLLECT_SCRIPT = repo_path('src', 'collect_data.py')


def load_sets():
    with open(SETS_PATH, 'r') as f:
        cfg = yaml.safe_load(f)
    default_frames = cfg.get('frames', 1500)
    return [{
        'map': s['map'], 'weather': s['weather'], 'out': s['out'],
        'frames': s.get('frames', default_frames),
    } for s in cfg['sets']]


def is_done(out):
    return os.path.exists(repo_path('dataset', out, '.done'))


def mark_done(out):
    os.makedirs(repo_path('dataset', out), exist_ok=True)
    with open(repo_path('dataset', out, '.done'), 'w') as f:
        f.write('ok\n')


def frames_saved(out):
    rgb_dir = repo_path('dataset', out, 'rgb')
    if not os.path.isdir(rgb_dir):
        return 0
    return len([n for n in os.listdir(rgb_dir) if n.endswith('.png')])


def start_server(scfg):
    """Launch CARLA headless in its own process group; return the Popen."""
    cmd = [scfg['script']]
    if scfg.get('headless', True):
        cmd.append('-RenderOffScreen')
    cmd.append(f"-carla-rpc-port={scfg.get('port', 2000)}")
    # start_new_session=True puts it in a new process group so we can kill
    # the whole tree later (CarlaUE4.sh spawns child processes).
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL, start_new_session=True)
    return proc


def wait_until_ready(port, timeout):
    """Poll until the server accepts a connection, or give up."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            c = carla.Client('localhost', port)
            c.set_timeout(2.0)
            c.get_server_version()
            return True
        except Exception:
            time.sleep(2.0)
    return False


def stop_server(proc):
    """Kill the server's whole process group, escalating to SIGKILL fast."""
    if proc is None:
        return
    try:
        pgid = os.getpgid(proc.pid)
    except ProcessLookupError:
        return
    # SIGTERM first
    try:
        os.killpg(pgid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        proc.wait(timeout=8)
    except subprocess.TimeoutExpired:
        pass
    # always follow with SIGKILL to catch UE4 processes that ignore SIGTERM
    try:
        os.killpg(pgid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    try:
        proc.wait(timeout=8)
    except subprocess.TimeoutExpired:
        pass


def main():
    scfg = load_config('server')
    sets = load_sets()
    print(f"Loaded {len(sets)} sets from sets.yaml")

    done, skipped, failed = [], [], []

    for i, s in enumerate(sets, 1):
        out = s['out']
        tag = f"[{i}/{len(sets)}] {s['map']}/{s['weather']} -> {out}"

        if is_done(out):
            print(f"SKIP (done) {tag}")
            skipped.append(out)
            continue

        print(f"\n=== {tag} ===")
        server = None
        try:
            server = start_server(scfg)
            if not wait_until_ready(scfg.get('port', 2000), scfg.get('boot_timeout', 60)):
                print(f"FAILED {tag}: server did not become ready")
                failed.append((out, "server boot timeout"))
                continue

            cmd = [sys.executable, COLLECT_SCRIPT,
                   '--map', s['map'], '--weather', s['weather'],
                   '--out', out, '--frames', str(s['frames'])]
            result = subprocess.run(cmd)

            saved = frames_saved(out)
            if saved >= s['frames']:
                mark_done(out)
                done.append(out)
                if result.returncode != 0:
                    print(f"NOTE {out}: data complete ({saved}) despite "
                          f"exit {result.returncode} (teardown crash).")
            else:
                print(f"FAILED {tag}: only {saved}/{s['frames']}, exit {result.returncode}")
                failed.append((out, f"{saved}/{s['frames']} frames"))
        finally:
            stop_server(server)
            # wait until the port is actually free before the next server
            for _ in range(15):
                try:
                    c = carla.Client('localhost', scfg.get('port', 2000))
                    c.set_timeout(1.0)
                    c.get_server_version()
                    # still responding -> old server not fully dead yet
                    time.sleep(2)
                except Exception:
                    break   # port free / no server -> good to proceed
            time.sleep(2)

    print("\n" + "=" * 50)
    print(f"collected now : {len(done)}")
    print(f"skipped (done): {len(skipped)}")
    print(f"failed        : {len(failed)}")
    if failed:
        print("-" * 50)
        for out, why in failed:
            print(f"  FAIL {out}: {why}")
        print("Re-run to retry failed/remaining sets.")
    print("=" * 50)


if __name__ == '__main__':
    main()