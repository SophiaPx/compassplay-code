# Written for CompassPlay.
import json
import os
import queue
import signal
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

HOME = os.path.expanduser("~")
LAKE = os.path.join(HOME, ".elan/bin/lake")
REPL = os.environ.get("SGS_REPL_PATH", "./repl/.lake/build/bin/repl")
WS = os.environ.get("SGS_MATHLIB_PATH", "./mathlib4")
REPL = os.path.abspath(REPL)
WS = os.path.abspath(WS)
PREAMBLE = "import Mathlib\nset_option maxHeartbeats 400000\nopen BigOperators Real Nat Topology Rat"


@dataclass
class VerifyResult:
    ok: bool
    n_errors: int
    n_sorries: int
    wall: float
    raw: dict


class LeanREPL:

    def __init__(self, preamble: str = PREAMBLE, boot_timeout: float = 180.0):
        self._preamble = preamble
        self._spawn(boot_timeout)

    def _spawn(self, boot_timeout: float):
        self.proc = subprocess.Popen(
            [LAKE, "env", REPL],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            cwd=WS, text=True, start_new_session=True,
        )
        self._base_env = self._query({"cmd": self._preamble}, timeout=boot_timeout).get("env", 0)

    def _killgroup(self):
        try:
            os.killpg(os.getpgid(self.proc.pid), signal.SIGKILL)
        except Exception:
            try:
                self.proc.kill()
            except Exception:
                pass

    def _query(self, d: dict, timeout: float = 60.0) -> dict:
        self.proc.stdin.write(json.dumps(d) + "\n\n")
        self.proc.stdin.flush()
        done = threading.Event()
        timed_out = {"v": False}

        def watchdog():
            if not done.wait(timeout):
                timed_out["v"] = True
                self._killgroup()

        th = threading.Thread(target=watchdog, daemon=True)
        th.start()
        out = ""
        try:
            while True:
                line = self.proc.stdout.readline()
                if line.strip() == "" and out:
                    break
                if not line:
                    break
                out += line
        finally:
            done.set()
        if timed_out["v"]:
            raise TimeoutError("lean verify timeout")
        return json.loads(out) if out.strip() else {}

    def verify(self, lean_code: str, timeout: float = 60.0) -> VerifyResult:
        t = time.time()
        try:
            r = self._query({"cmd": lean_code, "env": self._base_env}, timeout=timeout)
        except Exception as e:
            self._killgroup()
            try:
                self._spawn(boot_timeout=180.0)
            except Exception:
                pass
            return VerifyResult(False, 1, 0, time.time() - t, {"exc": str(e)})
        errs = [m for m in r.get("messages", []) if m.get("severity") == "error"]
        sorries = r.get("sorries", [])
        ok = len(errs) == 0 and len(sorries) == 0
        return VerifyResult(ok, len(errs), len(sorries), time.time() - t, r)

    def close(self):
        try:
            self.proc.stdin.close()
        except Exception:
            pass
        self._killgroup()


class LeanPool:

    def __init__(self, n_workers: int = 8, preamble: str = PREAMBLE, boot_timeout: float = 180.0):
        self.repls = [None] * n_workers
        with ThreadPoolExecutor(max_workers=n_workers) as ex:
            def boot(i):
                self.repls[i] = LeanREPL(preamble, boot_timeout)
            list(ex.map(boot, range(n_workers)))
        self._free = queue.Queue()
        for r in self.repls:
            self._free.put(r)
        self.n_workers = n_workers

    def verify_batch(self, codes, timeout: float = 25.0):
        results = [None] * len(codes)

        def work(i_code):
            i, code = i_code
            r = self._free.get()
            try:
                results[i] = r.verify(code, timeout=timeout)
            finally:
                self._free.put(r)

        with ThreadPoolExecutor(max_workers=self.n_workers) as ex:
            list(ex.map(work, list(enumerate(codes))))
        return results

    def close(self):
        for r in self.repls:
            if r is not None:
                r.close()


def build_proof_code(theorem_statement: str, proof_body: str) -> str:
    stmt = theorem_statement.rstrip()
    if stmt.endswith(":= by"):
        return stmt + "\n" + proof_body
    if stmt.endswith(":="):
        return stmt + " by\n" + proof_body
    return stmt + "\n" + proof_body


if __name__ == "__main__":
    repl = LeanREPL()
    for name, code in [
        ("TRUE", "theorem a (n:Nat): n+0=n := by simp"),
        ("FALSE", "theorem b (n:Nat): n+1=n := by simp"),
        ("SORRY", "theorem c (n:Nat): n+1=n := by sorry"),
    ]:
        r = repl.verify(code)
        print(f"{name}: ok={r.ok} errors={r.n_errors} sorries={r.n_sorries} wall={r.wall*1000:.0f}ms")
    repl.close()
