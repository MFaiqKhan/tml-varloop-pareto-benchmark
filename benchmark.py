"""
Pareto Benchmark Harness for Thinking Machines Lab's TML-d24-varloop-research (IQ4_NL).
Sweeps loop_num_passes = [1, 2, 3, 4, 5] across the 35 hardest tasks in HumanEval+
using llama.cpp's llama-server with explicit dual-GPU MoE tensor overrides.
"""

import argparse
import os
import re
import subprocess
import sys
import tempfile
import time
import pandas as pd
import requests
from datasets import load_dataset
from openai import OpenAI


def get_total_vram_mb() -> int:
    out = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"]
    ).decode().strip().splitlines()
    return sum(int(x) for x in out)


def wait_for_server(proc: subprocess.Popen, port: int, log_path: str, timeout: int = 300) -> None:
    t0 = time.time()
    while time.time() - t0 < timeout:
        if proc.poll() is not None:
            with open(log_path) as lf:
                raise RuntimeError(f"llama-server exited early (code {proc.returncode}):\n{lf.read()[-2000:]}")
        try:
            if requests.get(f"http://127.0.0.1:{port}/health", timeout=2).status_code == 200:
                return
        except Exception:
            pass
        time.sleep(2)
    with open(log_path) as lf:
        raise RuntimeError(f"Server startup timed out after {timeout}s:\n{lf.read()[-2000:]}")


def check_solution_safe(raw_text: str, test_code: str, entry_point: str, timeout_sec: int = 5) -> bool:
    clean = re.sub(r"<think>.*?</think>", "", raw_text, flags=re.DOTALL).strip()
    m = re.search(r"```python\s*(.*?)```", clean, flags=re.DOTALL)
    code = m.group(1) if m else clean
    full_script = f"{code}\n\n{test_code}\n\ncheck({entry_point})\n"

    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as tf:
        tf.write(full_script)
        tmp_path = tf.name

    try:
        res = subprocess.run([sys.executable, tmp_path], timeout=timeout_sec, capture_output=True)
        return res.returncode == 0
    except Exception:
        return False
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def run_benchmark(model_path: str, server_bin: str, num_tasks: int, port: int, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    server_dir = os.path.dirname(os.path.abspath(server_bin))

    server_env = os.environ.copy()
    server_env["LD_LIBRARY_PATH"] = f"{server_dir}:{server_env.get('LD_LIBRARY_PATH', '')}"

    ds = load_dataset("evalplus/humanevalplus", split="test")
    tasks = sorted(ds, key=lambda x: len(x["canonical_solution"]), reverse=True)[:num_tasks]

    all_results = []

    for loops in [1, 2, 3, 4, 5]:
        print(f"\n{'=' * 55}\nRunning {num_tasks}-Task Sweep | loop_num_passes = {loops}\n{'=' * 55}")
        log_path = os.path.join(out_dir, f"server_{loops}.log")

        # Explicitly route layers 12-23 fused 3D MoE expert tensors to CUDA1 to prevent CUDA0 OOM
        cmd = [
            server_bin, "-m", model_path,
            "--override-kv", f"inkling2_loop.loop.count=int:{loops}",
            "--fit", "off",
            "-ngl", "99",
            "--split-mode", "layer",
            "--tensor-split", "1,1",
            "-ot", r"blk\.(1[2-9]|2[0-3])\.ffn_.*_exps=CUDA1",
            "-c", "4096",
            "--jinja", "--temp", "1.0", "--port", str(port),
        ]

        log_f = open(log_path, "w")
        proc = subprocess.Popen(cmd, stdout=log_f, stderr=subprocess.STDOUT, env=server_env, cwd=server_dir)

        try:
            wait_for_server(proc, port, log_path)
            vram_mb = get_total_vram_mb()
            print(f"Server live | Total Dual-GPU VRAM: {vram_mb} MB ({vram_mb / 1024:.2f} GB)")

            client = OpenAI(base_url=f"http://127.0.0.1:{port}/v1", api_key="none")

            for idx, row in enumerate(tasks):
                prompt = (
                    "Complete this Python function. Output ONLY valid Python inside a ```python block.\n\n"
                    + row["prompt"]
                )
                t0 = time.perf_counter()
                resp = client.chat.completions.create(
                    model="tml-varloop",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=2048,
                    extra_body={"chat_template_kwargs": {"reasoning_effort": "medium"}},
                )
                dt = time.perf_counter() - t0

                out = resp.choices[0].message.content or ""
                comp_tokens = resp.usage.completion_tokens if resp.usage else 0

                raw_dump = resp.model_dump()
                timings = raw_dump.get("timings") or {}
                decode_tps = timings.get("predicted_per_second", comp_tokens / max(dt, 1e-5))
                prefill_tps = timings.get("prompt_per_second", 0.0)

                reasoning_text = raw_dump["choices"][0]["message"].get("reasoning_content") or ""
                if not reasoning_text:
                    think_m = re.search(r"<think>(.*?)</think>", out, flags=re.DOTALL)
                    reasoning_text = think_m.group(1) if think_m else ""

                passed = check_solution_safe(out, row["test"], row["entry_point"])

                all_results.append({
                    "loops": loops,
                    "task_id": row["task_id"],
                    "passed": int(passed),
                    "decode_tps": round(decode_tps, 2),
                    "prefill_tps": round(prefill_tps, 2),
                    "e2e_tps": round(comp_tokens / max(dt, 1e-5), 2),
                    "latency_sec": round(dt, 2),
                    "completion_tokens": comp_tokens,
                    "think_chars": len(reasoning_text),
                    "vram_gb": round(vram_mb / 1024, 2),
                })

                if (idx + 1) % 5 == 0 or idx == 0:
                    print(
                        f"[{loops}x] {idx + 1:02d}/{num_tasks} | {row['task_id']} | "
                        f"Pass={passed} | {decode_tps:.1f} tok/s | think={len(reasoning_text)}c"
                    )
        finally:
            proc.terminate()
            proc.wait(timeout=15)
            log_f.close()

        raw_df = pd.DataFrame(all_results)
        raw_df.to_csv(os.path.join(out_dir, "varloop_35_raw.csv"), index=False)

    summary = pd.DataFrame(all_results).groupby("loops").agg(
        tasks=("task_id", "count"),
        pass_at_1=("passed", lambda x: round(x.mean() * 100, 1)),
        decode_tps=("decode_tps", lambda x: round(x.mean(), 1)),
        avg_latency=("latency_sec", lambda x: round(x.mean(), 1)),
        avg_tokens=("completion_tokens", lambda x: round(x.mean(), 0)),
        avg_think_chars=("think_chars", lambda x: round(x.mean(), 0)),
        vram_gb=("vram_gb", "max"),
    ).reset_index()

    summary.to_csv(os.path.join(out_dir, "varloop_35_summary.csv"), index=False)
    print("\n=== FINAL SUMMARY ===")
    print(summary.to_markdown(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark TML-d24-varloop-research across recurrent loop counts.")
    parser.add_argument("--model", required=True, help="Path to TML-d24-varloop-research-IQ4_NL.gguf")
    parser.add_argument("--server-bin", required=True, help="Path to llama-server binary")
    parser.add_argument("--tasks", type=int, default=35, help="Number of hardest HumanEval+ tasks to evaluate")
    parser.add_argument("--port", type=int, default=8080, help="Local HTTP port for llama-server")
    parser.add_argument("--out-dir", default="results", help="Output directory for CSVs and logs")
    args = parser.parse_args()

    run_benchmark(args.model, args.server_bin, args.tasks, args.port, args.out_dir)
