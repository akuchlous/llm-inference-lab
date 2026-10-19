import asyncio
import time
import aiohttp

URL = "http://localhost:8000/v1/chat/completions"
MODEL = "mistralai/Mistral-7B-Instruct-v0.3"
CONCURRENCY_LEVELS = [1, 2, 4, 8]
MAX_TOKENS = 64

PROMPT = (
    "Explain why KV cache matters for LLM inference performance. "
    "Keep the answer technical and concise."
)

async def one_request(session):
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": PROMPT}],
        "max_tokens": MAX_TOKENS,
        "temperature": 0,
    }

    async with session.post(URL, json=payload) as resp:
        resp.raise_for_status()
        data = await resp.json()
        return data["usage"]["completion_tokens"]

async def run_level(concurrency):
    timeout = aiohttp.ClientTimeout(total=120)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        start = time.perf_counter()
        token_counts = await asyncio.gather(
            *(one_request(session) for _ in range(concurrency))
        )
        elapsed = time.perf_counter() - start

    total_tokens = sum(token_counts)
    throughput = total_tokens / elapsed
    avg_latency = elapsed  # all requests are launched together; wall-clock completion window

    return elapsed, total_tokens, throughput, avg_latency

async def main():
    print(
        f"{'Concurrency':>11} | {'Elapsed(s)':>10} | {'Output tok':>10} | "
        f"{'Throughput tok/s':>16}"
    )
    print("-" * 60)

    for c in CONCURRENCY_LEVELS:
        elapsed, total_tokens, throughput, _ = await run_level(c)
        print(
            f"{c:11d} | {elapsed:10.2f} | {total_tokens:10d} | {throughput:16.2f}"
        )

if __name__ == "__main__":
    asyncio.run(main())
