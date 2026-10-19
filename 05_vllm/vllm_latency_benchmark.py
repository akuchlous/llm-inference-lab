import asyncio
import json
import time
import aiohttp

URL = "http://localhost:8000/v1/chat/completions"
MODEL = "mistralai/Mistral-7B-Instruct-v0.3"
CONCURRENCY_LEVELS = [1, 2, 4, 8, 16, 32, 64, 128, 256]
MAX_TOKENS = 64

async def one_request(session, idx):
    payload = {
        "model": MODEL,
        "messages": [
            {
                "role": "user",
                "content": f"Request {idx}: Explain why KV cache matters for LLM inference in about 100 words."
            }
        ],
        "max_tokens": MAX_TOKENS,
        "temperature": 0,
        "stream": True,
    }

    start = time.perf_counter()
    first_token_time = None
    completion_tokens = 0

    async with session.post(URL, json=payload) as resp:
        resp.raise_for_status()
        async for raw_line in resp.content:
            line = raw_line.decode("utf-8").strip()
            if not line or not line.startswith("data: "):
                continue
            data = line[6:]
            if data == "[DONE]":
                break
            event = json.loads(data)
            delta = event["choices"][0].get("delta", {})
            text = delta.get("content")
            if text:
                if first_token_time is None:
                    first_token_time = time.perf_counter()
                completion_tokens += 1

    end = time.perf_counter()
    ttft = (first_token_time - start) if first_token_time else None
    total = end - start
    return ttft, total, completion_tokens

async def run_level(concurrency):
    timeout = aiohttp.ClientTimeout(total=300)
    connector = aiohttp.TCPConnector(limit=0)
    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
        start = time.perf_counter()
        tasks = [one_request(session, i) for i in range(concurrency)]
        results = await asyncio.gather(*tasks)
        wall_time = time.perf_counter() - start

    ttfts = [r[0] for r in results if r[0] is not None]
    totals = [r[1] for r in results]
    output_tokens = sum(r[2] for r in results)

    avg_ttft_ms = (sum(ttfts) / len(ttfts)) * 1000
    avg_total_s = sum(totals) / len(totals)
    p95_total_s = sorted(totals)[max(0, int(len(totals) * 0.95) - 1)]
    throughput = output_tokens / wall_time if wall_time > 0 else 0.0

    return avg_ttft_ms, avg_total_s, p95_total_s, throughput

async def main():
    print(f"{'Concurrency':>11} | {'Avg TTFT ms':>11} | {'Avg total s':>11} | {'P95 total s':>11} | {'Throughput tok/s':>16}")
    print("-" * 79)
    for c in CONCURRENCY_LEVELS:
        avg_ttft_ms, avg_total_s, p95_total_s, throughput = await run_level(c)
        print(
            f"{c:11d} | {avg_ttft_ms:11.2f} | {avg_total_s:11.2f} | "
            f"{p95_total_s:11.2f} | {throughput:16.2f}"
        )

if __name__ == "__main__":
    asyncio.run(main())
