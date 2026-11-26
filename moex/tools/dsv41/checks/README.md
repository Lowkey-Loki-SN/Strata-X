# DeepSeek-V4.1 engine checks (GPU)

The checks behind DSV41_PLAN §12 and DSV41_BACKLOG #34, #43, #50, #51 and #55, kept so they can be run again. Each needs the
GPUs free: run it in its own scope after `moex_gpus_free` (moex/env.sh), never beside the live server. Their data and logs
live in `STRATA_CHECK_DIR` (default `~/AI/Models/dsv41-1017/ref-runs/engine-checks`): `long-44k-prompt.json` (the 44K
report prompt of the cold-read A/Bs) and `v43-number.png` (a picture of "7319").

| Script | What it checks |
|---|---|
| `serve_cancel.py LOG` | #34 (b): a short read stopped on its last window, sent again: the first token's distribution equals an uncancelled run's |
| `serve_cancel_a.py LOG` | #34 (a): a long read stopped between prompt-path chunks: the engine resumes at the last checkpoint, not the chunk end |
| `ab_every.py LOG PROMPT N SEED` | one arm of a cold-read A/B (`ARM_EXE`, `ARM_ARGS` and the environment set the arm): the prompt-path line, the first token's top-8, decode speed, the loop measure |
| `pagecache.py --evict FILES` | drops files' pages from the page cache (the engram tables before a cold read) |
| `vision_followup43.py EXE TAG` | #43 through the whole server: a picture question with thinking on, then a follow-up - the follow-up must reuse the first request's last-turn checkpoint; `STRATA_CHECK_TEXT_FIRST=1` puts the question before the picture (#63: the cut at the span's end) |
| `nsys_layers.py SQLITE` | #55: a decode profile split at each layer's routing hand-off (pre / mid / wait per GPU) |
| `ck_deep.py EXE TAG` | the deep checkpoints (`--prompt-cache-deep`): a 65K-token conversation carried past 65,536 a turn at a time, then changed right after the crossing - it must resume at 65,536, not the root |
| `drivetest.py SECONDS THREADS ON OFF [CHUNK_KB [KEEP_MS]]` | #59 without the engine: the pack's records read with O_DIRECT as the stream does, nonstop or ON s of load and OFF s idle (`DRIVETEST_FILES`: other files; `DRIVETEST_PIECE`: their piece); a line a second, SLOW under 150 MB/s with the load on |
| `ck_deep_reply.py EXE TAG` | #62: a mark crossed in a REPLY gets a deep checkpoint at the next request (a snapshot) - the conversation changed after it must resume at that snapshot, past 65,536 |
| `vision_warmup.py [PICTURE]` | #73 / #58: three encoder processes (no warm-up twice, warm-up once each) - the warmed first passes must equal an unwarmed second pass; CPU-heavy |
| `eg_reader.py EXE TAG ARM` | #59: the engram rows through the n-gram reader (`reader`) or the mapped reads (`mapped`), a ~12K-token text cold and again; the arms' first-token top-8 must agree |
| `verify_q3q2u.py` | the q3q2u pack (DSV41_PLAN §16): experts.tsv's types and strides, every file's size, the copied MXFP4 downs against the native records |
| `kld_q3q2u.sh OUT`, `speed_q3q2u.sh OUT` | §16: q3q2u against the 217 GB pack - KLD on wikitext and the agent text (exact and with the route bias); a cold 44K read each |
| `serve_stop_layers.py LOG_PREFIX` | #78 (`STRATA_DSV41_STOP_LAYERS=1`): three stops inside chunks (at a read's start, after a checkpoint, inside GPU 1's last chunk) - the state goes back, the next request is bit for bit a reference engine's (one tier: `--expert-slots 1800`) |
| `serve_layer_major.py LOG_PREFIX` | #76 (`STRATA_DSV41_LAYER_MAJOR=1`): the 44K read, follow-ups from checkpoints the windows made, a stop inside a window - bit for bit with the chunk loop (`LM_EXE`, `LM_BOUNCE`) |
| `serve_lm_speed.py LOG_PREFIX ARM [K=V ...] ...` | #76's speed: each arm a fresh engine (the engram tables evicted) reading the 44K prompt cold; prints the times and checks every arm's first-token logits equal (`EXTRA_ARGS`: engine flags) |
| `serve_pic_lm.py EXE TAG` | #76 with a picture through the whole server: a ~31K-token text, then the picture - layer-major off and on, the first token bit for bit |
| `serve_same_default.py LOG_PREFIX OLD NEW` | a new build with its opt-in switches off against the build it replaces: the 44K read and a follow-up, 32 greedy tokens each, the top 8 to 5 decimals |
| `eg_fetch_bench.py TOKENS MODE [SCALES] [RECORDS]` | DSV41_PLAN §18 without the engine: a cold chunk's engram fetch as `eg_rows` reads it - all at once or `window:W` rows ahead, with or without the scale pages, N record readers meanwhile; the fetch's time and the drive's 250 ms samples |
| `floor_probe.py [SECONDS]` | §18 without the engine: lone reads (4 KB, 128 KB, a mapping's page faults) at one and two in flight - their times, and the drive's in-flight count (`/sys/block/nvme0n1/inflight`) |
| `drive_guard_test.cpp` | §18: the drive guard's rules on counters it writes (no drive, no engine) - the floor within 0.6 s, held through the slow and the in-between samples, full depth within 2 s; build line inside |
| `serve_guard.py LOG_PREFIX TODAY NEW` | §18 through the serve protocol: today's build, the new one with a request at the guard's floor and one through 8 s of the slow mode's counters - the first token's logits bit for bit, the floor's in-flight count, when the guard went down and back (`GATE_A`, `GATE_SKIP_B`) |
| `serve_guard_episode.py LOG_PREFIX NEW [HASH]` | §18 against the drive itself: the guard on the drive's counters, the engram cold, another program's lone 4 KB read every 100 ms timed meanwhile - its times at the floor and at full depth |
| `serve_guard_decode.py LOG_PREFIX TODAY NEW` | §18: the decode at the guard's floor, the route bias off - the logits hash and 32 greedy tokens bit for bit with today's build |
| `serve_short_warm.py LOG_PREFIX EXE` | §41: a conversation's short reads at the config's settings - a first turn, then `TURNS` follow-ups of `SHORT_N` new tokens, `DECODE_N` generated a turn: each read's time, drive records, waits, decode and first-token logits (`EXTRA_ENV`, `FIRST_N`; `NO_BIAS=1` for an exactness check; `STRATA_ROUTE_TRACE` records a trace for `prompt_cache_sim.py`) |

`vision_followup43.py` imports `serve_pic.py`: run it with this directory on `PYTHONPATH`.
