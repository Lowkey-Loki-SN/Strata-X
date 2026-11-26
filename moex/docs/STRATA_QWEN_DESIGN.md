# Strata (Qwen3.8-Flash-Next) design inventory: an audit checklist for the moex ports

**Source:** `~/AI/Strata-X` at HEAD `82f46a8` (branch `moex`). This is a read-only audit made 2026-10-07. Every cite below was read in the code, and a sample was re-verified.

**What is excluded:** the port itself, meaning `src/glm/`, `include/strata/glm/`, `moex/`, and the uncommitted edits to `pool.cpp/.hpp`, `native_expert.cpp/.hpp`, `native_mmvq.cu/.hpp`, `dequant_bf16.cu`, `moe_mmq.cu` and `CMakeLists.txt`. For those edited files, lines come from `git show HEAD:<file>` and are marked **(H)**. All other files are cited at their working-tree lines, which equal HEAD.

**The rule being audited:** a port matches every item here, unless Strata has no such mechanism, or the alternative is objectively better. In the second case the port must state the measurement.

**Aliases used in cites** (a `.hpp` partner lives under `include/strata/core/` unless stated):
`gen` src/program/generate.cpp · `srv` serve/server.py · `fe` serve/frontend.py · `rc` serve/runconfig.py · `tok` tools/strata_tokenizer.py · `su` setup.py · `cal` tools/calibrate.py ·
`es`/`esh` src/core/expert_source.cpp/.hpp · `ec`/`ech` src/core/expert_cache.cpp/.hpp · `pin`/`pinh` src/core/pinned.cu / pinned.hpp · `ses`/`sesh` src/core/session.cpp/.hpp · `lay`/`layh` src/core/layer.cpp/.hpp ·
`vf`/`vfh` src/core/verify.cpp/.hpp · `vk` src/kernels/cuda/verify_kernels.cu · `mtp`/`mtph` src/core/mtp.cpp/.hpp · `pool`/`poolh` src/kernels/cpu/pool.cpp / include/strata/kernels/cpu/pool.hpp, always (H) ·
`pf` src/prefill/prefill.cpp · `ew` src/kernels/cuda/elementwise.cu · `smp` src/kernels/cuda/sampler.cu · `DET` docs/DETAILS.md · `paper` docs/paper/Strata-Paper.pdf.

`paper` is dated 2026-09-24 (about engine 0.1.0). Where it disagrees with the code, the code wins; see §9.

**Item format:** `- [ ] **decision** — what Strata does. Default. [cite] Why: "quoted reason"`. Defaults come in two kinds:
- `eng` is the engine's built-in default (`struct Options`).
- `ship` is what `setup.py` writes into the config. **`ship` is what users run, so it is the default to match.**

## 0. Baseline and port hazards

- [ ] **Shipped engine command line.** The server runs `exe --serve` plus `cfg.args`. Setup writes these base args:
  `--pack P --native SHARD1 [--ple-gguf G] --expert-profile data/expert-profile.bin --expert-cache auto --prefill auto --spec 4 --spec-min-p 0.5 --mtp RT --max-context N`
  Setup adds these only when the condition holds:
  - `--kv int8` when the context is over 8K.
  - `--kv-resident 32768` when the context is at least 64K and the RAM fits it.
  - The low-RAM flags.
  - `--ple-io ram` on a rotational disk.
  - `--pool-workers` (hybrid CPU, or calibrated).
  - Calibrated `--pcie-frac` and `--spec-min-p`.
  - `--vram-reserve-mib`.
  - `--remote-expert-opt` with 2 or more GPUs.

  The server adds `--layer-split auto` itself when `gpu` lists 2 or more cards. [su:4773-4868; srv:607-608,1817-1930] Why: (none stated in setup for 4 / 0.5; the paper §4 method is "up to 4 tokens per window, drafts kept while their probability is at least 0.5").
- [ ] **Engine struct defaults are A/B/off values, not product values.** `spec 0`, `expert_cache 0`, `prefill 0` (the token path), `kv fp16`, `spec_min_p 0`, `max_context 4096`, `greedy`. [gen:310-640] `--serve` refuses to start without `--spec ≥2` and `--prefill`; `--mtp` is optional. [gen:5538-5549] Why: "slower, never wrong: the verify window still confirms every emitted token".
- [ ] **The suffix drafter widens the window.** `--suffix-draft 3` is on by default. It sets `mtp_max_t = spec` and `spec = min(spec+2, 8)`. With `--spec 4` that gives MTP windows of at most 4 (3 drafts) and lookup windows of at most 6 (5 drafts). [gen:614-620,2086-2092]
- [ ] **Qwen geometry is compiled in.** These must be re-derived per model, not copied:
  - Shape: H=2560, FF=640, NE=512, k=10, hc=4, hc_lr=320, and `MAXT`/`kVerifyMaxT`=8. [include/strata/kernels/cpu/expert.hpp:33-45,139; include/strata/kernels/verify_kernels.hpp:21]
  - Layer mix: a layer is QSA iff `l%4==3`, giving 12 QSA and 36 GDN layers. [include/strata/core/layout.hpp:28-65]
  - Tokens: vocab 248320; EOS 248044/248046; turn token 248045; image pad 248056. [gen:563-565,597-599,2490]
  - Refusals: the pool refuses other widths [es:2291-2296]; the native router is 512×10 only [lay:371-372]; helper GPUs need k==10 [src/core/remote_experts.cpp:246].
  - Format limits: profile pairs are u16 [ec:161-164]; `tools/make_profile.py` assumes N_LAYER=48 [:33].
  - Why: "a different `n_embd` or a different expert width is REFUSED rather than mis-indexed" [esh:18-21].
- [ ] **Exactness is the core contract.** A verify window's token t is "what plain greedy decode would produce after token t, BIT FOR BIT", so drafts change speed only. [vfh:3-8] The paper measured KL 0.022 against llama.cpp, "below the 0.058 that llama.cpp's own CPU and GPU builds differ by". [paper §6 finding 3]

## 1. Process and serving architecture

### 1.1 Split of work

- [ ] **The Python front end owns HTTP, the chat template, tokenization and output parsing. The engine takes token ids in and gives token ids out.** [fe:12-14] Why: "the template is Jinja and the formats change often; the engine boundary is token ids in, text deltas out". The engine has no tokenizer at all. [gen:9-12] Why: "C1 compares logits against llama.cpp on the SAME ids".
- [ ] **The engine is one persistent subprocess per model** (`--serve`, "plan v0.3 P8"). [gen:5032-5048]

### 1.2 Launch and lifecycle (server side)

- [ ] **Command.** `[exe, "--serve", *cfg.args, *effort_end_args]`. `--serve` is the only flag the server always adds. [srv:607-608,1850-1868]
- [ ] **Config keys that become flags.** [srv:1817-1930]
  - Two or more `gpu` → `--layer-split auto`; `layer_split` is validated (#644).
  - `split_skip_if_fits`, `vram_elastic` (#533), `expert_profile_save` (#477) map to their flags.
  - `parallel:N` → `--batch N`.
  - `effort_position:"end"` → `--tail-role-token <id of "system">` (#458). The server feature-detects this by finding the flag's bytes inside the engine binary. [srv:1791-1800]
- [ ] **Subprocess.** `Popen(stdin/stdout PIPE, text utf-8, bufsize=1)`. stderr is appended to `cfg.log`, or discarded. A start stamp goes into the log (#779). [srv:589,600-608]
- [ ] **Child environment.** `CUDA_DEVICE_ORDER=PCI_BUS_ID` and `CUDA_VISIBLE_DEVICES=cfg.gpu` (`HIP_VISIBLE_DEVICES` on HIP), then `cfg.env`, then `lib_dirs` prepended to the library path. [srv:1933-1964] Why: "#51 … CUDA's own default order (fastest first) can number the cards otherwise".
- [ ] **Windows job object.** Every child joins a kill-on-close job and is opted out of power throttling. [serve/winjob.py:1-9,67-97] Why: a crashed server left the engine running; "#691 … 36.6 -> 43.2 tok/s".
- [ ] **Readiness.**
  - The constructor blocks reading stdout, with no timeout.
  - `INFO k=v …` lines are merged into `engine.info`.
  - `READY <ctx> stop` sets `max_context`, and the `stop` word sets `can_stop`. [srv:610-621; gen:7162-7199,7231]
  - A log-tail thread prints the load phases, plus "still starting" every 20 s. [srv:394-443] Why: "people closed the window thinking it had hung".
  - A start failure quotes the last 20 log lines (#496). [srv:622-632]
- [ ] **Before loading.** The server checks the port is free before the minutes-long load (#769). [srv:4795-4798] Lazy load is opt-in and text-only. [srv:586-587,4824-4826] `before_load` is a hook with a 120 s timeout; if it fails, the failure is logged and loading continues. [srv:2339-2345]
- [ ] **`min_free_vram_mib`** (server only, default 0). Free VRAM is read via NVML or amdgpu sysfs and re-checked every 0.5 s for up to 15 s; if still short, the request gets 503 GpuBusy. If VRAM can't be read, loading proceeds. [srv:2316-2354]
- [ ] **No supervisor process.** The next request's `ensure_loaded()` restarts a dead engine: up to 3 tries, 15 s apart. The request that failed is not retried. [srv:2334-2372] Why: the dead engine's VRAM "may take a while to come back … (PR #637)".
- [ ] **Unload and shutdown.** `QUIT\n`, close stdin, wait 20 s; then terminate and wait 20 s; then kill and wait 20 s; then EngineStuck (503). [srv:1542-1579] Why: "the engine frees its memory itself - unpinning tens of GB can take a while". `idle_unload_s` defaults to 0 (never). SIGTERM is turned into KeyboardInterrupt (#96). [srv:2435-2471,4988-5010]

### 1.3 Wire protocol

- [ ] **Framing.**
  - One `\n`-terminated UTF-8 line per message; token ids are decimal, separated by commas. [srv:1335-1338]
  - The engine reads stdin with `read(2)` on a detached thread into a deque, and the main loop runs one command at a time. [gen:7076-7130] Why: "glibc's exit() … waits for stdin's lock … would hang in exit()".
  - stdout is unbuffered; everything that is not protocol goes to stderr. [gen:1504-1507] Why: "THE INTERESTING OUTPUT IS THE OUTPUT BEFORE A CRASH".
- [ ] **Commands.** [gen:7114,7753-7815,7250-7368,8015-8170]
  - `GEN <max_new> [k=v…] <ids>`
  - `GENI <max_new> [k=v…] <SVE1 embeddings file> <ids>` (vision)
  - `BGEN|BGENI <slot> …`, `BSTOP <slot>`, `BYIELD <slot>` (batch slots)
  - `STOP`
  - `QUIT`, or EOF on stdin
  - `SAVE <path>`, `RESTORE <path>`
  - `VRAM [mib]` (#533)
  - Any other line gets `ERR expected: …` and the engine continues.
- [ ] **Sampling keys go on the GEN line, between `max_new` and the ids.** [srv:835-891; gen:8022-8066]
  - The server sends only non-default values: `temperature` (only if >0), `top_p` (only if <1), `top_k` (0 or >64 is sent as 64), `min_p`, `penalty_repeat/freq/present`, `penalty_last_n` (64 when any penalty is on), `seed` (only if >0).
  - Also: `pcie_frac=` and `spec_min_p=` (calibration, no restart), `ckpt=0` (#830), `cvec=0|1`.
  - When keys are absent the engine uses greedy, top_k 20 and no penalties. Unknown keys are skipped.
  - Why: "the engine's sampled path keeps at most 64 candidates".
- [ ] **Replies.** [gen:6608-6616,8472,8599,9514,9739-9746,9944-9993; srv:1357-1398]
  - At start: `INFO …` once, then `READY <ctx> stop`.
  - Per request, in order: `RESUME n`, then `PP <reached> <total> <ms> <tok/s>` per prompt chunk, then `REUSED n`, then `T <id>` per token, then `DONE …`.
  - Errors: `ERR <msg>`.
  - Batch: `BT`, `BDONE`, `BADM`, `YIELDED`.
  - Session files: `SESSION`, `SWAIT`, `SAVED`, `RESTORED`, `SERR`, `FATAL`.
- [ ] **DONE fields.** They are append-only, so an older server reads the head and ignores the tail. [gen:9933-9951; srv:789-802]
  1. generated (includes EOS)
  2. prompt
  3. prompt_ms
  4. decode_ms
  5. finish: `stop`, `length` or `cancel`
  6. drafts accepted
  7. drafts offered
  8. reused
  9. hits
  10. lookups
  11. ram_blobs
  12. file_blobs
  13. file_mb
  14. prompt_read (#471)
  15. offloaded (#588)
  16. +5 more with `--lookup-chain`
- [ ] **EOS is enforced in the engine; stop strings in the server.**
  - The engine stops by itself on `eos_ids {248044, 248046}` in serve mode: it prints the token's `T` line, then `DONE … stop`. [gen:563-565,9739-9746]
  - The server also filters those ids out of the output. [srv:2954-2957]
  - Stop strings are server-only. A possible partial match is held back. On a match the server sends `STOP`. [srv:2043-2097,2869-2889]
  - There is no grammar decoding. JSON modes are "schema prompting followed by server validation", and a failure returns 502. [serve/structured.py:1-8]
- [ ] **`STOP` semantics.**
  - The reader thread sets an atomic flag, and the flag is cleared at the start of every command. [gen:7114,7801] Why: "a STOP that arrived between requests is stale".
  - The flag is checked per prompt chunk, per prompt verify window, and after each decode window, so the latency is at most one chunk or one window. [pf:1774; gen:8548,8618,9777]
  - (Inferred race: a STOP that arrives before its GEN is dequeued gets cleared.)
  - The server sends STOP, then drains to *this* request's DONE while holding the fifo (#481). [srv:1399-1426]
- [ ] **Cancellation.** It is checked on every PP line (#879). A watcher thread checks the client socket every 0.5 s (`MSG_PEEK`, #430/#431). [srv:3675-3698,1384-1386]
- [ ] **Context limit.** The engine sends ERR unless `n + max_new + 8 ≤ ctx`. When `max_tokens` is unset, the server uses `ctx − 8 − prompt`; `fit_max_tokens` (opt-in) clamps instead. [gen:8172-8176; srv:2782-2803] Why: "Requests longer than that are refused, never silently cut" [DET].
- [ ] **logprobs, `n`, `best_of`: not supported.** The protocol carries ids only. [serve/responses.py:391,455]
- [ ] **Server silence watchdog.** Default 300 s (`engine_silence_s`). Before the first PP the allowance grows by up to 32768/50 s of prompt; after that it is 3× the last chunk's time. On expiry the server kills the engine (#481). [srv:201-213,1343-1356,1428-1441]

### 1.4 Tokenizer, template, reasoning, tools (all in Python)

- [ ] **Tokenizer.** Strata's own pure-Python byte-level BPE, reading `vocab.json`, `merges.txt` and `token_type.json` from the pack's `tokenizer/` dir.
  - It uses the `regex` package and the QWEN35 pre-tokenizer copied from llama.cpp. [tok:1-15,46-64]
  - Words longer than 64 symbols use a heap (#268). [tok:141-204]
  - Why: it is "the reference implementation of record … A C++ port follows".
- [ ] **Template.** The pack's `tokenizer/chat_template.jinja` (exported from the GGUF), else `serve/chat_template.jinja`.
  - Rendered with jinja2 `ImmutableSandboxedEnvironment(trim_blocks, lstrip_blocks, loopcontrols)` and `tojson(ensure_ascii=False)`; `raise_exception` becomes a 400. [fe:35-60; srv:4866-4868] Why: "same Jinja settings as transformers' apply_chat_template".
  - Shape: ChatML. The generation prompt ends `<|im_start|>assistant\n<think>\n`. Past assistant turns are re-rendered with their `reasoning_content`. [serve/chat_template.jinja:57-94,159-183]
- [ ] **Prompt hygiene.** [fe:146-239,53-58,242-302]
  - Special-token text typed inside a message becomes plain text (#537, #554).
  - Empty assistant turns are dropped (#843).
  - Late system messages become user messages in place. Why: merging "would change the prompt's start and cost the conversation cache every turn".
- [ ] **Reasoning.**
  - The parser starts in the "reasoning" state, because the prompt already ends in `<think>`, and switches at `</think>`. [fe:593-594,832-854]
  - Effort mapping: none → enable_thinking=false; high/xhigh/max → xhigh. [fe:75-100]
  - Opt-ins: a thinking budget (#123) [srv:2991-3074]; loop recovery (#728) [srv:147-192].
  - On by default: 256 identical tokens in a row end the reply as `length` (#606). [srv:141-144]
- [ ] **Tool calls.** Qwen's XML format: `<tool_call><function=N><parameter=P>V</parameter>…`. [fe:458-506,553-583,603-655,873-890,532-550]
  - Values are JSON-decoded unless the tool's schema says string.
  - A `</parameter>` counts only before `<parameter=` or `</function>` (#210).
  - A call cut off by the end of output is not reported (#211).
  - A call inside the reasoning counts only when it was an act (#804).
  - A call inside a code fence or inline code is text (#1058).
  - A forced `tool_choice` works by the server writing the call's opener itself; there is no grammar.
- [ ] **Detokenization.** Incremental, through a `codecs` UTF-8 incremental decoder; partial tags are held back. [srv:2015-2040; fe:589-591] Why: the old re-decode "cost 2 ms per token after 8K tokens".

### 1.5 Prompt and prefix caching (the engine decides; the server sends the whole prompt every time)

- [ ] **The state splits in two.** [gen:1288-1299,582-585]
  - POSITIONAL: the 12 QSA layers' K/V, the pooled indexer keys and the MTP K/V. These are reused in place and never copied.
  - RUNNING: the 36 GDN states plus conv history, the QSA indexer tails (dead key, block_pos) and the PLE history, about 118 MB. These are copied into checkpoints.
  - Why: "rewinding to a position just means writing from there again".
- [ ] **Matching is an exact stored prefix, not a fuzzy LCP.** [gen:8197-8258,8206]
  - Candidates: the live token list, or any checkpoint whose ids and image keys are an exact prefix of the prompt, with length 1..n−1.
  - The longest wins. After those come idle batch slots, then parked conversations.
  - The control-vector setting must match.
  - Why: "The last prompt token always starts the next verify window".
- [ ] **Checkpoints.** [gen:582-604,8900-8984,6541-6598]
  - Taken (a) before the last `<|im_start|>`; (b) at the root, the first turn boundary at least 2048 tokens in, which stays pinned; (c) every 16384 fresh tokens.
  - 6 are kept: the root plus the rest by least recent use.
  - Opt-in extras: a tail checkpoint (`--prompt-cache-tail`) and a message-boundary checkpoint (`STRATA_CACHE_MESSAGE_BOUNDARY=1`).
  - Why (root): dropping it costs "a 30K system prompt … ~30 s at ~1,000 tok/s".
- [ ] **Live state after a request.** `prompt[0..n−1)` plus every committed token. Not kept when the request sent `ckpt=0` (#830). [gen:9021-9024,9807-9813]
- [ ] **Short reads.** A segment of at most 64 fresh tokens goes through verify windows, not the batched prompt path. [gen:605-608,8475-8490] Why: "batched path costs ~300 ms per run … a window costs ~16 ms a token".
- [ ] **Parking** (opt-in, `--conversation-cache-mib`, 4 slots, keeps 2560 MiB of RAM free). [gen:6303-6510; include/strata/core/conversation_cache.hpp:242-297]
  - On a switch or rewind, the outgoing session (state, checkpoints, used K/V pages, draft K/V) is snapshotted to host RAM.
  - Eviction is oldest-first; superseded copies of the same conversation go first (#342).
  - Capture is incremental.
- [ ] **SAVE / RESTORE** (llama.cpp's `/slots` actions).
  - File: STRSESS v1, with model and config fingerprints and a payload hash. Only the deepest checkpoint is stored.
  - Write: to a temp file, then rename and fsync; O_DIRECT on Linux.
  - Refused with a layer split, `--peer-device` or `--batch`.
  - [gen:7844-8011; src/core/conversation_file.cpp:40-49,79-82]
- [ ] **The server keeps prompts byte-stable so the cache hits.** Late system messages stay in place; the billing stamp is pinned; Responses system messages are merged; reasoning is replayed. Cached tokens are reported to clients (`cached_tokens`, `cache_read_input_tokens`). [fe:242-365; serve/responses.py:1-21; srv:3399-3405]

### 1.6 Concurrency

- [ ] **One request at a time.** `Service.fifo = threading.Lock()`: no queue bound, no FIFO guarantee, no wait timeout. The HTTP server is a `ThreadingHTTPServer`. [srv:2155,2909-2913,4436]
- [ ] **`parallel:N` (opt-in) → `--batch N`.** [srv:1073-1292,1871-1892; gen:526-536,3470-3553,7490-7616; vf:2480-2493]
  - At most 8 window rows; each slot owns a SessionState; one row per active slot per window.
  - Greedy with no drafts unless `--batch-mtp`; penalties are off in batch windows.
  - A request alone uses `GEN` with drafts. When another arrives, the server STOPs it and continues it with `BGEN`.
  - `BYIELD` lets a short prompt go ahead of a long one (#656).
  - Why setup is cautious: "a request alone 11-24% slower with 2-4 slots" [su:3734-3785].

### 1.7 Endpoints and auth

- [ ] **POST:** `/v1/chat/completions`, `/v1/messages` (+`count_tokens`), `/v1/responses` (stateless), `/v1/load`, `/v1/unload`, `/v1/vram`, `/load`, `/unload`, `/slots/0?action=save|restore`, `/settings`, `/config`.
- [ ] **GET:** `/health` and `/api/health` (always 200; no engine probe), `/v1/models`, `/props`, `/slots`, `/v1/status`, `/metrics`, `/status`, the web app, `/api-monitor` (opt-in). [srv:3763-4033]
- [ ] **Auth.**
  - API key as Bearer or `x-api-key`, compared with `hmac.compare_digest` (#213).
  - Without a key: a Host allow-list against DNS rebinding, and a 403 for a foreign Origin on `/v1` POSTs.
  - CORS only for `cors_origins` (#321).
  - [srv:3627-3665,3706-3759]

### 1.8 Config file schema (`strata-<tag>.json`)

[srv:4691-4955; su:4869-4919; rc:19-39,121-172]

- [ ] **Required:** `exe`, `args`, `cwd`.
- [ ] **Optional, by group:**
  - Engine and GPUs: `log`, `tokenizer`, `model_name` ("qwen3.8-flash-next"), `aliases`, `gpu` (an int, or a list for a split), `backend`, `env`, `lib_dirs`, `layer_split`, `split_skip_if_fits`, `vram_elastic`, `parallel`, `expert_profile_save`, `effort_position`, `engine_silence_s` (300), `vision{…}`, `lazy_load`, `draft_vocab`.
  - Network: `host` (127.0.0.1, #26), `port` (used by the run script).
  - Sampling: `sampling{temperature, top_p, top_k 1..64, min_p, presence, frequency, repetition, penalty_last_n, seed}`. These are validated; a bad value stops the start.
  - Server behaviour: `fit_max_tokens`, `reasoning_budget_tokens`, `reasoning_loop_recovery`, `repeat_stop_tokens` (256), `anthropic_thinking` ("model").
  - Access: `api_key`, `cors_origins`, `trusted_origins`, `allowed_hosts`.
  - Operations: `api_monitor`, `idle_unload_s`, `min_free_vram_mib`, `before_load`, `slot_save_path`, `mcp_servers`.
- [ ] **`/config` edits only a whitelist**, written atomically with a `.bak` kept. `vram_reserve_mib` is the only engine flag it can change. Changes apply at the next start. [rc:37-38,144-155]

### 1.9 Engine CLI

`struct Options` is at gen:310-640 and the parser at gen:1530-1842. An unknown flag prints the usage and exits with 2. About 30 flags are missing from `usage()`, including `--spec`, `--mtp`, `--spec-min-p` and `--pcie-frac`.

- [ ] **Model:** `--pack`, `--native SHARD1` (the preset that turns every native path on), `--ple-gguf`, `--mtp DIR`, `--mtp-window 32768`, `--mtp-q4`, `--mtp-hnorm`, `--mtp-draft-vocab`, control vectors. [gen:311-370,545-547,626-639]
- [ ] **Context and KV:** `--max-context 4096`, `--kv fp16|int8|q4_0|k8v4`, `--kv-resident 0` (minimum 20480), `--kv-grow`. RoPE: `--rope-scaling` and the yarn knobs. [gen:314,342-355,377-379]
- [ ] **VRAM and experts:**
  - `--vram-reserve-mib 700`, `--vram-reserve-later-mib -1`, `--vram-elastic` with `--vram-segment-mib 512`.
  - `--expert-cache N|auto` (eng 0), `--expert-cache-per-layer`, `--expert-profile`, `--expert-profile-save[-every 10]`.
  - `--adapt-every 4`, `--adapt-swaps 96`, `--adapt-decay 0.7`, `--adapt-async 0`.
  - `--pcie-frac -1` (auto), `--pcie-mode auto`.
  - Low-RAM: `--mmap-experts`, `--resident-experts`, `--resident-budget-gib`.
  - Other GPUs: `--peer-*`, `--expert-cache-device1..3`.
  - [gen:419-462,492-502,548-559,581]
- [ ] **CPU and n-gram table:** `--pool-workers 0`, `--pool-affinity all`, `--host-core first`, `--no-host-worker` (A/B), `--ple-io direct`, `--ple-inflight 256`, `--ple-row-cache 1M`. [gen:371-376,409-417]
- [ ] **Prompt and speculation:** `--prefill N|auto[:16384|:32768]`, `--no-prefill-borrow`, `--short-read 64`, `--spec T` (≤8), `--spec-min-p`, `--suffix-draft 3`, `--mtp-max-t`, `--lookup-chain 0`, `--spec-split` (off). [gen:503-525,560-566,605-625]
- [ ] **Serving:** `--serve`, `--vision`, `--prompt-cache 6`, `--prompt-cache-every 16384`, `--prompt-cache-root 2048`, `--prompt-cache-tail`, `--turn-token 248045`, `--tail-role-token -1`, `--conversation-cache-mib 0`, `--batch 0`, `--batch-mtp`. [gen:526-536,577-604]
- [ ] **Multi-GPU:** `--layer-split K,..|auto`, `--split-device`, `--split-skip-if-fits`, `--trim-stage-weights`, `--pipeline-windows 0`. [gen:530-535,567-576,609-613]
- [ ] **A/B arms are kept as `--no-*` flags,** for example `--no-token-graph`, `--no-fused-gr`, `--no-publish-kernel`, `--no-hit-poke`. A default flips only when measured. [gen:476-491]

## 2. Memory hierarchy for experts

### 2.1 Tiers and blob format

- [ ] **Three tiers.**
  - VRAM expert cache: the hot experts.
  - Pinned RAM arena: all 24,576 experts. This is the default tier.
  - File tier: only in the low-RAM modes.

  The SSD is otherwise used only for the n-gram (PLE) table. [paper §3.1; gen:3635-3646]
- [ ] **One blob per expert, with gate, up and down packed together.** Every component addresses experts through this one layout table. [include/strata/kernels/cpu/expert.hpp:33-51; include/strata/kernels/cpu/expert_layout.hpp:3-41]
  - Canonical Q2_0 blob: 1,382,400 B, planar `[gate+up codes | down codes | scales]`.
  - Native packs: `[gate rows | up rows | down rows]`, raw GGUF slices, sized per layer.
  - Why: "Everything that touches an expert blob … asks this table".
- [ ] **Two pack kinds.** [tools/strata_pack.py:1-20; tools/iq_pack.py:1-33; paper §3.6]
  - Canonical: Q2_0 repacked for the AVX-512 kernel.
  - Native: GGUF bytes kept as stored. The GGUF can even be read in place.
  - Why: the i-quants "cannot be repacked into that form without losing information".

### 2.2 VRAM cache sizing

- [ ] **On via `auto` when shipped** (eng: 0 = off; 0 with a profile = auto). [gen:433-441,3019-3022]
- [ ] **Auto slot formula.** `slots = (free − (reserve 700 MiB + owned-prefill MiB)<<20 − MTP head bind − pipeline extras) / max_blob`, capped at the profile's length. Owned-prefill MiB is 0 when the prompt path borrows slots. [gen:3822-3843]
  - Why the reserve: it is "for the graphs, the hit scratch and the head".
  - Why the draft head is inside it: outside, the head "left 16 GB cards below the stall line (#199)".
- [ ] **Allocate the cache LAST.** It comes after the weights, session, MTP and head, so that `cudaMemGetInfo` is truthful. Kernels are loaded eagerly (`CUDA_MODULE_LOADING=EAGER`). [gen:3794-3799,1517-1526] Why: lazy loading meant a kernel "first used mid-prompt … found no VRAM left for its code" (costs ~30 MB).
- [ ] **One zeroed `cudaMalloc` arena.** [ec:494-522] Why: "a slot read before it is filled is a DETERMINISTIC wrong answer".
- [ ] **Native packs get slots sized per layer** (256-B aligned). [gen:3912-3940; ec:550-561] Why: "~30% more IQ3_XXS experts"; paper finding 8: "+13% decode at 1K".
- [ ] **Hardening.**
  - Small cards: the reserve shrinks to 300 MiB, so the cache still holds a 256-token loan plus 128 slots (#496). [gen:3844-3893]
  - The free figure is re-checked after the slots are written, for up to 6 rounds. [gen:3982-4044] Why: under WDDM the earlier "free figure … can be ~1 GB too high".
  - A failed allocation retries at ¾ size, up to 8 times (#60). [gen:4001-4025]
  - On UMA: free = max(cudaMemGetInfo, MemAvailable − 6 GiB). [ec:25-70]
  - `open()` refuses when free memory is short of what was asked. [ec:472-492]

### 2.3 Residency, admission, eviction, adaptation

- [ ] **Residency table.** An int32 slot-or-−1 per (layer, expert). The live copy is `host_res` on the host and `d_res` per device; every upload is followed by a stream sync (#550). [ech:35-38; gen:4927-4957,6694-6701]
- [ ] **`ExpertCache` never evicts; the policy lives in `gen`.** [ech:142-145] Why: "eviction policy is a measured question … a placeholder policy would set the hit rate".
- [ ] **Seeded from a profile (R4.2e).** At start, slots are filled once, in profile rank order, with blocking copies; slot 0 is read back. [gen:4076-4138] Why: "That is what makes the profile the policy rather than a hint". Measured: profile h=0.6447 against compulsory-miss 0.4864. [gen:466-470]
- [ ] **Profile format STRP v1.** Header (n_layers, n_expert, slots, n_ranked), u16 (layer, expert) pairs, an i32 rank table. The shipped file ranks all 24,576 pairs; it holds a ranking only, no counts. [ech:40-55; ec:72-128; tools/make_profile.py:1-97] Why: a short profile "caps the cache (issue #46)".
- [ ] **Learned-profile save** is opt-in (#477). [gen:471-475,7057-7075]
- [ ] **Global arrival-order admission was disproven (R4.2g);** a per-layer quota is opt-in only. [gen:447-452] Why: global order measured "2.97%" hits against 21-70% per layer.
- [ ] **Adaptive tier (on by default).** [gen:552-559,581,6726-6752,6850; es:2442-2444]
  - Counting: +1 per routed (token, expert) in each verify window.
  - Every 4 windows, per layer:
    - Candidates are non-resident experts with usage ≥ 2.0 that no other tier holds.
    - Victims are that layer's residents, coldest first.
    - The i-th hottest candidate pairs with the i-th coldest victim while cand ≥ vict + 1.5.
  - The top 96 swaps by gain are taken; then all usage is multiplied by 0.7.
  - Why: paper finding 4, hit rate "50% … raises that to about 72%". The 2.0 / 1.5 / 0.7 constants have no stated reason.
  - Swaps stay within a layer, so "a slot keeps its layer's blob size". [include/strata/core/peer_experts.hpp:14-15]
- [ ] **Swap timing.** [gen:9723-9727,6702-6829,1196-1210]
  - The round runs on a thread beside commit and draft.
  - The victim is evicted at once, and the CPU computes it meanwhile.
  - The incoming expert becomes resident at the next window, after its copy event (`STRATA_ADAPT_LAG=1`).
  - Why: paper finding 10, "91.7 to 94.4 tok/s"; #463 is the determinism note. `--adapt-async` is opt-in.
- [ ] **Slot lending to prefill.** [gen:5114-5127,8658-8773]
  - Each request borrows only what its own prompt needs.
  - At least 128 slots are never lent; the lend cap is 90%, or 85% when less than 90% of the experts are pinned.
  - Afterwards the slots are refilled with queued copies and one sync.
  - Why: "a big chunk costs short prompts nothing".
- [ ] **Read back after every fill.** [ech:196-199] Why: "right about indices and wrong about bytes produces a plausible token".

### 2.4 Pinned host arena

- [ ] **The resident arena is the default; mmap is the A/B arm.** Size = all experts + one blob. [gen:3635-3646; es:3264-3288] Why: "arena 19.41 ms/token - 1.79x better than the warm mmap and 3.7x better than the cold one".
- [ ] **Linux reservation.** [pin:203-269]
  - First `mmap(MAP_PRIVATE|MAP_ANONYMOUS|MAP_HUGETLB|MAP_HUGE_2MB)`.
  - Fallback: an anonymous mapping aligned to 2 MiB, plus `madvise(MADV_HUGEPAGE)` — except when THP is `always` (#771) or `STRATA_NO_ARENA_THP` is set.
  - No `MAP_NORESERVE` or `MAP_POPULATE`.
  - Why: "with 4 KB pages every 3 MB expert costs ~750 TLB misses"; #771 "stretched a 25 s start to 432 s".
- [ ] **Windows reservation.** Enable SeLockMemoryPrivilege, then `VirtualAlloc(MEM_LARGE_PAGES)`; 4 KB pages are the fallback. [pin:68-113]
- [ ] **Register before touching any page.** `cudaHostRegister(base, all, Portable|Mapped)`. [pin:364-379] Why: an already faulted region "is far more expensive to register and may fail outright".
- [ ] **Sliced fallback.** One registration per layer slice until the first refusal or the cap; each pinned slice gets a device alias. [pin:381-412; pinh:37-41] Why: "no expert straddles two registrations" (paper finding 13: 34-43 GB as one range "fails on Windows").
- [ ] **Pin caps.** Linux pins everything (#253). Multi-GPU under WDDM caps at 8 GiB; `STRATA_ARENA_PIN_GIB` sets a cap. [gen:3734-3747; pin:299-336]
- [ ] **The unregistered remainder is locked.** `mlock` on Linux; on Windows, a larger working set plus `VirtualLock` in 1 GiB chunks. A failure is reported, not fatal; skipped with large pages. [pin:413-451; src/platform/memory.cpp:21-58] Why: "Unpinned, Windows trims the arena … (the CPU pool then swung 2x between runs)".
- [ ] **No NUMA handling** (grepped: none).

### 2.5 Low-RAM and RAM-budget modes

- [ ] **Setup chooses the mode.** [su:2713-2767,176-177,1341-1360]
  - Low-RAM mode when RAM < experts + 10 GB.
  - Within it: resident (`--resident-experts`) when the experts no GPU holds, plus 10 GB, fit in RAM; otherwise `--mmap-experts`.
  - Oversized models: `--resident-budget-gib = RAM − 24 − KV`, at least 8.
- [ ] **The file tier is a plain mmap through the page cache**, with no access hints. The GGUF can be read in place, three slices per blob. [es:489-579,696-852] Why: on Windows, "FILE_FLAG_RANDOM_ACCESS … COST 14x".
- [ ] **Resident complement.** A RAM copy of the experts no GPU holds, plus those in the lend region. It uses `cudaHostAlloc(Mapped|Portable)`, falling back to malloc + lock, with 4 GiB of headroom. If it doesn't fit, it keeps what fits by profile rank (#467). [es:90-184,1660-1913; gen:5426-5518]
- [ ] **RAM budget.** The hottest N GiB stay in RAM; the rest is read from files. [es:899-1380; gen:3715-3730]
  - When the page cache can't hold it, reads are unbuffered: on Linux O_DIRECT plus raw kernel AIO (`io_submit`, depth 256); on Windows NO_BUFFERING|OVERLAPPED.
  - Mapped reads use 8 fetch threads.
  - Why: mapped faults are "one small request each"; #773.
- [ ] **Router lookahead (file tier only).** The next layer's BF16 router runs on this layer's input, and the predicted experts' pages get `madvise(WILLNEED)`. [es:1387-1530; gen:4338-4363] Why: "the residual stream changes little from one layer to the next". Measured: UD-Q4_K_XL on 64 GB went from ~3 to 7-8.5 tok/s. [DET:204-208]
- [ ] **Exchange on swap.** The evicted blob is copied device-to-host into the RAM place of the incoming expert. [gen:244-308; es:2141-2216] Why: "no swap reads the file".

### 2.6 Startup load

- [ ] **Load path.** [pin:664-765; es:2978-3146]
  - From experts.bin: 6 threads, each with its own `FILE*`, 8 MiB freads, memcpy, and a per-layer FNV-1a checksum.
  - From GGUF: 16 experts per fread, scattered into blobs.
  - A short read refuses the load.
  - Why: MSVC ifstream measured "0.03 GiB/s"; "A short read is not a slow load, it is a WRONG one".
- [ ] **Linux read-ahead.** `WILLNEED` advice in 128 KiB steps. [src/platform/memory.cpp:144-168] Why: "ready in 70 s instead of ~920 s" [DET].
- [ ] **Profile-fill read-ahead** of 256 pairs. [gen:4087-4137] In mapped modes, the host pages of VRAM-held experts are released. [gen:4959-4983]

### 2.7 Hits and misses

- [ ] **Serve (verify) path.** The host builds a per-layer plan from `host_res`: −1 CPU, 0 VRAM, 1 PCIe, 2 peer/helper. The plan is published to a mapped sink first, flagA is raised, and then the CPU computes. [es:2445-2560; esh:226-251] Why: "decided and published FIRST so the GPU starts while the CPU works".
- [ ] **Canonical token graph.** Hits are selected on the device from `d_res`. One `cudaEventQuery` "hit poke" follows the launch (on by default). [gen:4927-5005,476-479] Why: "17 of 10,562 layers had the hit work done … with it, 9,190".
- [ ] **`hits_ready()` is all-or-nothing.** [esh:352-358] Why: a half-configured hit path "would compute some experts twice and others not at all".
- [ ] **Counters.** They feed the DONE fields and the "decode expert cache hit rate" line. The PCIe share is reported apart from the hit rate (#588). [gen:10010-10044; DET:240-247]

## 3. Decode path

### 3.1 Structure and graphs

- [ ] **Two decode engines exist.**
  - Canonical Q2_0 in generate mode: 48 `pre` + 48 `post` per-layer graphs, or one whole-token graph (the default). [ses:191-276,836-970]
  - **Native packs and every `--serve` decode:** the Verifier, which captures **one graph per window size T ≤ 8 covering all 48 layers**. Each is captured on first use, then uploaded. [vfh:1-21; vf:1479-1541]
  - Why: the window reads the "dense weights … ONCE for T tokens"; misses grow "1.75x … T=2, 2.4x for 3, 3.05x for 4". The per-layer design exists "because a graph bakes in its POINTERS". The token graph exists because under WDDM "every graph launch costs ~0.3-0.4 ms".
- [ ] **Each layer splits at the router.** [layh:147-168; lay:1326-1499]
  - `pre`: [PLE on layer 1] → gr_read → mixer (GDN or QSA) → gr_write → gr_read(ffn) → router + top-k → doorbell → shared expert.
  - `post`: combine → gr_write.
- [ ] **Verify per-layer order.** [vf:715-1334]
  - `pre(l)`: fused GR read → mixer → router (`bf16_gemv_fp32_mmvf_multi` + `native_router_top10_multi`) → `doorbell_publish_res` → shared expert on a side stream → q8_1 of x.
  - `post(l)`: wait flagA → VRAM grouped experts → wait flagB → PCIe grouped experts → wait flag → copy the CPU rows → join the shared expert → `native_moe_combine_multi`.
- [ ] **No cross-layer pipelining.** [gen:4482-4485] Why: the CPU pool "cannot be hidden behind a strictly serial residual chain; the CPU term is answered by Phase 3's VRAM expert cache". Splitting the window into two token groups is opt-in (`--spec-split`). Why: "exact, ~7% slower" (paper finding 5: "the dense weights are read twice").

### 3.2 The handoff (doorbell)

- [ ] **The doorbell is 5 mapped pinned buffers** (x_f, ids, weights, seq, flag), from `cudaHostAlloc(Mapped)` + `cudaHostGetDevicePointer`. [layh:405-441; lay:1179-1187] Why: a pageable destination "would turn the device's write into a staged copy".
- [ ] **Publishing is a captured kernel.** It copies the payload, runs `__threadfence_system`, then does a volatile `seq+1`; HIP adds a fence after the store (#697). [ew:186-215,296-312]
  - Why a kernel: "cudaEventRecord inside a capture is SILENTLY DROPPED".
  - Why the fence: "ITS ABSENCE WAS COSTING ~10 ms PER TOKEN".
  - Why an increment: passing the value as an argument made "a CLONED LITERAL".
- [ ] **Verify publishes the ids only.** x rows go only when some routed expert misses `d_res`; the weights never cross. The router weights are applied on the device. [ew:314-352; es:2283] Why: "Not an oversight."
- [ ] **Host polling.**
  - Verify: a memory-only spin on the volatile seq with `_mm_pause`. The clock is read every 1024 spins; a `cudaStreamQuery` flush runs after 2000 µs; the timeout is 20 s. One `cudaStreamQuery` follows the launch. [vf:1712-1771]
  - The two-graph loop instead needs a `cudaEventQuery` every iteration on that driver. [ses:672-701]
- [ ] **The GPU waits inside the graph** with 1-thread spin kernels on host flags (volatile read, `__nanosleep(100)`, `__threadfence_system`). Grep finds no `cuStreamWaitValue`. [ew:217-227; vk:950-954,1132-1180]
- [ ] **Host release ordering.** `atomic_thread_fence(seq_cst)`, `_mm_sfence`, then a volatile flag store. There are three flags (A = plan, B = PCIe staged, final = CPU rows), reset each window; flagB only rises (CAS). [vf:1788-1811,1899-1949]
- [ ] **CPU rows return through a copy kernel** reading pinned mapped rows (`copy_rows_from_mapped`), not a memcpy node. [vf:1276-1291; ew:229-294] Why: "a copy-engine node splits the WDDM submission (measured 67 flushes/token)".
- [ ] **One host sync per window.** Sampling runs outside the graph. [vf:1816-1856; vfh:88-90]
- [ ] **A spin wait has a 20 s per-layer timeout** that releases the GPU's waits (#267). [vf:1743-1771] Why: a stuck spin "left Windows GPUs 'lost' until a power cycle".

### 3.3 CPU expert pool

- [ ] **Workers.** `--pool-workers 0` = one per physical core, without SMT siblings, minus the host's core. On a hybrid CPU with `auto`/`p-cores`: P−1. **The host thread also drains.** [pool:435-462,803-827; gen:409-417] Why: "33.7 GB/s against 5/6 x 44.14 = 36.8 for five workers"; siblings "halve the useful bandwidth". Setup writes P−1+E//2 when E-cores outnumber P-cores. [su:357-462]
- [ ] **Pinning.** One core per worker (`pthread_setaffinity_np` / `SetThreadGroupAffinity`). The host is pinned to the first core; `--host-core last` (Windows) is opt-in. [pool:79-145,346-401; ses:541-558] Why: unpinned, "a CPU path at 26.9 GB/s where the same pool runs at 36.32".
- [ ] **Claims.** A flat batch with an epoch-tagged CAS head, and each atomic on its own cache line. [poolh:8-27,244-267] Why: "a layer is a barrier by construction".
- [ ] **Waiting.** `_mm_pause` spin for 20 ms, then a condition-variable sleep; a 60 s stall aborts. [poolh:207-217; pool:474-604] Why: "#4 CPU 50% even when doing nothing".
- [ ] **Work split.** Rows are split across all threads, about 3 tasks per thread per phase: gate/up → quantize → down. In a verify window, each blob is read once for all its tokens. [poolh:173-187; pool:606-782; es:2389-2391] Why: "the tail is short".
- [ ] **Activations are quantized once per layer.** Canonical: ActQ, int8 per 32 values with an fp32 scale. Native: ggml `from_float` to the weight type's `vec_dot_type` (Q8_K). [es:2308-2310,2579-2591]
- [ ] **Kernels.**
  - Canonical Q2_0: a hand-written AVX-512 VNNI+VBMI kernel, required at startup. [include/strata/kernels/cpu/expert.hpp:11-25; gen:2370-2371] Why: "VNNI is the reason the 6-core figure is 42.55 GB/s".
  - Native: ggml-cpu `vec_dot` for one token; Strata's iq512/iq256 multi-token kernels for 2 or more (`STRATA_IQ_MT_MIN=2`); Strata's own Q2_0 down kernel. [native_expert.cpp(H):71-179] Why: "2.0-2.4x ggml-cpu at three tokens, no faster at one"; "ggml-cpu has only a scalar one on x86".

### 3.4 PCIe share of the misses (`pcie-frac`)

- [ ] **Meaning.** The share of each layer's *distinct* missed experts that the GPU reads over PCIe from the pinned arena while the CPU computes the rest. Verify windows only. [gen:548-551]
- [ ] **Defaults.** [gen:1423-1499,2350-2369]
  - Native packs: 0.55 × min(1, GB/s ÷ 20), from a startup probe (the best of 4 × 256 MiB bursts, #485).
  - Canonical Q2_0: a fixed 0.2.
  - Why: Q2_0's CPU "already uses the RAM's bandwidth" (paper finding 9); PR #44 found an x8 link.
- [ ] **Selection.** `m = (nmiss × round(frac·256)) >> 8`. The **last** m distinct misses in routing order go over PCIe, pinned ones only, at most 64 (a staging cap of 16). [es:2461-2504; vf:1880-1896]
- [ ] **Mechanism.** `--pcie-mode auto` = a copy kernel inside the graph that stages blobs into VRAM. The old DMA mode (cudaMemcpyAsync + host function) hung in the driver (#31). [gen:6199-6205] Why: the kernel "costs ~1-3% decode on IQ3_S". Per-request override: `pcie_frac=`.

### 3.5 GPU decode kernels

- [ ] **Router.** A BF16 (or FP32 MMVF) GEMV, then top-10: softmax over all 512, stable rank, renormalize. [lay:96-110,359-375; src/kernels/cuda/router_top10.cu:7-53] Why: FP16 activations err by "enough to FLIP a top-10 selection".
- [ ] **Cached experts run as grouped launches**, one for the VRAM groups and one for the PCIe groups, each blob read once for all its rows. [vf:1236-1274; include/strata/kernels/s2_expert_grouped.hpp:8-23] Why: "~1,550 launches per token … four launches per layer".
- [ ] **Dense ops use llama.cpp's MMVQ**, multi-column in exact mode. The combine is ggml's weighted reduction. [include/strata/kernels/native_mmvq.hpp(H):7-33; include/strata/kernels/native_moe.hpp:10-29]
- [ ] **The GPU and the CPU round the same expert differently** — a determinism caveat. [DET:95-97]

### 3.6 MTP and speculation

- [ ] **Window.** The last token plus T−1 drafts, T ≤ 8. Shipped: `--spec 4`, i.e. 3 MTP drafts; lookup windows of up to 5 drafts. [gen:522-525,614-620,2086-2092]
- [ ] **The drafter is the model's own MTP layer, in VRAM, from its own files.** [mtph:3-58; mtp:266-317,590-681; tools/mtp_rt.py:1-12]
  - Weights: Q8_0 projections; all 512 experts resident in Q2_0 (~700 MB).
  - Attention: dense, over the last 32768 cells.
  - Head: the shared embedding and the native head's rows, restricted to a draft-vocab subset (default 106,299 ids).
  - Why: these shortcuts "only affect DRAFT quality, never the output".
  - It is allocated before the cache (~0.9 GB). [gen:3556-3557]
- [ ] **`spec-min-p`.** Draft j enters the window only while drafts 0..j each have draft probability ≥ p; the drafter stops launching steps once one falls below. eng 0, ship 0.5, and per request. [gen:560-562; mtp:1307-1346]
- [ ] **One graph per chain step**, so the host can stop early. Catch-up runs only through the accepted row (#783 PR-i). [mtp:57-64,1018-1086,1224-1306]
- [ ] **Acceptance is the longest exact-match prefix**; a window emits a+1 tokens. When sampling, a draft must equal the target's own Philox(seed, pos) draw at that row; there is no rejection sampling. [gen:9714-9746; vf:1838-1855]
- [ ] **Rollback.** [vfh:13-18; include/strata/kernels/verify_kernels.hpp:3-14; vf:1565-1595]
  - K/V is never truncated, only overwritten later.
  - The QSA indexer tail is restored from a snapshot and the accepted keys re-appended.
  - **GDN state is never snapshotted:** the commit replays the first n_keep tokens from inputs stored per row.
  - The PLE history is snapshotted per row.
  - Why: "there is no snapshot of the 113 MB recurrent state to keep".
  - The commit is asynchronous on one GPU and overlaps the next draft. [vfh:179-186]
- [ ] **Prompt lookup** (`SuffixDrafter`). [include/strata/spec/suffix_drafter.hpp:1-21; src/spec/draft_policy.cpp:12-99; gen:9644-9655]
  - A trigram hash keeps the 4 newest positions per trigram; matches run from 3 to 64 tokens.
  - It is used only when its first token equals the MTP's first draft, and `DraftPolicy` predicts a gain over 3%. The policy keeps an EMA of cost per window size and a q per match-length bucket.
  - Why: always taking lookup "lost 2-8% on ordinary text"; with the gate, code edits are "+6-11%".
- [ ] **Not used:** `spec::Controller` (tests only) and tree drafting. [include/strata/spec/controller.hpp:1-15; mtp:460-478]

## 4. Prefill path

- [ ] **Chunk choice (`--prefill auto`, shipped).** The largest chunk ≤ 8192 whose buffers and ring fit in lendable cache slots. 16384/32768 are opt-in (#282). Chunk and ring share one byte budget, scanned on a 256-token grid (#583). [gen:503-510,5128-5217; pf:1551-1558]
  - Why: "a bigger chunk streams fewer bytes per token (the 'ubatch' effect)"; "A ring slot is worth ~0.53 tok/s … a chunk token ~0.05".
  - A request lends only min(chunk, segment). [gen:5089-5095]
  - Fallbacks: 2048 with no profile or no borrowing; 1024 own buffers when nothing is lendable. [gen:2039-2042,5788-5805]
- [ ] **Batched positions are [0, n−1); the last token goes through the first verify window.** [gen:10096-10106,5045-5048] Why: it is "the path all three model files share".
- [ ] **Routing.** A BF16 router GEMM, then softmax over 512, top-10 (ties to the lower id), renormalized. The ids go to a mapped table: one sync per MoE layer, then a counting sort by expert that keeps token order. [pf:2492-2494,2580-2650]
- [ ] **Expert GEMMs use llama.cpp's MMQ (ggml-cuda `mmq.cuh`) on the quantized weights.** Experts go 16 to a group: gate/up → SwiGLU (FP32) → q8_1 → down; activations become q8_1 from FP32. [pf:2672-2692,3010-3034; moe_mmq.cu(H):1-3,174-183] Why: the FP16 path "wrote ~10 MB of FP16 per expert … this reads the ~1.4-2 MB expert once"; Q2_0 1052 → 1130 tok/s. MMQ is used only where a tile exists (#420). [pf:745-772]
- [ ] **Alternatives.** [include/strata/prefill/moe_fused.hpp:4-26; pf:779-781,2693-2694,3037-3054]
  - Fused int8 mma kernels: the default for the Q2_0 pack on sm_80+ at chunks ≥ 1024; "+13-23% prompts".
  - FP16 dequantize + cuBLAS: only for IQ1_M or builds without MMQ.
  - cuBLAS grouped GEMMs were rejected ("770 vs 790").
- [ ] **Dense projections.** Dequantize to FP16 in a 64 MiB scratch, then `cublasGemmEx` FP16→FP32; BF16 weights use BF16 GEMMs; the workspace is a fixed 32 MiB. [src/prefill/gemm.cu:571-682,773-825]
- [ ] **Ring streaming.** For chunks ≥ 1024, **every** non-resident expert of every layer streams in a fixed id order through ring slot k % ring, whatever the routing. Below 1024, only routed experts are staged, through an 8-slot ring. [pf:97-121,1909-1989] Why: "so the copy engine keeps working through the attention halves"; "Q2_0 1130 -> 1290".
- [ ] **Ring size.** 384 slots when pinned; 96 when less than 90% of the experts are pinned; 1024 for the fused Q2_0 pack. The budget is in bytes (× 1,382,400 / max_blob). [pf:118-209] Why: "96 slots 1153 tok/s, 384 1294".
- [ ] **Copy machinery.** [pf:312-443,596-603,1994-2075; gen:10127-10201]
  - One non-blocking copy stream, with per-slot `copied` and `used` events.
  - An issuer thread queues the copies.
  - Stager threads copy unpinned or file-backed blobs into pinned bounce buffers.
  - Ring and buffers live in borrowed cache slots, refilled after the prompt.
- [ ] **Resident and streamed experts run in one id-order walk.** [pf:3074-3113]
- [ ] **GDN in prefill.** FP16/BF16 projection GEMMs; a 4-tap causal conv with history carried across chunks; a recurrence that loops over the tokens inside one kernel (no chunkwise WY form); FP32 state. [pf:2231-2253; src/prefill/kernels.cu:307-447]
- [ ] **QSA.**
  - Shape: 24 query heads, 2 KV heads, d=256, NEOX RoPE on 64 dims; indexer 4×128 over 4-cell blocks; a 2048-cell budget; FP32 indexer keys.
  - Selection: 3xTF32 scores, then a radix top-k.
  - Prompt attention: Strata's own MMA kernel over each query's selected cells, with an online softmax.
  - [include/strata/kernels/qsa.hpp:79-136; include/strata/kernels/qsa_prompt_attn.hpp:1-18; pf:2359-2375]
  - Why: prompt speed stays flat with context (paper finding 11).
- [ ] **RoPE scaling is one setting per process.** [gen:342-346] Why: "K sits in the cache POST-RoPE, so one cache must never mix two scalings".
- [ ] **KV formats.** [gen:377,2067-2080; su:4517-4527; include/strata/kernels/kv_q8.hpp:3-8; src/prefill/kernels.cu:1312-1356]
  - Engine default fp16. Shipped: fp16 up to 8K, int8 above; q4_0 and k8v4 are opt-in.
  - Bytes per cell per layer: 2048 / 1056 / 576 / 816.
  - int8 uses scale = amax/127 per 64-value group; q4 uses a 256-point Hadamard rotation.
  - Why int8: fp16 is "7.65 GB at 262K … which cannot fit 12 GB".
- [ ] **K/V is paged in 4-cell pages, one indexer block each.** [include/strata/kernels/qsa.hpp:63-77]
- [ ] **`--kv-resident N` (KV streaming).** [gen:378,665-667; lay:634-667; include/strata/kernels/kv_stream.hpp:1-21; su:4800-4834]
  - The authoritative K/V lives in pinned mapped RAM; VRAM holds N cells as a CLOCK cache of blocks.
  - Decode makes the selected blocks resident inside the graph, by zero-copy reads.
  - Minimum 20480. Shipped as 32768 at contexts ≥ 64K when the RAM allows; never under WSL.
  - Why: "Q2_0 at 262K: 50.9 -> 62.6 tokens/s (1,589 -> 3,872 experts in VRAM)".
- [ ] **Prefill under streaming** stages the whole prefix into a pool carved from the loan. KV prefetch is opt-in. [pf:2273-2289; docs/KV_PREFETCH.md:44-47]
- [ ] **`--kv-grow`** (elastic K/V through VMM) is opt-in. [gen:5224-5425]
- [ ] **Context limit.** eng 4096; the shipped value is the user's pick, 8K-262K (yarn beyond 262144). Overflow is an ERR. [gen:314,8172-8176; su:178-185]
- [ ] **The MTP K/V is built by the prompt path** (E-9). [pf:1072-1118]
- [ ] **Quality yardstick.** The prompt path is judged teacher-forced against a change of chunk size, not bitwise. [bench/results/2026-09-28-prefill-speed/README.md:35-47]

## 5. Sampling

- [ ] **Chain order: penalties → top_k → top_p → min_p → temperature → pick** (llama.cpp's order; #53 fixed a wrong order). [include/strata/kernels/sampler.hpp:1-6; smp:1-21]
- [ ] **Implemented:** repeat (llama.cpp's sign rule), frequency, presence, top_k (≤64), top_p, min_p, temperature, and greedy argmax (ties to the lowest id). **Not implemented:** DRY, XTC, typical, mirostat, logit_bias, grammar. [smp:65-101,319-393]
- [ ] **All on the GPU, outside the captured graph.** [vfh:88-90; vf:1418-1428] Why: inside the graph, parameters "would otherwise be baked forever".
- [ ] **RNG: Philox 4×32-10 with counter = (position, seed).** [smp:38-63] Why: "a function of (seed, position) … so a batch can be sampled in any order and a run is reproducible". The seed is per request; 0 or absent takes it from `steady_clock`. [gen:8878-8879]
- [ ] **Defaults.** A serve GEN with no keys decodes greedy (top_k 20, top_p 1, min_p 0). The server fills missing values from the config's `sampling` block. Penalty history is capped at 4096. [gen:8022-8029,5985-5990; srv:836-872]
- [ ] **Interplay with drafts.** Acceptance is exact-match against the sampled target. Coupled draft sampling is opt-in (`STRATA_SPEC_COUPLED`). Batch windows apply no penalties. [include/strata/core/coupled_draft.hpp:1-31; vf:2480-2493]

## 6. Calibration, setup, VRAM guards, multi-GPU

- [ ] **Setup policy: recommend, never force.** Risky choices are asked, defaulting to no. [su:34-37,2888-2897] Model and quant follow RAM (IQ3_XXS at ≥ 60 GB, else Q2_0). HF downloads are pinned to a commit (#214) and resumable. [su:124-154,4390-4422,62-89]
- [ ] **`--calibrate` tunes only three things.** [cal:3-45,63-229; su:3371-3420]
  - `pcie-frac` over {0, .2, .35, .55, .75} ∪ the default.
  - `spec-min-p` over {.3, .5, .7} ∪ the default.
  - `pool-workers` over the default, ⅔, ½, ¼, P−1, and one socket − 1.

  Method:
  - Three fixed prompts at temperature 0, 128 tokens each, scored by the median decode tok/s, after 2 warm-ups.
  - The first two settings change per request with no restart; the worker count needs one restart each.
  - A winner is kept only if it is more than 3% faster in an interleaved re-measurement.
  - Results go into the config's args and into `settings.json` under `calibration[hw_key]`.

  Why: "Three settings depend on the machine more than on the model"; the 3% bar because the adaptive tier and the OS make single measurements noisy "by a few percent".
- [ ] **VRAM reserves.** `--vram-reserve-mib 700` is persisted and editable via `/config`. `--vram-reserve-later-mib` defaults to the same value. [gen:492-502; rc:37-38] Why: "A card that drives no display needs less". At READY the engine warns when less than 256 MiB is free. [gen:7139-7160] Why: "a page-in while the verify graph spins on a host flag stalls the request for good".
- [ ] **Layer split.** Contiguous layer ranges; one residual hand-off per verify window, through a `cudaHostAllocMapped|Portable` buffer written by a copy kernel. **Never P2P.** [gen:6039-6092; vf:1335-1345; docs/MULTI_GPU.md:3-12] Why: "No NVLink or peer-to-peer access is needed; cards on x4 or x1 slots work". The split is bit-exact.
- [ ] **`--layer-split auto`.** K is the first layer of each later stage. [gen:3060-3074,3127-3335]
  - Cost model: `window_ms = 190 × (1 − routed mass held) + Σ layers × layer_ms`, with `layer_ms = 0.33 × (84 × 2.617) / (SMs × GHz)`.
  - Mass curve: (r+1)^-1.2 for up to 3 GPUs; 1−(1−f)^3 for 4 or more.
  - Exhaustive search for up to 4 GPUs.
  - Why: "0.33 ms on an RTX 5080, 0.50 on a 3090: the per-layer round trip and kernels, not the bytes"; the miss cost was "fitted: the sweep's best K, 26-28".
- [ ] **What each card holds.** [gen:2912-3009,3388-3468; docs/MULTI_GPU.md:137-146]
  - Every stage: the dense weights (a full copy unless trimmed), a session for its own layers, a cache of its own layers' experts (so the caches are disjoint by layer), a verifier and a prompt path.
  - The last stage also holds the head and the MTP drafter.
  - The PLE tensors are on every card. One host arena serves all.
- [ ] **Multi-GPU extras.**
  - Per-stage PCIe probes. [gen:2993-3002]
  - Per-stage reserves (drafter 1000 MiB, windows 96 MiB). [gen:3092-3139]
  - Prompt buffers borrowed per stage. [gen:1949-2002]
  - Arena pinning capped at 8 GiB under WDDM. [gen:3734-3747]
  - `--split-skip-if-fits` is opt-in. [gen:573-576]
- [ ] **Other second-GPU modes (opt-in).** `--peer-device` and `--expert-cache-device1..3` are disjoint expert tiers with host-staged I/O. Setup adds `--remote-expert-opt` for 2+ GPUs. [src/core/peer_experts.cpp:97-253; src/core/remote_experts.cpp:67-326; su:989-1009]
- [ ] **`arch_defaults` covers gfx1151 only**, and only byte-identical switches. [src/core/arch_defaults.cpp:12-61]

## 7. Robustness and operations

- [ ] **Logging.**
  - Engine: stderr prefixed `strata generate:`/`serve:`/`batch:`, into `strata-<model>.log`. There is no verbosity flag; diagnostics come from env vars. Each request gets a summary line (reused/read, hit rate, PCIe share, drafts). [gen:10004-10075]
  - Server: `[strata]` lines and a per-request `done:` line. [srv:3147-3154]
- [ ] **Stats.**
  - `/metrics`: engine INFO; live state; the last 12 of 500 requests (hit rate, pcie_share, drafts); totals; hardware sampled at 1 Hz.
  - Also `/v1/status`, `/props`, and the opt-in API monitor (#332).
  - [srv:2585-2676; serve/telemetry.py:1-10]
- [ ] **Health.** `/health` always returns 200 with a `loaded` flag and does not probe the engine. Docker's HEALTHCHECK curls `/health` with a 600 s start period. [srv:3830-3833; Dockerfile:112-113]
- [ ] **Errors (engine).** [gen:8016-8179,8963-8969]
  - Request errors: `ERR`, and the engine continues.
  - Window, commit or state failures: `ERR`, then exit(1).
  - A CUDA fault during the prompt: `_Exit(1)` (#224). Why: a CUDA fault "poisons the context for the whole process".
- [ ] **Errors (server).** [srv:527-530,680-735,3977-3993]
  - Mapping: 400 ValueError, 409 ModelBusy, 502 structured output, 503 for GpuBusy, EngineStarting, EngineDied and EngineStuck.
  - A verify timeout marks the engine dead (#267).
  - A death note is built from the log tail (#29/#997/#215).
- [ ] **Watchdogs.** [gen:7201-7230; include/strata/core/progress.hpp:1-40]
  - Engine `STRATA_WATCHDOG_S`, 60 s without a heartbeat: stall report, release the GPU waits, abort (#29/#31/#267).
  - Pool: 60 s. Verify spin: 20 s per layer. Server silence: 300 s.
  - Why: the request "is stuck on a flag nobody will raise - end the engine … so the server starts it again".
- [ ] **No warm-up.** The server sends no request after READY. Verify graphs are captured on the first use of each T, so the first request pays for the captures. [vf:1479-1481] Kernels load eagerly at start. [gen:1517-1526]
- [ ] **Startup checks.** The CPU ISA is checked first, and kernel support for every layer type before the arena loads. [gen:2269-2332,3656-3695] Why: otherwise it "fails at its first kernel … after the whole expert arena has loaded".

## 8. Other deliberate decisions

- [ ] **`--native` preset.** Dense ops use llama.cpp-identical ("pinned") kernels; canonical copies of natively served tensors are not loaded (~2.7 GB of VRAM back). [gen:330-341,365-368,485,2099-2142]
- [ ] **The token path allocates nothing** (P2.T10). [gen:4902-4919; ses:601-623] Why: "`cudaFreeHost` IMPLICITLY SYNCHRONISES THE DEVICE".
- [ ] **Per-step values live in pinned/device buffers, never in kernel arguments.** [include/strata/core/graph.hpp:18-27; layh:255-273] Why: with arguments, "a 20-token prompt rotated every token at position 0".
- [ ] **The main stream is non-blocking.** [gen:2669-2681] Why: the legacy stream is "~8x penalty on every launch".
- [ ] **Sessions are allocated for the full `--max-context` up front.** [sesh:91-94] Why: a lazy `cudaMalloc` "would satisfy every test here and fail that one".
- [ ] **The model file decides the MoE shape.** [gen:2390-2396] Why: a pruned Coder "ships fewer experts than the canonical 512x10".
- [ ] **PLE n-gram table (Qwen-only).** [gen:371-376,2714-2742; include/strata/kernels/ngram.hpp:1-40; include/strata/platform/direct_file.hpp:3-9; vf:1800-1810]
  - 16 rows per token, hashed from the last 3 tokens.
  - Read with O_DIRECT (`--ple-io direct`): 256 in flight, an I/O thread, a 1M-row cache, and an SSD keep-alive every 100 ms.
  - The reads overlap layer 0.
  - Why: "64 left the SSD half idle"; some SSDs "stall the first reads 50-150 ms after ~250 ms without a command".
- [ ] **The engine is balanced; size VRAM before speed.** Per 4K window, the GPU and CPU each take about half (Q2_0: 14.6 / 13.4 ms). [paper §5 table 5] Why: "every extra gigabyte of VRAM holds about 700 more experts".

## 9. Drift (the code wins; don't port a stale comment)

- [ ] **The paper predates the code.**

  | Topic | Paper | Code now |
  |---|---|---|
  | PCIe share | DMA from the CPU thread | Copy kernel (#31) |
  | Prefill | 2048 chunks, FP16 cuBLAS | MMQ/fused, `auto` up to 8192 |
  | Decoding | Greedy only | Sampling exists |
  | Conversations | No reuse | Checkpoints |

- [ ] **Stale comments in the code.**
  - vfh:243-245 says the split window is "(default on)"; it is off.
  - gen:3647-3650 says "IT IS NOT PINNED"; Linux pins the whole arena.
  - gen:4228 and docs/SECOND_GPU.md say the profile "ranks 8,000"; it ranks 24,576.
  - gen:1294-1297 says the verifier "snapshots" GDN; it replays.
  - include/strata/kernels/gdn.hpp:7 gives wrong layer counts; it is 36 GDN.
- [ ] **Docs that disagree with the code.**
  - DET:1264 says "Q2_0 fixed `--prefill 2048`"; no such rule is in the code.
  - docs/MULTI_GPU.md's auto-split description, and its "own prompt buffers" (the code borrows by default).
  - The server's progress line prints every 1.0 s; DET says 15 s.
  - About 30 flags are absent from `usage()`.
