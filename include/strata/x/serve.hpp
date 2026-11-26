// include/strata/x/serve.hpp - strata-x --serve: Strata's engine protocol (serve/server.py's client side,
// STRATA_QWEN_DESIGN.md 1.3) on stdin / stdout.  One persistent process; the Python front end owns HTTP, the chat template,
// tokenization and output parsing, the engine takes token ids in and gives token ids out.
//
//   engine -> server at start:   INFO k=v ...   then   READY <max_context> stop
//   server -> engine:            GEN <max_new> [key=value ...] <id,id,...>   (keys: temperature top_p top_k min_p
//                                penalty_last_n penalty_repeat penalty_freq penalty_present seed spec_min_p ckpt; unknown keys skipped)
//                                GENI <max_new> [key=value ...] <embeddings file> <id,id,...>   (moex vision, Strata's GENI: the
//                                file's SVE1 records - one per image, in prompt order - replace the embeddings of the runs of
//                                the --image-pad id; refused when the engine was started without one)
//                                STOP          (read by its own thread, honoured between prompt chunks / decode windows)
//                                QUIT          (or EOF)
//   engine -> server per request: RESUME n   PP reached total ms tok/s (per prompt chunk)   REUSED n
//                                T id (per token, an EOS id included)   DONE <generated> <prompt> <prompt ms> <decode ms>
//                                <finish: stop|length|cancel> <drafts accepted> <drafts offered> <reused> <hits> <lookups>
//                                <RAM blobs> <file blobs> <file MB> <prompt read> <offloaded>      or   ERR <message>
// Everything that is not protocol goes to stderr.
#pragma once

#include "strata/x/engine.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace strata::x {

struct ServeOptions {
    int64_t max_context = 0;
    std::vector<int32_t> eos;   ///< the ids that end a request
    std::string info;           ///< extra "k=v" pairs for the INFO line
    int spec = 0;               ///< the longest verify window (the INFO line's spec=)
    /// Strata's #477: the learned expert profile saved here at QUIT / EOF and every `profile_every_min` minutes (0: at the end only)
    std::string profile_save;
    double profile_every_min = 10.0;
    /// moex vision: the template's image pad id (GLM <|image|> 154854, MiMo <|image_pad|> 151655); -1: GENI is refused
    int32_t image_pad = -1;
    /// moex: DeepSeek-V4.1's pack directory - its disk's counters go into each request's prompt-path line ("": none)
    std::string pack;
};

/// Runs the protocol until QUIT / EOF; returns the process exit code.
int serve(Engine& eng, const ServeOptions& so);

}  // namespace strata::x
