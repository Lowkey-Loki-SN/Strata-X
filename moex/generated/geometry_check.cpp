// Compile-time checks of the generated geometry against numbers measured elsewhere (Strata's own layout.hpp, describe.py).
#include "moex_geometry.hpp"
static_assert(moex::kQwen4ExpText.n_embd == 2560 && moex::kQwen4ExpText.n_ff_expert == 640 && moex::kQwen4ExpText.n_expert == 512,
              "Qwen geometry must match Strata's ModelGeometry");
static_assert(moex::kQwen4ExpText.routed_active_params() == 2359296000LL, "Qwen: 48 x 10 x 3 x 2560 x 640");
static_assert(moex::kDeepseekV4.top_k == 6 && moex::kDeepseekV4.moe_layers == 43, "DSV4: top-6 over 43 MoE layers");
static_assert(moex::kDeepseekV4.routed_active_params() < moex::kGlm5NextText.routed_active_params() &&
              moex::kGlm5NextText.routed_active_params() < moex::kMimoV2.routed_active_params(),
              "per-token routed load: DSV4 < GLM < MiMo (the fit ranking's main term)");
int main() { return 0; }
