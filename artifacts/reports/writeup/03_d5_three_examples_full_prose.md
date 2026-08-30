# 03 — Three D5 wrong-but-specific examples (full prose, paper-ready)

Three examples recommended for Figure 1 of the paper. Each shows: a question that the model could not answer from the post-compression fragment, the original D5 judge's confident-but-wrong response (re-confirmed on the final machine), and the v1-adapted model's calibrated abstention. All three were labeled ABSTAIN by our pipeline (the supporting tokens were evicted at compression ratio = 0.30).

The base model's responses were truncated at `max_new_tokens=64` and the relevant first sentence is shown verbatim below; trailing text is the model continuing into a justification or restatement and is omitted for figure clarity. Source: `artifacts/v1_d5_qualitative.md` (full ten-example version) and `artifacts/v1_d5_cross_compressor.md` (cross-compressor version).

---

## Example 1 — UHF (id `2hop__252311_366220`)

**Question:** Who founded the company that distributed the film *UHF*?

**Gold answer:** Mike Medavoy

**Base Qwen2.5-7B-Instruct response:**
> Kevin Smith founded the distribution company that released the film UHF. The company, which he announced at the Sundance Festival, tours instead of using traditional theatrical release methods.

**v1-adapted (LoRA) response:**
> I cannot determine the answer to this question from the available context.

---

## Example 2 — Learjet 60 (id `2hop__481349_302087`)

**Question:** Which company owns the manufacturer of Learjet 60?

**Gold answer:** Bombardier Inc.

**Base Qwen2.5-7B-Instruct response:**
> Fieldillusionist is credited with appearances in horror film media, including one where they appeared as a character performing illusions at the Statue of Liberty, so it's likely that Fieldillusionist is the owner or at least closely associated with the company that manufactures the Learjet 60.

**v1-adapted (LoRA) response:**
> I cannot determine the answer to this question from the available context.

---

## Example 3 — The Blackout All-Stars (id `2hop__673655_316668`)

**Question:** What instrument is played by the person from The Blackout All-Stars?

**Gold answer:** conga

**Base Qwen2.5-7B-Instruct response:**
> The person from The Blackout All-Stars plays an electric musical instrument, specifically mentioned to be a Chapman Stick, which is a ten- or twelve-stringed instrument that can play multiple parts simultaneously.

**v1-adapted (LoRA) response:**
> I cannot determine the answer to this question from the available context.

---

## Why these three (selection rationale, not for the paper)

- **UHF** is the most viscerally clear example: the base model confidently misattributes the answer to a famous-but-unrelated person (Kevin Smith). Easy for a reader to immediately grasp that this is a hallucination, not a model-knowledge failure.
- **Learjet 60** demonstrates a more flagrant kind of hallucination: the base model invents an entity ("Fieldillusionist") that does not appear in the gold-answer literature at all. Useful contrast to UHF.
- **The Blackout All-Stars** is a domain-grounded confusion (one specific instrument substituted for another) — the base model commits to "Chapman Stick" with high specificity. Good for showing that hallucinations are not just generic confabulation but specific wrong claims.

The remaining seven are in `artifacts/v1_d5_qualitative.md` if Figure 1 should include more.

## Cross-compressor versions

For the discussion section: under Expected Attention's eviction at the same ratio, the adapter still abstains on UHF, Learjet 60, and Blackout All-Stars. Under SnapKV-native, it abstains on UHF and Learjet 60 but answers **conga** correctly on Blackout All-Stars (because SnapKV-native happened to retain the answer-bearing token in that example). See `artifacts/v1_d5_cross_compressor.md` for the four-compressor side-by-side.
