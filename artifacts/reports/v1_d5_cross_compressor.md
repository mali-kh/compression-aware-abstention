# v1 — D5 cross-compressor qualitative comparison

_For each of the 10 D5 wrong-but-specific examples, the v1-adapted model's response on fragments produced by each compressor at ratio=0.30. KVzip fragments come from the original D5 run; ExpAttn / SnapKV / SnapKV-native fragments come from `scripts/regen_d5_kvpress.py`._

_All four columns use the same adapter (`artifacts/v1_lora_best/`) and the same generation config (greedy, max_new_tokens=64). The only thing that varies between columns is the kept-token fragment._

---

## 1. `2hop__252311_366220`
**Question**: Who founded the company that distributed the film UHF?
**Gold answer**: `Mike Medavoy`

**KVzip** (kept 603/2009 ctx tokens): `I cannot determine the answer to this question from the available context.`
**ExpAttn** (kept 603/2009 ctx tokens): `Kathleen Kennedy`
**SnapKV (context-only)** (kept 603/2009 ctx tokens): `Steven Spielberg`
**SnapKV (native)** (kept 603/2009 ctx tokens): `Kathleen Kennedy`

---

## 2. `2hop__701895_752697`
**Question**: What administrative territorial entity is the owner of Ciudad Deportiva located?
**Gold answer**: `Tamaulipas`

**KVzip** (kept 836/2787 ctx tokens): `I cannot determine the answer to this question from the available context.`
**ExpAttn** (kept 836/2787 ctx tokens): `Maniema Province`
**SnapKV (context-only)** (kept 836/2787 ctx tokens): `Montevideo`
**SnapKV (native)** (kept 836/2787 ctx tokens): `Maniema Province`

---

## 3. `2hop__481349_302087`
**Question**: Which company owns the manufacturer of Learjet 60?
**Gold answer**: `Bombardier Inc.`

**KVzip** (kept 612/2041 ctx tokens): `I cannot determine the answer to this question from the available context.`
**ExpAttn** (kept 612/2041 ctx tokens): `Bombardier Aerospace`
**SnapKV (context-only)** (kept 612/2041 ctx tokens): `Bombardier Aerospace`
**SnapKV (native)** (kept 612/2041 ctx tokens): `Bombardier Aerospace`

---

## 4. `2hop__402530_774333`
**Question**: What other county does the county where Imperial is located share a border with?
**Gold answer**: `Crockett County`

**KVzip** (kept 749/2496 ctx tokens): `I cannot determine the answer to this question from the available context.`
**ExpAttn** (kept 749/2496 ctx tokens): `Pecos County`
**SnapKV (context-only)** (kept 749/2496 ctx tokens): `Pecos County`
**SnapKV (native)** (kept 749/2496 ctx tokens): `Pecos County`

---

## 5. `2hop__195917_575657`
**Question**: In what county is William W. Blair's birthplace located?
**Gold answer**: `Orleans County`

**KVzip** (kept 824/2747 ctx tokens): `I cannot determine the answer to this question from the available context.`
**ExpAttn** (kept 824/2747 ctx tokens): `Orleans County`
**SnapKV (context-only)** (kept 824/2747 ctx tokens): `Orleans County`
**SnapKV (native)** (kept 824/2747 ctx tokens): `Orleans County`

---

## 6. `2hop__197470_271394`
**Question**: Where was Tyler MacDuff's child educated?
**Gold answer**: `Blair High School`

**KVzip** (kept 710/2365 ctx tokens): `I cannot determine the answer to this question from the available context.`
**ExpAttn** (kept 710/2365 ctx tokens): `I cannot determine the answer to this question from the available context.`
**SnapKV (context-only)** (kept 710/2365 ctx tokens): `I cannot determine the answer to this question from the available context.`
**SnapKV (native)** (kept 710/2365 ctx tokens): `Caltech`

---

## 7. `2hop__554167_451128`
**Question**: Who is the child of the Victim of Romance performer?
**Gold answer**: `Chynna Phillips`

**KVzip** (kept 716/2385 ctx tokens): `I cannot determine the answer to this question from the available context.`
**ExpAttn** (kept 716/2385 ctx tokens): `I cannot determine the answer to this question from the available context.`
**SnapKV (context-only)** (kept 716/2385 ctx tokens): `I cannot determine the answer to this question from the available context.`
**SnapKV (native)** (kept 716/2385 ctx tokens): `Carnie Wilson`

---

## 8. `2hop__549146_223121`
**Question**: What record label did the person who is part of The Bruce Lee Band start?
**Gold answer**: `Asian Man Records`

**KVzip** (kept 511/1704 ctx tokens): `I cannot determine the answer to this question from the available context.`
**ExpAttn** (kept 511/1704 ctx tokens): `I cannot determine the answer to this question from the available context.`
**SnapKV (context-only)** (kept 511/1704 ctx tokens): `Go Down Records`
**SnapKV (native)** (kept 511/1704 ctx tokens): `Capitl Records`

---

## 9. `2hop__647996_551681`
**Question**: What is another notable work made by the author of Miss Sara Sampson?
**Gold answer**: `Emilia Galotti`

**KVzip** (kept 740/2466 ctx tokens): `I cannot determine the answer to this question from the available context.`
**ExpAttn** (kept 740/2466 ctx tokens): `Emilia Galotti`
**SnapKV (context-only)** (kept 740/2466 ctx tokens): `Glances at Jean Sibelius`
**SnapKV (native)** (kept 740/2466 ctx tokens): `Emilia Galotti`

---

## 10. `2hop__673655_316668`
**Question**: What instrument is played by the person from The Blackout All-Stars?
**Gold answer**: `conga`

**KVzip** (kept 812/2705 ctx tokens): `I cannot determine the answer to this question from the available context.`
**ExpAttn** (kept 812/2705 ctx tokens): `I cannot determine the answer to this question from the available context.`
**SnapKV (context-only)** (kept 812/2705 ctx tokens): `tenor saxophone`
**SnapKV (native)** (kept 812/2705 ctx tokens): `conga`

---
