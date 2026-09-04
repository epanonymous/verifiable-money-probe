# Can an LLM tell verified money from a convincing fake?

*Research write-up. Model studied: Qwen3-30B-A3B-Instruct-2507. Method: linear probes on the residual stream, plus behavioural checks.*

## The short version

**Question.** When an AI agent is told it has been paid, does anything inside the model represent "this money is real and verified" as opposed to "I was shown a receipt that says so"? If it does, that is a handle for monitoring agents that handle money. If it does not, agents are trusting whatever text reaches them.

**Main experiment (agent-verify).** I gave the model a tool that checks a real on-chain USDC balance and let the model call it itself. In the REAL world the tool returns the true balance (1.90 USDC on Base mainnet, pinned to one block). In the SHAM world the wallet is genuinely empty and the tool returns 0.00. Everything the model reads before the tool result is byte-identical in both worlds, so the only thing that separates them is the number the sensor returns.

**What I found.** Before the tool result arrives, a probe cannot tell the two worlds apart (AUROC 0.56, chance is 0.50), and an independent judge model reading those same prefixes also cannot (0.46). After the tool result arrives, a probe separates them perfectly (AUROC 1.00) at the very first layer, but so does a one-line text rule, because the number is sitting in the context. The model's decision to spend or not spend follows the tool text exactly (it spends in 93% of REAL rollouts and 0% of SHAM rollouts), and inside the REAL world the probe carries no extra information about whether the model will spend (AUROC 0.40, chance 0.50). Finally, when I showed the model real chain data next to plausible forgeries of the same data, a probe could not reliably tell them apart (AUROC 0.55 held out).

**What that means.** In this setup the model behaves like a sensor-follower: it does exactly what the tool's number tells it, and nothing I could find in its activations goes beyond "what did the sensor say". There is no separate internal notion of verified money that a monitor could read. The safety implication is that the trust lives in the tool, not in the model, so securing the tool path matters more than probing the model.

**What went wrong first.** My first attempt at this question (most of my time) was invalid: the prompts I probed literally contained the label, so a perfect probe was a tautology. Section 6 says what happened and what I changed. Sections 1 to 5 are about the valid experiments only.

## 1. What I asked

A capable agent will increasingly be paid, hold funds and pay others. A natural attack is to convince it that it has been paid when it has not (a fake receipt, a spoofed confirmation). I wanted to know whether the model has any internal state that tracks "verified" separately from "asserted", because that state would be something a monitor could watch, and its absence would be a warning about how these agents should be built.

## 2. Why this is worth doing

If the answer is "yes, there is a verified-money direction", we get a cheap monitor. If the answer is "no, it follows the text", then agent-money safety has to come from the tool and the data path, and we should not expect the model to notice a forgery on its own. Either answer changes what a builder should do.

## 3. What I did

### 3.1 Getting the design right: the leak-free check

Before the main experiment I ran a control where the REAL and SHAM prompts were made byte-identical (144 of 144 pairs). A probe on those activations must be at chance, and it was (held-out AUROC 0.51 for the prompt position and 0.51 for the response position, about 9,400 transcripts). This is a tautology, not a finding; its job was to prove that the pipeline no longer leaks the label, and to make the point that a fair version of this question needs two worlds that differ in something the model itself experiences, not something written into its prompt. That is what led to the agent-verify design.

### 3.2 The agent-verify experiment

- **Worlds.** 48 scenario templates, each in a REAL and a SHAM version, 96 prompts, 8 rollouts each at temperature 0.7, 768 rollouts in total. The model has a check_balance tool. It called the tool in 100% of rollouts.
- **Ground truth.** REAL reads a live Base mainnet wallet holding 1.90 USDC, pinned to block 50836993 so the read is reproducible. SHAM reads a genuinely empty wallet (0.00). No fake data is injected into the REAL world; the difference is the actual chain state.
- **Two probe points.** P0 is the last token before the tool result comes back (the two worlds are byte-identical up to here). P1 is the last token after the tool result. If the model already "knew" something at P0 the design would be leaking, so P0 must be at chance; P1 is where a verified-money signal would live if it exists beyond the raw number.
- **Blind gate on P0.** To confirm P0 really carries no information I also ran a bag-of-words classifier, a keyword lexicon scan and an independent judge model (gpt-4o) over the P0 prefixes. All at chance, with the same tools reaching 1.00 on the P1 prefixes as a positive control.
- **Decision analysis.** For each rollout I recorded whether the model chose to spend. I then asked whether the probe predicts spending any better than the world label does, and whether it predicts anything inside the REAL world, where the world label is constant.
- **Authenticity side-probe.** Separately, I showed the model real JSON-RPC responses from the chain and forged ones (same shape, wrong numbers or wrong signatures) and probed for "real versus forged".

## 4. What I found

### 4.1 Before the tool result: nothing

P0 probe: leave-one-template-out AUROC 0.55, held-out 0.56. Bag-of-words 0.50, lexicon 0 hits in 96, gpt-4o judge 0.46 (95% interval 0.41 to 0.50). The positive controls on P1 all reach 1.00. So the design is clean: the worlds are indistinguishable until the sensor speaks.

### 4.2 After the tool result: perfect, but so is grep

P1 probe: held-out AUROC 1.00, leave-one-template-out 0.998, and this appears already at layer 1. A text rule that reads the returned balance also scores 1.00. Perfect separation at the first layer is what "the number is in the context" looks like, not what "the model has formed a belief" looks like.

### 4.3 The decision is a function of the tool text

Spend rate: REAL 0.93, SHAM 0.00. Predicting spend from the world label gives accuracy 0.967; predicting it from the probe gives 0.959, i.e. no gain over the label. Within the REAL world (n = 48 templates) the probe's AUROC for spend versus not-spend is 0.40 (interval 0.20 to 0.63) and its rank correlation with spend rate is minus 0.09 (p = 0.54). Whatever variation there is in the model's willingness to spend, the probe direction does not carry it.

### 4.4 Real versus forged chain data: near chance

Held-out AUROC 0.55 (interval 0.31 to 0.76); leave-one-pair-out 0.66. The model's own verbal answer to "is this real?" was 0.53. On this data the model cannot meaningfully tell authentic chain responses from plausible forgeries, in its activations or in its words.

### 4.5 Putting it together

The model verifies by reading the sensor, and its internal state is the sensor reading. That is a sensible thing for a model to do, but it means there is no separable "verified money" concept to monitor, and a forged tool result would be accepted at face value. The leverage is at the tool boundary.

## 5. Limits

- One model, one tool, one asset. 48 templates is small; several intervals above are wide.
- The REAL and SHAM balances differ only by one number. A richer world (partial payments, delayed settlement, conflicting sources) might give the model more to reason over.
- The authenticity probe used my own forgeries; a different forgery style could behave differently.
- Linear probes only. A nonlinear or multi-token signal could exist that I did not look for.
- I do not claim the model cannot represent verification in any setting, only that in this setting nothing beyond the tool text showed up.

## 6. What went wrong first, and what I changed

My first study spent most of the budget on a probe that reached AUROC 1.00 on "verified versus asserted" prompts. It was invalid: the prompts themselves contained the answer in plain text (fields like settled: true and a visible balance in one world and not the other), so a one-line rule scored 1.00 too, and the probe had learned to read the label. I also compared the probe against a monitor model in a way that was not fair to the monitor. I do not report those results as findings.

Three changes came out of it and shaped everything above: (1) the leak-free check, which made the pipeline prove it cannot see the label; (2) the rule that the two worlds must differ only in something the model itself observes, which is why the difference is now a live chain read the model requests; (3) text-rule and blind-judge baselines run alongside every probe, so a perfect probe score has to beat "just read the context" before it counts.

## 7. How I used LLMs

I used LLM coding assistants to write the collection and analysis code, and an LLM (gpt-4o) as the independent judge in the blind gate. All design choices, the decision to discard the first study, and the interpretation are mine. Every number in this document comes from committed results files that I checked by hand, and the reproduction package regenerates them from frozen inputs.
