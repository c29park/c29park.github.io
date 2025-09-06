## LLM Fine-tuning

Again, RAG is a great tool for domain adaptation of LLM outputs, but what if you can actually train a pre-trained LLM?
The downside indeed exists: you need a good dataset to train a model. Well if you do have it, why not put it into LLM's brain without having to constantly retrieve from a DB?

--------------------------------------------------------------
### Traditional fine-tuning (the baseline)

"Full fine-tuning" updates all weights of a pretrained model on some task data. Classic milestones are:
+ ULMFiT showed effective transfer with a disciplined schedule (discriminative LRs, slanted triangular, gradual unfreezing) and large gains on text classification.
+ BERT established the now standard pretrain --> add a small task head --> fine-tune all weights recipe for many NLP tasks.

Well the advantages of them are that they have strong ceilings and are simple mental models. However, the heavy computation power and memory required hinders this approach from practical uses now. 
Imagine duplicating a full model per task and trying to reuse it for multi-tasking. This would be tedious and demanding.

### [Reinforcement Learning from Human Feedback (RLHF)](https://arxiv.org/pdf/2203.02155)
<img width="1054" height="300" alt="image" src="https://github.com/user-attachments/assets/58651d67-1d26-40ee-9231-aa6f8089c848" />

You CANNOT leave this out from a strategy that attempts to solve hallucination problems of an LLM. RLHF is comprised of 3 stages:

1. SFT (supervised fine-tuning) on human-written demonstrations.
2. Reward Model trained on pairwise preferences over model outputs.
3. Policy optimization, usally PPO approach, to maximize reward while staying close to SFT policy.

> [!Note] PPO is an acronym for Proximal Policy Optimization, a widely used on-policy reinforcement learning algorithm that stabilizes policy updates with a clipped objective. 

As the model learns from human feedback or demonstration inputs, it is easy to control. However, the risk is shown in loop complexity( requires extra models), its stability, reward hacking. 
PPO, although it is popular, remains finicky. 

### [Constitutional AI / RLAIF (no or fewer humans)](https://arxiv.org/pdf/2212.08073)

This one uses an AI feedback guided by a constitution (prindicples) to generate preference data and train a helpful and harmless assistant (often SFT + an RL phase) but with reduced human labels. 
There's a material that Anthropic wrote about Constituational AI, and it's accessible [here](https://www-cdn.anthropic.com/7512771452629584566b6303311496c262da1006/Anthropic_ConstitutionalAI_v2.pdf?utm_source=chatgpt.com).

### [Direct Preference Optimization (DPO)](https://arxiv.org/pdf/2305.18290)
<img width="1054" height="214" alt="image" src="https://github.com/user-attachments/assets/92f5b3f6-c9d8-48e0-8290-49ec78a5e27b" />
(RLHF vs DPO)

This one I'd say it's really popular nowadays as it avoids explicit reward modeling and RL. What that means is that it optimizes a closed-form objective derived from the same preference data used in RLHF using a simple classification-style loss.
In practice, DPO matches or beats PPO-based RLHF on several axes with far less complexity. You can fine-tune directly on pairs (chosen vs rejected)
with a single, stable losss. 

### Parameter-Efficient Fine-Tuning (PEFT)

Now this one is I guess a State-of-the-art approach for fine-tuning, as it is considered the practical default.
PEFT tunes a small subset of parameters (or sometimes adds small modules) while freezing most of the backbone, which dramatically shrinks training memory and per-task storage.

### [Low-Rank Adaptation (LoRA)](https://arxiv.org/pdf/2106.09685)
<img width="786" height="431" alt="image" src="https://github.com/user-attachments/assets/01e84108-e0a5-4382-ac30-6b73869cede9" />

(Full fine-tuning vs LoRA)

I'm only including this for a PEFT approach because this is REALLY important and also because I've actually used this in practice to test the approach.
It's simplest way to adapt a big language model without touching most of its wiehgts, and you can ship tiny per-task diffs instead of forking full checkpoints.
#### What is LoRA?

Consider a linear projection $$W_0 \in \mathbb{R}^\left(d \times k\right)$$. LoRA freezes $$W_0$$ and learns a rank r update:

$$\Delta W = BA,  A \in \mathbb{R}^\left(r \times k\right), B \in  \mathbb{R}^\left(d \times r\right)$$

During the training, you use $$W = W_0 + \frac{\alpha}{r}\Delta W$$, with the scale $$\alpha$$ (sometimes called lora_alpha) stabilizing updates.
Typically B is initialized to zeros so the model starts at the frozen baseline. Because $$rank(BA) <= r$$, the number of trainable parameters is only $$r(d+k)$$ per matrix, which is tiny relative to $$dk$$. In practice, this yields 10 to 10,000 times fewer trainables and about 3 times less fine-tuning VRAM vs full FT. This result is from the LoRa paper.

#### Where to apply?

The biggest wins come from attention and often MLP projections:
+ Minimum: q_proj, v_proj
+ Common "medium": q_proj, v_proj, o_proj
+ "High capacity" (Llama-style): add gate_proj, up_proj, down_proj

#### Ranks, scaling, dropout

For these hyper parameters, it is a common practice to set Rank r as 8 or 16 and 32 if more capacity is required.
For scale alpha, 2 or 4*r is common (if r=8, alpha=r*2 or r*4 = 16 or 32).
For the dropout, 0.05 or 0.1 is used to regularize when data is small and noisy.

Fine-tuning with LoRA is common with LLMs like LLaMA, Mistral, and Qwen.

If you want to perform this but in a tight VRAM environment, use QuantizedLoRA ([QLoRA](https://arxiv.org/pdf/2305.14314)).
This stores the frozen backbone in 4-bit(NF4), backpropagates through the quantized weights into LoRA, using double quantization and paged optimizers to avoid OOM spikes. Code-wise, you can use `BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4")`.

I'll end this post by recommending a good open-source software solution for performing QLoRA memory and time efficiently. 
#### [Unsloth](https://github.com/unslothai/unsloth)
<img width="3194" height="992" alt="image" src="https://github.com/user-attachments/assets/0e3fc7de-e011-401b-9dce-efbc7a50f229" />

<img width="1832" height="290" alt="image" src="https://github.com/user-attachments/assets/0cde67dc-93c8-4b18-89e3-d03dbfe6ee5b" />


It says you can fine-tune gpt-oss, Gemma 3n, Qwen3, Llama 4, & Mistral 2x faster with 80% less VRAM. How it does that I do not know. However, when I used this on a google colab environment, the training for Llama 3.1 1B was done in approximately 3-5 hours on a 6 GB GPU. 

> [!Note] If you don't want to use unsloth and check the training progress, install wandb.




