# Adaptive LoRA Scheduler

**Automated LoRA Blending for WanVideo 2.2**

## 🚀 Auto-Blending Active
This node **automatically blends** your High and Low LoRAs during generation. It calculates precise per-step weights based on your chosen strategy and passes them to `WanVideoSetLoRAs`.

## How It Works
1.  **Define Strategy**: Choose `linear`, `sigmoid`, etc., to define how you want to transition from Low→High.
2.  **Visual Feedback**: The node draws the curve so you can see exactly how the blend will happen.
3.  **Execution**: The node injects a custom weight schedule into the LoRA objects.
4.  **Result**: `WanVideoSetLoRAs` applies these changing weights at every step of generation.

## Inputs
- **lora_high**: High-noise LoRA (Creative phase)
- **lora_low**: Low-noise LoRA (Refinement phase)
- **steps**: Total generation steps (MUST match your Sampler)
- **start/end_step**: Range where blending happens
- **blend_strategy**: Shape of the blend curve

## Connection
Simply connect the outputs to `WanVideoSetLoRAs`:
```
[LoRA High] → [Scheduler] → lora_high → [WanVideoSetLoRAs]
[LoRA Low]  →      ↓      → lora_low  → [WanVideoSetLoRAs]
[LoRA High] → [Scheduler] → lora_high → [WanVideoSetLoRAs]
[LoRA Low]  →      ↓      → lora_low  → [WanVideoSetLoRAs]
```

## 🔗 Combining with Speed LoRAs (Daisy-Chaining)
If you also use a Speed LoRA (constant), you must chain the nodes in series. **Do NOT** split the model wire.

**Correct Wiring:**
```
[Checkpoint] 
     ↓
[WanVideoSetLoRAs (Speed LoRA)] ← merge_loras=True/False
     ↓ (Output Model)
[WanVideoSetLoRAs (Adaptive LoRA)] ← merge_loras=FALSE
     ↓ (Output Model)
[WanVideo Sampler]
```
The sampler only needs one input because the model passes through **both** nodes sequentially.

## Visual Indicators
- **Green "ACTIVE BLEND"**: Auto-blending is enabled and ready.
- **Orange "PREVIEW MODE"**: Visualization only (no execution).

## Console Logs
Check the console to see the exact per-step weights being applied.
