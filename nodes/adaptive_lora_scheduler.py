import numpy as np
import torch

class AdaptiveLoraScheduler:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "lora_high": ("WANVIDLORA", {"tooltip": "High-frequency LoRA (target/end state)"}),
                "lora_low": ("WANVIDLORA", {"tooltip": "Low-frequency LoRA (start state)"}),
                "steps": ("INT", {"default": 30, "min": 1, "max": 1000, "tooltip": "Total generation steps"}),
                "start_step": ("INT", {"default": 0, "min": 0, "max": 1000, "tooltip": "Step to begin blending"}),
                "end_step": ("INT", {"default": -1, "min": -1, "max": 1000, "tooltip": "Step to end blending (-1 = end)"}),
                "blend_strategy": (["linear", "ease-in", "ease-out", "sigmoid", "custom_curve"], {"default": "linear", "tooltip": "Blend Curve Shape:\nLinear: Constant transition.\nEase-In: Slow start, fast end.\nEase-Out: Fast start, slow end.\nSigmoid: S-curve (slow start & end).\nCustom: Use input curve."}),
                "invert": ("BOOLEAN", {"default": False, "tooltip": "Invert blending direction (High→Low instead of Low→High)"}),
                "adaptive_mode": (["off", "frequency_analysis", "basic_variance"], {"default": "frequency_analysis", "tooltip": "Modulates the blend curve based on image complexity.\nFrequency Analysis: Uses FFT to detect texture vs structure.\nBasic Variance: Simple contrast check."}),
                "tuning_goal": (["Standard", "Encourage Motion", "Preserve Details"], {"default": "Standard", "tooltip": "How to interpret the complexity:\nStandard: Balanced.\nEncourage Motion: For simple images, boosts High/Structure LoRA to force movement.\nPreserve Details: For complex images, reduces High LoRA to protect textures."}),
                "adaptive_intensity": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 5.0, "step": 0.1, "tooltip": "Strength of the adaptive effect."}),
                "preview_only": ("BOOLEAN", {"default": False, "tooltip": "Generate preview without processing LoRAs"}),
                "diagram_enabled": ("BOOLEAN", {"default": True, "tooltip": "Show the blend curve diagram in the UI."}),
            },
            "optional": {
                "custom_curve": ("FLOAT", {"forceInput": True, "default": []}),
                "images": ("IMAGE", {"tooltip": "Input images for adaptive analysis (Single image for I2V, or video frames)"}),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = ("WANVIDLORA", "WANVIDLORA")
    RETURN_NAMES = ("lora_high", "lora_low")
    FUNCTION = "run"
    CATEGORY = "WanVideo/Scheduling"

    def run(self, lora_high, lora_low, steps, start_step, end_step, blend_strategy, invert, adaptive_mode, tuning_goal, adaptive_intensity, preview_only, diagram_enabled, custom_curve=None, images=None, unique_id=None):
        print(f"\n{'='*60}")
        print(f"[Adaptive LoRA Scheduler] Executing...")
        print(f"{'='*60}")
        
        if steps < 1: steps = 1
        
        # Handle end_step defaults
        if end_step < 0 or end_step > steps: end_step = steps
        if start_step > end_step: start_step = 0
            
        start_step = max(0, min(start_step, steps))
        end_step = max(start_step, min(end_step, steps))
        
        duration = end_step - start_step
        
        print(f"  Strategy: {blend_strategy}")
        print(f"  Total Steps: {steps}")
        print(f"  Active Range: {start_step} → {end_step} (duration: {duration})")
        
        # 1. Generate Curve (0.0 -> 1.0)
        # We start with a standard 0->1 transition pattern
        if duration > 0:
            t = np.linspace(0, 1, duration)
            if blend_strategy == "linear":
                curve_segment = t
            elif blend_strategy == "sigmoid":
                curve_segment = 1 / (1 + np.exp(-10 * (t - 0.5)))
            elif blend_strategy == "ease_in":
                curve_segment = t * t
            elif blend_strategy == "ease_out":
                curve_segment = 1 - (1 - t) ** 2
            elif blend_strategy == "step":
                curve_segment = np.where(t < 0.5, 0.0, 1.0)
            else:
                curve_segment = t

            # --- Adaptive Logic ---
            if adaptive_mode != "off" and images is not None:
                print(f"  Adaptive Mode: {adaptive_mode} (Goal: {tuning_goal}, Int: {adaptive_intensity})")
                try:
                    complexity_score = self._calculate_complexity(images, adaptive_mode)
                    print(f"  > Measured Complexity (Freq Ratio): {complexity_score:.3f}")
                    
                    # LOGIC:
                    # complexity_score is 0.0 (Smooth) to 1.0 (Texture/Noise).
                    # modifier = exponent for curve (0->1).
                    # curve^k.
                    # k < 1: Convex (Bulges up).
                    # k > 1: Concave (Sags down).
                    
                    # Target:
                    # Low LoRA = 1.0 - Curve.
                    # High LoRA = Curve. (Wait, let's map it clearly)
                    
                    # If we perform High->Low fade (Standard):
                    # High Starts 1.0, Ends 0.0.
                    # High = 1.0 - curve_segment.
                    
                    # If we want High LoRA STRONGER (Stay high longer):
                    # We need (1.0 - curve) to stay high.
                    # So curve must stay low (Sag).
                    # So curve must be concave.
                    # So exponent > 1.0.
                    
                    # If we want High LoRA WEAKER (Drop fast):
                    # We need (1.0 - curve) to drop fast.
                    # So curve must rise fast (Bulge).
                    # So curve must be convex.
                    # So exponent < 1.0.
                    
                    # LOGIC MAPPING:
                    # 1. Base Modifier calculate from complexity (-1 to +1 range roughly)
                    # Complexity 0.5 is neutral.
                    
                    base_mod = (complexity_score - 0.5) * 2.0 * adaptive_intensity
                    # Range: -intensity to +intensity.
                    
                    # 2. Apply Tuning Goal
                    # Standard:
                    #   Complex Image (Score > 0.5) -> High Detail -> We want to PROTECT details.
                    #   So we want Low LoRA (Detail Model) to take over sooner?
                    #   Or do we want High LoRA (Structure) to respect the image?
                    #   Usually High Noise destroys detail. So for Complex Image, we want High LoRA WEAKER (Drop fast).
                    #   So Complex -> Exponent < 1.0 (Bulge).
                    #   But let's stick to the previous implementation's direction unless user overrides.
                    #   Previous: Complex -> Exponent > 1.0 (Sag / Stay High). This might be why user complained about finding combos.
                    
                    final_exponent = 1.0
                    
                    if tuning_goal == "Preserve Details":
                        # Goal: Protect textures.
                        # Complex (1.0) -> Drop High LoRA Fast -> Exponent < 1.0.
                        # Smooth (0.0) -> Can handle High LoRA -> Exponent > 1.0.
                        if complexity_score > 0.5:
                             # High Detail -> Drop fast -> Reduce exponent
                             final_exponent = 1.0 / (1.0 + base_mod) 
                        else:
                             # Low Detail -> Keep structure -> Increase exponent
                             final_exponent = 1.0 + abs(base_mod)
                             
                    elif tuning_goal == "Encourage Motion":
                        # Goal: Force movement even if detailed.
                        # Complex (1.0) -> Keep High LoRA Strong -> Exponent > 1.0.
                        final_exponent = 1.0 + abs(base_mod)
                        
                    else: # Standard / Balanced
                        # Hybrid approach.
                        # Usually: detailed images need careful handling.
                        if complexity_score > 0.6:
                             # Very complex -> Protect slightly
                             final_exponent = 0.8
                        elif complexity_score < 0.3:
                             # Very smooth -> Boost motion
                             final_exponent = 1.2
                        else:
                             final_exponent = 1.0

                    print(f"  > Calculated Exponent: {final_exponent:.3f}")
                    curve_segment = np.power(curve_segment, final_exponent)
                    
                except Exception as e:
                    print(f"\033[91m[Adaptive Error] Failed to calculate complexity: {e}\033[0m")
            # ----------------------
            
            # 2. Logic: High LoRA Strength
            # DEFAULT (invert=False): High -> Low (1.0 -> 0.0)
            # This matches standard diffusion (High Noise -> Low Noise)
            # So we want curve to go DOWN.
            # We take 1.0 - segment (which goes 1->0)
            
            if not invert:
                # High->Low
                active_segment = 1.0 - curve_segment
                print(f"  Mode: Normal (High LoRA 1.0 → 0.0)")
                
                # Full curve assembly
                full_curve = np.zeros(steps)
                full_curve[:start_step] = 1.0        # Start High
                full_curve[start_step:end_step] = active_segment # Fade down
                full_curve[end_step:] = 0.0          # End Low
            else:
                # Invert=True: Low->High (0.0 -> 1.0)
                # High LoRA starts at 0.0 and goes to 1.0
                active_segment = curve_segment
                print(f"  Mode: Inverted (High LoRA 0.0 → 1.0)")
                
                full_curve = np.zeros(steps)
                full_curve[:start_step] = 0.0        # Start Low
                full_curve[start_step:end_step] = active_segment # Fade up
                full_curve[end_step:] = 1.0          # End High
        else:
             full_curve = np.zeros(steps)

        full_curve = np.clip(full_curve, 0.0, 1.0)
        
        # Log sample points
        print(f"\n  Per-Step Blend Weights (High LoRA Strength):")
        sample_indices = [0, steps//4, steps//2, 3*steps//4, steps-1]
        for idx in sample_indices:
            if idx < steps:
                val = full_curve[idx]
                print(f"    Step {idx:3d}: High={val:.1%}, Low={1.0-val:.1%}")
        
        print(f"\n  → LoRAs will be BLENDED with these weights at each step")
        print(f"{'='*60}\n")
        
        # 3. Apply to LoRAs
        out_lora_high = []
        if isinstance(lora_high, list):
            for l in lora_high:
                l_copy = l.copy()
                l_copy["strength"] = full_curve.tolist() 
                l_copy["merge_loras"] = False 
                out_lora_high.append(l_copy)
        
        out_lora_low = []
        low_curve = 1.0 - full_curve
        if isinstance(lora_low, list):
            for l in lora_low:
                l_copy = l.copy()
                l_copy["strength"] = low_curve.tolist()
                l_copy["merge_loras"] = False 
                out_lora_low.append(l_copy)
                
        # 4. Update Backend Visuals
        # (Pass tuning info to UI)
        adp_info = {'complexity': complexity_score, 'modifier': final_exponent} if (adaptive_mode != "off" and images is not None) else None
        
        if unique_id and diagram_enabled:
            self.update_node_ui(unique_id, full_curve, start_step, end_step, steps, adp_info)

        return (out_lora_high, out_lora_low)

    def update_node_ui(self, unique_id, curve, start_step, end_step, total_steps, adaptive_info=None):
        try:
            from server import PromptServer
            import io
            import base64
            import matplotlib.pyplot as plt
            
            if PromptServer.instance is None: return

            # Setup plot
            plt.figure(figsize=(6, 2.5), dpi=100) # Slightly taller for labels
            fig = plt.gcf()
            ax = plt.gca()
            
            # Dark mode style
            fig.patch.set_facecolor('#202020')
            ax.set_facecolor('#202020')
            ax.tick_params(colors='#aaaaaa', labelsize=8)
            for spine in ax.spines.values():
                spine.set_edgecolor('#444444')
            
            # Plot
            x = np.arange(total_steps)
            y_high = curve
            y_low = 1.0 - curve
            
            # Fill area
            ax.fill_between(x, y_high, alpha=0.3, color='#44aaff')
            # High LoRA Curve (Blue)
            ax.plot(x, y_high, color='#44aaff', linewidth=2, label='High (End)')
            # Low LoRA Curve (Orange)
            ax.plot(x, y_low, color='#ffaa44', linewidth=2, linestyle=':', label='Low (Start)')
            
            # Find Intersection
            idx_cross = np.abs(y_high - y_low).argmin()
            ax.axvline(x=idx_cross, color='#ffffff', linestyle=':', linewidth=1, alpha=0.6)
            
            # Markers
            ax.axvline(x=start_step, color='#666', linestyle='--', linewidth=1, alpha=0.5)
            ax.axvline(x=end_step, color='#666', linestyle='--', linewidth=1, alpha=0.5)
            # Equilibrium Line (50%)
            ax.axhline(y=0.5, color='#888', linestyle=':', linewidth=1, alpha=0.3)
            # ax.text(0, 0.52, "50% Impact", color='#666', fontsize=6, verticalalignment='bottom')
            
            # Labels
            ax.set_xlabel("Steps", color='#aaaaaa', fontsize=9)
            ax.set_ylabel("Strength", color='#aaaaaa', fontsize=9)
            ax.set_ylim(-0.1, 1.1)
            
            title_text = f"Blend Strength ({total_steps} steps)"
            if adaptive_info:
                title_text += f" | Adaptive: {adaptive_info['modifier']:.2f}x"
                
            ax.set_title(title_text, color='#aaaaaa', fontsize=10, pad=5)
            ax.grid(True, color='#333333', linestyle='--', alpha=0.3)
            ax.legend(loc='upper right', fontsize=7, facecolor='#202020', edgecolor='#444444')

            # Save to buffer
            buf = io.BytesIO()
            plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0.1)
            plt.close()
            buf.seek(0)
            
            # Encode
            img_b64 = base64.b64encode(buf.read()).decode('utf-8')
            
            # Numeric Stats
            sample_idxs = [0, total_steps//4, total_steps//2, 3*total_steps//4, total_steps-1]
            sample_idx_valid = [min(max(i, 0), total_steps-1) for i in sample_idxs]
            
            vals_high = [f"{y_high[i]:.2f}" for i in sample_idx_valid]
            vals_low = [f"{y_low[i]:.2f}" for i in sample_idx_valid]
            
            stats_html = f"""
            <div style="font-size:10px; color:#cccccc; margin-top:5px; font-family:monospace;">
                <div style="display:flex; justify-content:space-between;">
                    <span style="color:#44aaff">High: [{', '.join(vals_high)}]</span>
                </div>
                <div style="display:flex; justify-content:space-between;">
                    <span style="color:#ffaa44">Low : [{', '.join(vals_low)}]</span>
                </div>
            """
            
            if adaptive_info:
                 stats_html += f"""
                <div style="margin-top:3px; border-top:1px solid #444; padding-top:2px;">
                    <b>Adaptive Score:</b> {adaptive_info['complexity']:.2f} | 
                    <b>Mod:</b> {adaptive_info['modifier']:.2f}x
                </div>
                """
            stats_html += "</div>"
            
            html = f"<img src='data:image/png;base64,{img_b64}' style='width:100%; border-radius:4px; opacity:0.9;'>"
            html += stats_html
            
            PromptServer.instance.send_progress_text(html, unique_id)
            
        except Exception as e:
            print(f"[AdaptiveLoraScheduler] UI Update Failed: {e}")

    # --- Metrics Implementation ---
    
    def _calculate_complexity(self, images, mode):
        import time
        t0 = time.time()
        
        # images: [B, H, W, C] tensor
        total_frames = images.shape[0]
        h, w = images.shape[1], images.shape[2]
        
        print(f"    ------------------------------------------")
        print(f"    [Adaptive Analysis] Input: {total_frames} frames, Resolution: {w}x{h}")
        print(f"    [Adaptive Analysis] Mode: {mode}")
        
        # Ensure float
        if images.dtype in (torch.uint8, torch.int16, torch.int32):
            images = images.float() / 255.0
            
        scores = []
        
        # Intelligent Downsampling
        MAX_SAMPLES = 8 # Efficiency
        if total_frames > MAX_SAMPLES:
            step = total_frames // MAX_SAMPLES
            frame_indices = list(range(0, total_frames, step))[:MAX_SAMPLES]
        else:
            frame_indices = list(range(total_frames))
            
        print(f"    [Adaptive Analysis] Sampling {len(frame_indices)} frames")
        
        for i in frame_indices:
            img = images[i] # [H, W, C]
            
            # Convert to Grayscale for Analysis
            gray = 0.299*img[:,:,0] + 0.587*img[:,:,1] + 0.114*img[:,:,2]
            
            if mode == "frequency_analysis":
                # --- FFT Analysis ---
                # We want to measure the ratio of High Frequency energy to Low Frequency energy.
                # High Ratio = Texture/Noise/Edgy.
                # Low Ratio = Smooth/Gradient.
                
                # Move to CPU for numpy FFT (easier than torch fft for simple 2D)
                # Or use torch.fft
                f_transform = torch.fft.fft2(gray)
                f_shift = torch.fft.fftshift(f_transform)
                magnitude_spectrum = 20 * torch.log(torch.abs(f_shift) + 1)
                
                # Mask Center (Low Freq)
                rows, cols = gray.shape
                crow, ccol = rows//2, cols//2
                # Radius for Low Freq
                r = min(rows, cols) // 8
                
                # Total Energy
                total_energy = torch.sum(magnitude_spectrum)
                
                # Low Freq Energy (Center circle approx by square for speed)
                low_freq_energy = torch.sum(magnitude_spectrum[crow-r:crow+r, ccol-r:ccol+r])
                
                high_freq_energy = total_energy - low_freq_energy
                
                # Ratio: High / Total. 
                # Range 0.0 to ~1.0.
                # Usually sits around 0.3 - 0.7.
                ratio = (high_freq_energy / total_energy).item()
                
                # Normalize reasonably (0.2 -> 0.0, 0.8 -> 1.0)
                norm_score = (ratio - 0.2) / 0.6
                safe_score = min(max(norm_score, 0.0), 1.0)
                scores.append(safe_score)
                
            else: # Basic Variance
                var_score = torch.var(gray).item() * 10.0
                safe_score = min(max(var_score, 0.0), 1.0)
                scores.append(safe_score)
            
        avg_score = sum(scores) / len(scores) if scores else 0.0
        
        dt = (time.time() - t0) * 1000.0
        print(f"    [Adaptive Analysis] Finished in {dt:.2f}ms. Avg Score: {avg_score:.4f}")
        print(f"    ------------------------------------------")
        
        return avg_score


