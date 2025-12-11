from .nodes.adaptive_lora_scheduler import AdaptiveLoraScheduler

NODE_CLASS_MAPPINGS = {
    "AdaptiveLoraScheduler": AdaptiveLoraScheduler
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AdaptiveLoraScheduler": "Adaptive LoRA Scheduler"
}

WEB_DIRECTORY = "./web"

print("\033[94mAdaptive LoRA Scheduler Node: \033[92mLoaded\033[0m")
