import { app } from "../../scripts/app.js";

app.registerExtension({
    name: "Comfy.AdaptiveLoraScheduler",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "AdaptiveLoraScheduler") {
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;

                // Initialize curve data as empty initially
                this.curveData = null;
                this.scheduleMode = "normal";

                // Add custom canvas widget for visualization - ALWAYS VISIBLE
                const widget = this.addWidget("custom", "Blend Curve", "curve_preview", function (ctx, node, widget_width, y, widget_height) {
                    const margin = 10;
                    const h = widget_height - margin * 2;
                    const w = widget_width - margin * 2;
                    const x = margin;
                    const top = y + margin;

                    // Background
                    ctx.fillStyle = "#1a1a1a";
                    ctx.fillRect(x, top, w, h);

                    // Border
                    ctx.strokeStyle = "#444";
                    ctx.lineWidth = 1;
                    ctx.strokeRect(x, top, w, h);

                    // Grid lines
                    ctx.strokeStyle = "#333";
                    ctx.setLineDash([2, 2]);
                    ctx.beginPath();
                    ctx.moveTo(x, top + h / 2);
                    ctx.lineTo(x + w, top + h / 2);
                    ctx.stroke();
                    ctx.setLineDash([]);

                    if (node.curveData && node.curveData.length > 0) {
                        // Draw curve
                        ctx.beginPath();
                        const isPreview = node.scheduleMode === "preview";
                        const isActiveBlend = node.scheduleMode === "active_blend";
                        ctx.strokeStyle = isPreview ? "#FFA500" : (isActiveBlend ? "#00FF00" : "#4AF");
                        ctx.lineWidth = 2;

                        const data = node.curveData;
                        const len = data.length;

                        for (let i = 0; i < len; i++) {
                            const val = data[i]; // 0.0 to 1.0
                            const px = x + (i / Math.max(1, len - 1)) * w;
                            const py = top + h - (val * h);

                            if (i === 0) ctx.moveTo(px, py);
                            else ctx.lineTo(px, py);
                        }
                        ctx.stroke();

                        // Labels
                        ctx.fillStyle = "#AAA";
                        ctx.font = "10px monospace";
                        ctx.fillText("Low", x + 2, top + h - 2);
                        ctx.fillText("High", x + w - 26, top + 12);
                        ctx.fillText(len + " steps", x + w / 2 - 20, top + h - 2);

                        // Mode indicator
                        if (isPreview) {
                            ctx.fillStyle = "#FFA500";
                            ctx.font = "bold 11px monospace";
                            ctx.fillText("PREVIEW MODE", x + 4, top + 12);
                        } else if (isActiveBlend) {
                            ctx.fillStyle = "#00FF00";
                            ctx.font = "bold 11px monospace";
                            ctx.fillText("ACTIVE BLEND", x + 4, top + 12);
                        }
                    } else {
                        // Enhanced placeholder
                        ctx.fillStyle = "#555";
                        ctx.font = "12px monospace";
                        const placeholderText = "▶ Execute to see blend curve";
                        ctx.fillText(placeholderText, x + w / 2 - 100, top + h / 2);
                        ctx.font = "10px monospace";
                        ctx.fillText("(Graph updates after first run)", x + w / 2 - 85, top + h / 2 + 15);

                        // Draw sample curve shape
                        ctx.strokeStyle = "#333";
                        ctx.lineWidth = 1;
                        ctx.setLineDash([3, 3]);
                        ctx.beginPath();
                        for (let i = 0; i <= 20; i++) {
                            const t = i / 20;
                            const px = x + t * w;
                            const py = top + h - (t * h);
                            if (i === 0) ctx.moveTo(px, py);
                            else ctx.lineTo(px, py);
                        }
                        ctx.stroke();
                        ctx.setLineDash([]);
                    }
                }, { height: 120 });

                return r;
            };

            // Handle execution results to update curve data
            const onExecuted = nodeType.prototype.onExecuted;
            nodeType.prototype.onExecuted = function (message) {
                const r = onExecuted ? onExecuted.apply(this, arguments) : undefined;

                if (message && message.curve) {
                    this.curveData = message.curve;
                    this.scheduleMode = message.mode || "normal";

                    // Force canvas redraw
                    if (this.onResize) {
                        this.onResize(this.size);
                    }
                    this.setDirtyCanvas(true, true);
                }

                return r;
            };
        }
    }
});
