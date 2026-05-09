import {globalKeyDownManager} from "./keydown_manager.js";

export class CalibPreview {
    constructor(editor) {
        this.editor = editor;
        this.active = false;
        this.wrapper = document.querySelector("#calib-preview-wrapper");
        this.panel = document.querySelector("#calib-preview-panel");
        this.gallery = document.querySelector("#calib-preview-gallery");
        this.statusEl = document.querySelector("#calib-preview-status");
        this.generateBtn = document.querySelector("#calib-preview-generate-btn");
        this.exitBtn = document.querySelector("#calib-preview-exit");
        this._frames = [];

        if (!this.wrapper || !this.gallery || !this.generateBtn) {
            console.warn("CalibPreview: missing required DOM elements");
            return;
        }

        this.generateBtn.onclick = () => this._generate();
        if (this.exitBtn) {
            this.exitBtn.onclick = () => this.exit();
        }
        this.wrapper.onclick = (e) => {
            if (e.target === this.wrapper) this.exit();
        };
    }

    enter() {
        if (this.active) return;
        this.active = true;
        this.wrapper.style.display = "flex";
        this.gallery.innerHTML = "";
        this.statusEl.textContent = "";
        globalKeyDownManager.register(() => false, "calib-preview");
    }

    exit() {
        if (!this.active) return;
        this.active = false;
        this.wrapper.style.display = "none";
        this.gallery.innerHTML = "";
        globalKeyDownManager.deregister("calib-preview");
    }

    async _generate() {
        const world = this.editor.data.world;
        if (!world) {
            this.statusEl.textContent = "未加载场景";
            return;
        }
        this.generateBtn.disabled = true;
        this.statusEl.textContent = "正在生成...";
        this.gallery.innerHTML = "";

        try {
            const sceneName = world.frameInfo ? world.frameInfo.scene : world.sceneMeta?.scene;
            if (!sceneName) {
                this.statusEl.textContent = "无法获取场景名称";
                return;
            }
            const resp = await fetch("/render_calibration_preview", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ scene: sceneName })
            });
            const result = await resp.json();

            if (!result.success) {
                this.statusEl.textContent = "失败: " + (result.error || "未知错误");
                return;
            }

            this.statusEl.textContent = `共 ${result.total_frames} 帧 | camera: ${result.camera}`;

            if (result.frames.length === 0) {
                this.statusEl.textContent += "（无带标注的帧）";
                return;
            }

            result.frames.forEach(f => {
                const card = document.createElement("div");
                card.className = "calib-preview-card";

                const img = document.createElement("img");
                img.src = f.image_url + "?t=" + Date.now();
                img.alt = `frame ${f.frame}`;
                img.loading = "lazy";

                const info = document.createElement("div");
                info.className = "calib-preview-card-info";
                info.textContent = `${f.frame} · ${f.num_boxes} 个物体`;

                card.appendChild(img);
                card.appendChild(info);
                card.onclick = () => {
                    // 点击放大查看
                    if (card.classList.contains("calib-preview-card-expanded")) {
                        card.classList.remove("calib-preview-card-expanded");
                    } else {
                        document.querySelectorAll(".calib-preview-card-expanded")
                            .forEach(c => c.classList.remove("calib-preview-card-expanded"));
                        card.classList.add("calib-preview-card-expanded");
                    }
                };
                this.gallery.appendChild(card);
            });
        } catch (err) {
            this.statusEl.textContent = "请求失败: " + err.message;
        } finally {
            this.generateBtn.disabled = false;
        }
    }
}
