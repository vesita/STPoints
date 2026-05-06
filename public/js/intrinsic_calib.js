/**
 * 相机内参标定控制器
 *
 * 工作流：
 *   1. 选择目标相机，上传棋盘格图片
 *   2. 设置内角点行列数，检测角点
 *   3. 删除检测失败的图片
 *   4. 计算内参 → 保存到标定 JSON
 */
function IntrinsicCalib(data, editor) {
    this.data = data;
    this.editor = editor;
    this.active = false;
    this.cameraName = null;
    this.images = []; // [{file, filename, success, preview, corners, dataUrl}]

    this.wrapper = null;
    this.imageListEl = null;
    this.statusEl = null;
    this.detectBtn = null;
    this.calcBtn = null;
    this.saveBtn = null;
    this.result = null;

    this._queryDom = function () {
        if (this.wrapper) return true;
        this.wrapper = document.getElementById("intrinsic-calib-wrapper");
        if (!this.wrapper) return false;
        this.imageListEl = document.getElementById("intrinsic-image-list");
        this.statusEl = document.getElementById("intrinsic-status");
        this.detectBtn = document.getElementById("intrinsic-detect-btn");
        this.calcBtn = document.getElementById("intrinsic-calc-btn");
        this.saveBtn = document.getElementById("intrinsic-save-btn");
        return true;
    };

    this.init = function () {
        // 延迟绑定，enter 时再 _queryDom
    };

    this.enter = function (cameraName) {
        if (!this._queryDom()) {
            console.error("IntrinsicCalib: panel not found");
            return;
        }
        this.active = true;
        this.cameraName = cameraName || this._getActiveCameraName();
        this.images = [];
        this.result = null;
        this.wrapper.style.display = "";
        this._clearUI();
        this._bindEvents();
        this._updateStatus("选择棋盘格图片，设置内角点行列数后点击\"检测角点\"");
    };

    this.exit = function () {
        this.active = false;
        this.wrapper.style.display = "none";
        this._unbindEvents();
    };

    this._getActiveCameraName = function () {
        if (this.editor && this.editor.imageContextManager && this.editor.imageContextManager.bestCamera) {
            return this.editor.imageContextManager.bestCamera;
        }
        try {
            return this.data.world.cameras.names[0];
        } catch (e) {
            return null;
        }
    };

    // ── 事件绑定 ──────────────────────────────────────────────────────

    this._boundHandlers = {};

    this._bindEvents = function () {
        var self = this;
        var panel = document.getElementById("intrinsic-calib-panel");

        this._boundHandlers.exit = function () { self.exit(); };
        this._boundHandlers.upload = function () {
            document.getElementById("intrinsic-file-input").click();
        };
        this._boundHandlers.fileChange = function (e) { self._onFilesSelected(e); };
        this._boundHandlers.detect = function () { self._detectCorners(); };
        this._boundHandlers.calc = function () { self._calculate(); };
        this._boundHandlers.save = function () { self._save(); };

        document.getElementById("intrinsic-calib-exit").addEventListener("click", this._boundHandlers.exit);
        document.getElementById("intrinsic-upload-btn").addEventListener("click", this._boundHandlers.upload);
        document.getElementById("intrinsic-file-input").addEventListener("change", this._boundHandlers.fileChange);
        this.detectBtn.addEventListener("click", this._boundHandlers.detect);
        this.calcBtn.addEventListener("click", this._boundHandlers.calc);
        this.saveBtn.addEventListener("click", this._boundHandlers.save);
    };

    this._unbindEvents = function () {
        document.getElementById("intrinsic-calib-exit").removeEventListener("click", this._boundHandlers.exit);
        document.getElementById("intrinsic-upload-btn").removeEventListener("click", this._boundHandlers.upload);
        document.getElementById("intrinsic-file-input").removeEventListener("change", this._boundHandlers.fileChange);
        this.detectBtn.removeEventListener("click", this._boundHandlers.detect);
        this.calcBtn.removeEventListener("click", this._boundHandlers.calc);
        this.saveBtn.removeEventListener("click", this._boundHandlers.save);
    };

    // ── 文件选择 ──────────────────────────────────────────────────────

    this._onFilesSelected = function (e) {
        var files = Array.from(e.target.files);
        if (files.length === 0) return;

        this.images = [];
        this.result = null;
        this._hideResult();
        this.detectBtn.disabled = false;
        this.calcBtn.disabled = true;
        this.saveBtn.disabled = true;

        var self = this;
        var loaded = 0;

        files.forEach(function (file) {
            var reader = new FileReader();
            reader.onload = function (ev) {
                self.images.push({
                    file: file,
                    filename: file.name,
                    success: null,
                    preview: null,
                    corners: null,
                    dataUrl: ev.target.result
                });
                loaded++;
                if (loaded === files.length) {
                    self._renderImageList();
                    self._updateStatus("已加载 " + files.length + " 张图片，点击\"检测角点\"");
                }
            };
            reader.readAsDataURL(file);
        });
    };

    // ── 角点检测 ──────────────────────────────────────────────────────

    this._detectCorners = function () {
        if (this.images.length === 0) return;

        var rows = parseInt(document.getElementById("intrinsic-rows").value) || 6;
        var cols = parseInt(document.getElementById("intrinsic-cols").value) || 9;

        var formData = new FormData();
        formData.append("rows", rows);
        formData.append("cols", cols);
        for (var i = 0; i < this.images.length; i++) {
            formData.append("images", this.images[i].file);
        }

        this._updateStatus("正在检测角点...");
        this.detectBtn.disabled = true;

        var self = this;
        fetch("/detect_corners", {
            method: "POST",
            body: formData
        }).then(function (resp) {
            return resp.json();
        }).then(function (data) {
            self._onDetectResult(data);
        }).catch(function (err) {
            self._updateStatus("检测失败: " + err.message);
            self.detectBtn.disabled = false;
        });
    };

    this._onDetectResult = function (data) {
        var results = data.results || [];
        var successCount = 0;

        for (var i = 0; i < results.length; i++) {
            var r = results[i];
            var img = this.images.find(function (x) { return x.filename === r.filename; });
            if (!img) continue;

            img.success = r.success;
            img.preview = r.preview || null;
            img.corners = r.corners || null;
            if (r.success) successCount++;
        }

        this._renderImageList();

        var failCount = results.length - successCount;
        // 构建状态消息，如果有失败图片则附带清除按钮
        this.statusEl.innerHTML = "";
        var span = document.createElement("span");
        span.textContent = "检测完成: " + successCount + " 成功, " + failCount + " 失败。";
        this.statusEl.appendChild(span);

        if (failCount > 0) {
            var self = this;
            var clearBtn = document.createElement("button");
            clearBtn.className = "pnp-btn";
            clearBtn.textContent = "清除失败图片";
            clearBtn.style.cssText = "margin-left:8px;padding:2px 8px;font-size:12px;";
            clearBtn.addEventListener("click", function () {
                self.images = self.images.filter(function (x) { return x.success; });
                self._renderImageList();
                var okCount = self.images.length;
                self.calcBtn.disabled = okCount < 3;
                self.statusEl.innerHTML = "";
                var s = document.createElement("span");
                s.textContent = "已清除失败图片，剩余 " + okCount + " 张。";
                self.statusEl.appendChild(s);
            });
            this.statusEl.appendChild(clearBtn);
        }

        this.detectBtn.disabled = false;
        this.calcBtn.disabled = successCount < 3;
        this.result = null;
        this._hideResult();
    };

    // ── 图片列表渲染 ─────────────────────────────────────────────────

    this._renderImageList = function () {
        var list = this.imageListEl;
        list.innerHTML = "";
        var self = this;

        this.images.forEach(function (img, idx) {
            var card = document.createElement("div");
            card.className = "intrinsic-img-card" +
                (img.success === true ? " success" : img.success === false ? " fail" : "");

            var imgEl = document.createElement("img");
            if (img.success && img.preview) {
                imgEl.src = "data:image/jpeg;base64," + img.preview;
            } else {
                imgEl.src = img.dataUrl;
            }
            card.appendChild(imgEl);

            var nameEl = document.createElement("div");
            nameEl.className = "intrinsic-img-name";
            nameEl.textContent = img.filename;
            nameEl.title = img.filename;
            card.appendChild(nameEl);

            var removeBtn = document.createElement("button");
            removeBtn.className = "intrinsic-img-remove";
            removeBtn.textContent = "×";
            removeBtn.addEventListener("click", function (e) {
                e.stopPropagation();
                self.images.splice(idx, 1);
                self._renderImageList();
                var okCount = self.images.filter(function (x) { return x.success; }).length;
                self.calcBtn.disabled = okCount < 3;
                if (self.images.length === 0) {
                    self.detectBtn.disabled = true;
                    self._updateStatus("所有图片已删除");
                }
            });
            card.appendChild(removeBtn);

            list.appendChild(card);
        });
    };

    // ── 计算内参 ──────────────────────────────────────────────────────

    this._calculate = function () {
        var rows = parseInt(document.getElementById("intrinsic-rows").value) || 6;
        var cols = parseInt(document.getElementById("intrinsic-cols").value) || 9;

        // 收集成功的图片
        var imageDataList = [];
        var pending = 0;
        var self = this;

        var successImages = this.images.filter(function (x) { return x.success && x.corners; });
        if (successImages.length < 3) {
            this._updateStatus("有效图片不足，至少需要3张");
            return;
        }

        this._updateStatus("正在计算内参...");
        this.calcBtn.disabled = true;

        // 将每张图片转为 base64（去掉 data:image/...;base64, 前缀）
        successImages.forEach(function (img) {
            var base64 = img.dataUrl.split(",")[1];
            imageDataList.push({
                filename: img.filename,
                corners: img.corners,
                image_base64: base64
            });
            pending++;
        });

        fetch("/calibrate_intrinsics", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                rows: rows,
                cols: cols,
                images: imageDataList
            })
        }).then(function (resp) {
            return resp.json();
        }).then(function (data) {
            self._onCalibrateResult(data);
        }).catch(function (err) {
            self._updateStatus("计算失败: " + err.message);
            self.calcBtn.disabled = false;
        });
    };

    this._onCalibrateResult = function (data) {
        if (!data.success) {
            this._updateStatus("计算失败: " + (data.error || "未知错误"));
            this.calcBtn.disabled = false;
            return;
        }

        this.result = data;
        this._showResult(data);
        this._updateStatus("内参计算完成，重投影误差: " + data.error.toFixed(4) + " px");
        this.calcBtn.disabled = false;
        this.saveBtn.disabled = false;
    };

    // ── 结果显示 ──────────────────────────────────────────────────────

    this._showResult = function (data) {
        var resultEl = document.getElementById("intrinsic-result");
        resultEl.style.display = "";

        document.getElementById("intrinsic-img-count").textContent = data.image_count;
        document.getElementById("intrinsic-error").textContent = data.error.toFixed(4);

        // 显示 3x3 内参矩阵
        var K = data.intrinsic;
        var matrixText =
            "[" + K[0].toFixed(2) + ", " + K[1].toFixed(2) + ", " + K[2].toFixed(2) + "]\n" +
            "[" + K[3].toFixed(2) + ", " + K[4].toFixed(2) + ", " + K[5].toFixed(2) + "]\n" +
            "[" + K[6].toFixed(2) + ", " + K[7].toFixed(2) + ", " + K[8].toFixed(2) + "]";
        document.getElementById("intrinsic-matrix").value = matrixText;

        // 显示畸变系数
        var D = data.dist_coeffs;
        var distText = D.map(function (v) { return v.toFixed(6); }).join(", ");
        document.getElementById("intrinsic-dist").textContent = "[" + distText + "]";
    };

    this._hideResult = function () {
        document.getElementById("intrinsic-result").style.display = "none";
    };

    // ── 保存 ─────────────────────────────────────────────────────────

    this._save = function () {
        if (!this.result || !this.cameraName) return;

        var scene = this.data.world.frameInfo.scene;
        var self = this;

        this.saveBtn.disabled = true;
        this._updateStatus("正在保存...");

        fetch("/save_intrinsics", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                scene: scene,
                camera: this.cameraName,
                intrinsic: this.result.intrinsic,
                dist_coeffs: this.result.dist_coeffs
            })
        }).then(function (resp) {
            return resp.json();
        }).then(function (data) {
            if (data.success) {
                self._updateStatus("已保存到标定文件: " + scene + "/calib/camera/" + self.cameraName + ".json");
            } else {
                self._updateStatus("保存失败: " + (data.error || "未知错误"));
                self.saveBtn.disabled = false;
            }
        }).catch(function (err) {
            self._updateStatus("保存失败: " + err.message);
            self.saveBtn.disabled = false;
        });
    };

    // ── UI 工具 ───────────────────────────────────────────────────────

    this._updateStatus = function (msg) {
        if (this.statusEl) this.statusEl.textContent = msg;
    };

    this._clearUI = function () {
        this.imageListEl.innerHTML = "";
        this.statusEl.textContent = "";
        this.detectBtn.disabled = true;
        this.calcBtn.disabled = true;
        this.saveBtn.disabled = true;
        document.getElementById("intrinsic-file-input").value = "";
        this._hideResult();
    };
}

export { IntrinsicCalib };
