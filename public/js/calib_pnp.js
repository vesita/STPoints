import * as THREE from './lib/three.module.js';
import { psr_to_xyz } from "./util.js";
import { box_to_2d_points, points3d_homo_to_image2d } from "./image.js";

/**
 * PnP 外参矫正控制器
 *
 * 工作流：
 *   1. 用户选中 3D box → 提取正面 4 角点 3D 坐标 (P0~P3)
 *   2. 弹出独立标定窗口，加载相机图像并渲染 4 个可拖拽角点
 *   3. 用户直接在图像拖拽 P0~P3 到实际位置
 *   4. 调用 POST /solve_pnp → IPPE 求解新外参
 *   5. 预览更新后的投影，满意则保存
 */
function PnPCalib(data, editor) {
    this.data = data;
    this.editor = editor;

    // ── 状态 ──────────────────────────────────────────────────────────────
    this.active = false;
    this.corners = [null, null, null, null];
    this.points_3d = null;
    this.result = null;

    // ── DOM 缓存 ──────────────────────────────────────────────────────────
    this.wrapper = null;
    this.svg = null;
    this.handleGroup = null;
    this.imageEl = null;
    this.cornerEls = [];
    this.errorEl = null;
    this.solveBtn = null;
    this.resetBtn = null;
    this.saveBtn = null;
    this._draggingIdx = -1;

    // ── Z-view 角点标签 ─────────────────────────────────────────────────────
    this._zViewLabels = null;
    this._zViewObserver = null;

    // ── 3D 角点选择 ─────────────────────────────────────────────────────────
    this._cornerMarkerGroup = null;
    this._cornerClickHandler = null;
    this._selectedCornerIndices = []; // 用户选中的 8 角点中的索引
    this._allMarkers3d = null;        // 12 个标记：4 边中点 + 4 前角 + 4 后角

    this.init = function () {
        // DOM 元素由 _queryDom 延迟查询
    };

    /** 查询并缓存弹窗 DOM 元素 */
    this._queryDom = function () {
        if (this.wrapper) return true;
        this.wrapper = document.getElementById("pnp-sidebar-wrapper");
        if (!this.wrapper) return false;

        var view = this.wrapper.querySelector("#pnp-sidebar");
        this.svg = view.querySelector("#pnp-calib-svg");
        this.handleGroup = this.svg.querySelector("#pnp-calib-handles");
        this.imageEl = this.svg.querySelector("#pnp-calib-img");

        var ctrl = view.querySelector("#pnp-calib-controls");
        for (let i = 0; i < 4; i++) {
            this.cornerEls[i] = ctrl.querySelector(
                `.pnp-corner[data-idx="${i}"] .pnp-corner-pos`
            );
        }
        this.errorEl = ctrl.querySelector("#pnp-calib-error");
        this.solveBtn = ctrl.querySelector("#pnp-calib-solve");
        this.manual3dBtn = ctrl.querySelector("#pnp-calib-manual-3d");
        this.bruteBtn = ctrl.querySelector("#pnp-calib-brute");
        this.resetBtn = ctrl.querySelector("#pnp-calib-reset");
        this.saveBtn = ctrl.querySelector("#pnp-calib-save");
        this.onlyBoxCornersCheckbox = ctrl.querySelector("#pnp-only-box-corners");

        var pnp = this;
        this.solveBtn.onclick = function () { pnp.solve(); };
        this.manual3dBtn.onclick = function () { pnp.startManual3DSelection(); };
        this.bruteBtn.onclick = function () { pnp.bruteForce(); };
        this.resetBtn.onclick = function () { pnp.reset(); };
        this.saveBtn.onclick = function () { pnp.save(); };

        // 只捕捉 box 角点复选框
        if (this.onlyBoxCornersCheckbox) {
            this.onlyBoxCornersCheckbox.onchange = function () {
                if (pnp.editor.measureTool) {
                    pnp.editor.measureTool.onlyBoxCorners = this.checked;
                }
            };
        }
        view.querySelector("#pnp-calib-exit").onclick = function () { pnp.exit(); };

        // 点击背景关闭弹窗
        this.wrapper.onclick = function () { pnp.exit(); };
        view.onclick = function (e) { e.stopPropagation(); };

        // 弹窗拖拽
        var header = view.querySelector("#pnp-sidebar-header");
        this._initDrag(header, view);

        this.svg.addEventListener("mousemove", function (e) { pnp._onMouseMove(e); });
        this.svg.addEventListener("mouseup", function ()   { pnp._onMouseUp(); });
        this.svg.addEventListener("mouseleave", function (){ pnp._onMouseUp(); });

        // 点击图像直接放置角点（同时自动设置3D坐标）
        this.svg.addEventListener("click", function (e) {
            if (!pnp.active) return;
            // 如果已经有4个角点，不处理
            if (pnp.corners.every(function (c) { return c !== null; })) return;
            // 获取图片坐标（检查是否在图片区域内）
            var imgCoords = pnp._getImageCoords(e);
            if (!imgCoords) return; // 点击在图片外部，忽略
            var u = imgCoords.u;
            var v = imgCoords.v;
            // 找到第一个空的角点位置
            var pIdx = pnp.corners.findIndex(function (c) { return c === null; });
            if (pIdx < 0) return;
            // 设置2D角点位置
            pnp.corners[pIdx] = { u: u, v: v };
            // 自动从box边中点获取3D坐标（x=0 平面）
            if (pnp._allMarkers3d && pIdx < 4) {
                pnp.points_3d[pIdx] = pnp._allMarkers3d[pIdx];
                var pt = pnp._allMarkers3d[pIdx];
                console.log("PnPCalib: placed P" + pIdx + " at (" + u.toFixed(1) + ", " + v.toFixed(1) + ")" +
                    " 3D=(" + pt[0].toFixed(3) + "," + pt[1].toFixed(3) + "," + pt[2].toFixed(3) + ")");
            } else {
                console.log("PnPCalib: placed P" + pIdx + " at (" + u.toFixed(1) + ", " + v.toFixed(1) + ") [no 3D data]");
            }
            // 更新 UI
            pnp._updateUI();
            pnp._renderHandles();
            pnp.editor.imageContextManager.renderPnpHandles(pnp.corners);
        });

        return true;
    };

    // ── SVG 坐标转换 ──────────────────────────────────────────────────────

    /** 将鼠标事件转为图片像素坐标，null 表示在图片外 */
    this._getImageCoords = function (event) {
        var rect = this.svg.getBoundingClientRect();
        var vb = this.svg.viewBox.baseVal;
        if (!vb || vb.width === 0 || vb.height === 0) return null;
        var u = (event.clientX - rect.left) / rect.width * vb.width;
        var v = (event.clientY - rect.top) / rect.height * vb.height;
        if (u < 0 || u > vb.width || v < 0 || v > vb.height) return null;
        return { u: u, v: v };
    };

    // ── 弹窗拖拽 ──────────────────────────────────────────────────────────

    /** 初始化弹窗拖拽（通过 header 拖动整个弹窗） */
    this._initDrag = function (header, view) {
        var dragging = false;
        var startX, startY, origLeft, origTop;

        header.addEventListener("mousedown", function (e) {
            if (e.target.closest("#buttons")) return; // 不拦截按钮点击
            dragging = true;
            startX = e.clientX;
            startY = e.clientY;
            var rect = view.getBoundingClientRect();
            var parentRect = view.parentElement.getBoundingClientRect();
            origLeft = rect.left - parentRect.left;
            origTop = rect.top - parentRect.top;
            // 切换为像素定位
            view.style.left = origLeft + "px";
            view.style.top = origTop + "px";
            e.preventDefault();
        });

        document.addEventListener("mousemove", function (e) {
            if (!dragging) return;
            view.style.left = (origLeft + e.clientX - startX) + "px";
            view.style.top = (origTop + e.clientY - startY) + "px";
        });

        document.addEventListener("mouseup", function () {
            dragging = false;
        });
    };

    // ── 弹窗 SVG 角点渲染 ─────────────────────────────────────────────────

    /** 在弹窗 SVG 上绘制 4 个拖拽手柄 */
    this._renderHandles = function () {
        while (this.handleGroup.firstChild) {
            this.handleGroup.firstChild.remove();
        }

        for (var i = 0; i < 4; i++) {
            var c = this.corners[i];
            if (!c) continue;

            var g = document.createElementNS("http://www.w3.org/2000/svg", "g");
            g.setAttribute("class", "pnp-handle");
            g.setAttribute("data-idx", i);
            this.handleGroup.appendChild(g);

            var circle = document.createElementNS("http://www.w3.org/2000/svg", "circle");
            circle.setAttribute("cx", c.u);
            circle.setAttribute("cy", c.v);
            circle.setAttribute("r", 10);
            circle.setAttribute("class", "pnp-handle-circle");
            circle.setAttribute("data-idx", i);
            var pnp = this;
            circle.onmousedown = function (e) {
                e.stopPropagation();
                pnp._draggingIdx = parseInt(this.getAttribute("data-idx"));
                this.style.cursor = "grabbing";
            };
            g.appendChild(circle);

            var text = document.createElementNS("http://www.w3.org/2000/svg", "text");
            text.setAttribute("x", c.u + 14);
            text.setAttribute("y", c.v - 14);
            text.setAttribute("class", "pnp-handle-label");
            text.appendChild(document.createTextNode("P" + i));
            g.appendChild(text);
        }
    };

    /** 更新弹窗 SVG 上指定手柄的位置（按 data-idx 查找），u/v 为图片像素坐标 */
    this._updateHandle = function (idx, u, v) {
        var g = this.handleGroup.querySelector('[data-idx="' + idx + '"]');
        if (!g) return;
        g.querySelector("circle").setAttribute("cx", u);
        g.querySelector("circle").setAttribute("cy", v);
        g.querySelector("text").setAttribute("x", u + 14);
        g.querySelector("text").setAttribute("y", v - 14);
    };

    /** 在弹窗 SVG 上绘制重投影点（十字标记），用于对比用户拖拽位置 */
    this._renderReprojected = function () {
        // 移除旧的重投影标记
        var old = this.svg.querySelector("#pnp-reprojected-group");
        if (old) old.remove();

        if (!this.points_3d || !this.result || !this.result.success) return;

        var calib = this._getActiveCalib();
        if (!calib) return;

        // 将 3D 点转为齐次坐标并投影
        var homo = [];
        for (var i = 0; i < 4; i++) {
            homo.push(this.points_3d[i][0], this.points_3d[i][1], this.points_3d[i][2], 1);
        }
        var proj = points3d_homo_to_image2d(homo, calib);
        if (!proj) return;

        var g = document.createElementNS("http://www.w3.org/2000/svg", "g");
        g.setAttribute("id", "pnp-reprojected-group");
        var colors = ["#ff4444", "#44ff44", "#4444ff", "#ffff44"];

        for (var i = 0; i < 4; i++) {
            // proj 是图片像素坐标，viewBox = 图片尺寸，直接使用
            var u = proj[i * 2];
            var v = proj[i * 2 + 1];
            // 十字标记
            var line1 = document.createElementNS("http://www.w3.org/2000/svg", "line");
            line1.setAttribute("x1", u - 8); line1.setAttribute("y1", v);
            line1.setAttribute("x2", u + 8); line1.setAttribute("y2", v);
            line1.setAttribute("stroke", colors[i]); line1.setAttribute("stroke-width", "2");
            g.appendChild(line1);
            var line2 = document.createElementNS("http://www.w3.org/2000/svg", "line");
            line2.setAttribute("x1", u); line2.setAttribute("y1", v - 8);
            line2.setAttribute("x2", u); line2.setAttribute("y2", v + 8);
            line2.setAttribute("stroke", colors[i]); line2.setAttribute("stroke-width", "2");
            g.appendChild(line2);

            console.log("PnPCalib: reprojected P" + i + " img=(" + u.toFixed(1) + ", " + v.toFixed(1) +
                ") vs dragged = (" + this.corners[i].u.toFixed(1) + ", " + this.corners[i].v.toFixed(1) + ")");
        }
        this.svg.appendChild(g);
    };

    // ── 弹窗 SVG 鼠标拖拽 ─────────────────────────────────────────────────

    this._onMouseMove = function (event) {
        if (this._draggingIdx < 0) return;
        // 获取图片坐标（检查是否在图片区域内）
        var imgCoords = this._getImageCoords(event);
        if (!imgCoords) return; // 拖拽到图片外部，忽略
        this.onCornerDrag(this._draggingIdx, imgCoords.u, imgCoords.v);
    };

    this._onMouseUp = function () {
        this._draggingIdx = -1;
    };

    // ── 加载相机图像 ──────────────────────────────────────────────────────

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

    this._loadImage = function () {
        try {
            var camName = this._getActiveCameraName();
            if (!camName) return;
            var img = this.data.world.cameras.getImageByName(camName);
            if (!img) return;

            // 设置 SVG viewBox 匹配图片自然尺寸
            var w = img.naturalWidth || 2048;
            var h = img.naturalHeight || 1536;
            this.svg.setAttribute("viewBox", "0 0 " + w + " " + h);
            this.imageEl.setAttribute("width", w);
            this.imageEl.setAttribute("height", h);

            // 始终使用原始图像（不解畸变），保证像素坐标和 dist_coeffs 一致
            this.imageEl.setAttribute("xlink:href", img.src);

            // 图片可能已缓存加载完成，手动触发尺寸更新
            if (img.complete) {
                var nw = img.naturalWidth || 2048;
                var nh = img.naturalHeight || 1536;
                pnp.svg.setAttribute("viewBox", "0 0 " + nw + " " + nh);
                pnp.imageEl.setAttribute("width", nw);
                pnp.imageEl.setAttribute("height", nh);
            }
        } catch (e) {
            console.warn("PnPCalib: failed to load image", e);
        }
    };

    // ── 模式切换 ──────────────────────────────────────────────────────────

    /** 进入 PnP 标定模式 */
    this.enter = function (box) {
        if (!box) {
            console.warn("PnPCalib.enter: no box selected");
            return;
        }
        if (!this._queryDom()) {
            console.error("PnPCalib.enter: popup not found in DOM");
            return;
        }
        this.active = true;
        this.corners = [null, null, null, null];
        this.points_3d = [null, null, null, null];
        this.result = null;
        this._draggingIdx = -1;

        // 3D 角点选择模式：在 box 的 8 个角上放球体标记，
        // 用户在 3D 视图中依次点击 4 个角作为 P0-P3，
        // 2D 位置由当前外参自动投影（可拖拽微调）
        try {
            this._create3DCornerMarkers(box);
        } catch (e) {
            console.warn("PnPCalib: 3D corner markers failed:", e);
        }

        this._loadImage();
        this.wrapper.style.display = "block";
        this._updateUI();
        this._renderHandles();

        // 主相机图像上也同步显示手柄（仅当有有效角点时）
        var hasAny = this.corners.some(function (c) { return c !== null; });
        if (hasAny) {
            this.editor.imageContextManager.renderPnpHandles(this.corners);
        }
        var self = this;
        this.editor.imageContextManager.setPnpOnDrag(function (idx, u, v) {
            self.onCornerDrag(idx, u, v);
        });

        // Z-view 角点标签
        this._createZViewLabels();

        console.log("PnPCalib: entered, points_3d=", this.points_3d);
    };

    /** 开始手动选择 3D 点模式 */
    this.startManual3DSelection = function () {
        if (!this.active) return;

        // 重置 3D 点
        this.points_3d = [null, null, null, null];
        this._manualPickIndex = 0;

        // 移除现有的 3D 角点标记
        this._remove3DCornerMarkers();

        // 更新按钮状态
        this.manual3dBtn.textContent = "选择 P0...";
        this.manual3dBtn.disabled = true;

        // 设置测量工具的回调
        var pnp = this;
        this.editor.measureTool.onPick = function (point) {
            pnp._onManual3DPointPicked(point);
        };

        // 进入测量模式
        this.editor.measureTool.start();

        this._updateUI();
        console.log("PnPCalib: started manual 3D selection");
    };

    /** 手动选择了一个 3D 点 */
    this._onManual3DPointPicked = function (point) {
        if (this._manualPickIndex >= 4) return;

        var idx = this._manualPickIndex;
        this.points_3d[idx] = [point.x, point.y, point.z];

        console.log("PnPCalib: manual 3D point P" + idx + "=" +
            point.x.toFixed(3) + "," + point.y.toFixed(3) + "," + point.z.toFixed(3));

        this._manualPickIndex++;

        // 更新按钮文本
        if (this._manualPickIndex < 4) {
            this.manual3dBtn.textContent = "选择 P" + this._manualPickIndex + "...";
        } else {
            // 4个点都选完了
            this.editor.measureTool.stop();
            this.editor.measureTool.onPick = null;
            this.manual3dBtn.textContent = "手动选点";
            this.manual3dBtn.disabled = false;

            // 如果2D点也已经设置好了，自动求解
            var allCornersPlaced = this.corners.every(function (c) { return c !== null; });
            if (allCornersPlaced) {
                this.solve();
            }
        }

        this._updateUI();
    };

    /** 退出 PnP 标定模式 */
    this.exit = function () {
        this.active = false;
        this.corners = [null, null, null, null];
        this.points_3d = null;
        this.result = null;
        this._draggingIdx = -1;
        this._manualPickIndex = 0;

        // 停止测量模式并清理回调
        if (this.editor.measureTool) {
            this.editor.measureTool.stop();
            this.editor.measureTool.onPick = null;
        }

        if (this.wrapper) {
            this.wrapper.style.display = "none";
        }

        // 恢复 box editor 侧边栏
        var boxEditorWrapper = document.getElementById("main-box-editor-wrapper");
        if (boxEditorWrapper && this.editor.selected_box) {
            boxEditorWrapper.style.display = "";
        }
        if (this.editor.imageContextManager) {
            this.editor.imageContextManager.clearPnpHandles();
            this.editor.imageContextManager.setPnpOnDrag(null);
        }
        this._removeZViewLabels();
        this._remove3DCornerMarkers();

        // 恢复原始外参（用户未保存就关闭时）
        if (this._savedExtrinsic) {
            try {
                var sceneMeta = this.data.meta[this.data.world.frameInfo.scene];
                var camName = this._getActiveCameraName();
                sceneMeta.calib.camera[camName].extrinsic = this._savedExtrinsic;
                this.editor.imageContextManager.render_2d_image();
            } catch(e) { /* ignore */ }
            this._savedExtrinsic = null;
        }

        // 隐藏 top-4 结果
        var top4 = this.wrapper ? this.wrapper.querySelector("#pnp-top4-results") : null;
        if (top4) top4.style.display = "none";

        // 清除重投影标记
        var reproj = this.svg ? this.svg.querySelector("#pnp-reprojected-group") : null;
        if (reproj) reproj.remove();
        var allBoxes = this.svg ? this.svg.querySelector("#pnp-all-boxes-group") : null;
        if (allBoxes) allBoxes.remove();
    };

    // ── 角点拖拽回调 ──────────────────────────────────────────────────────

    /**
     * 用户拖拽 handle 后更新角点位置
     * @param {number} idx  0-3
     * @param {number} u    图像 x 坐标
     * @param {number} v    图像 y 坐标
     */
    this.onCornerDrag = function (idx, u, v) {
        if (!this.active) return;
        if (idx < 0 || idx > 3) return;
        this.corners[idx] = { u: u, v: v };
        this._updateUI();
        this._updateHandle(idx, u, v);
        this.editor.imageContextManager.updatePnpHandle(idx, u, v);
    };

    // ── 按钮操作 ──────────────────────────────────────────────────────────

    /** 检查 4 个 3D 点是否过于共线/共面退化 */
    this._check3DDistribution = function () {
        var pts = this.points_3d;
        if (!pts || pts.some(function(p){ return p === null; })) return null;

        // 计算 3 个方向的跨度
        var mins = [Infinity, Infinity, Infinity];
        var maxs = [-Infinity, -Infinity, -Infinity];
        for (var i = 0; i < 4; i++) {
            for (var j = 0; j < 3; j++) {
                if (pts[i][j] < mins[j]) mins[j] = pts[i][j];
                if (pts[i][j] > maxs[j]) maxs[j] = pts[i][j];
            }
        }
        var ranges = [maxs[0]-mins[0], maxs[1]-mins[1], maxs[2]-mins[2]];
        ranges.sort(function(a,b){ return a-b; });
        // ranges[0] = 最小跨度, ranges[2] = 最大跨度
        var ratio = ranges[2] > 0.001 ? ranges[0] / ranges[2] : 0;

        console.log("PnPCalib: 3D distribution ranges=[" +
            ranges[0].toFixed(3) + ", " + ranges[1].toFixed(3) + ", " + ranges[2].toFixed(3) +
            "] min/max ratio=" + ratio.toFixed(3));

        // 如果最小跨度与最大跨度之比 < 0.15，认为过于共线
        if (ratio < 0.15) {
            return "3D点分布过于共线（最小跨度/最大跨度=" + ratio.toFixed(2) +
                "），建议从不同面选角点以获得更好的分布";
        }
        return null;
    };

    /** 调用后端 solvePnP */
    this.solve = async function () {
        if (!this.active) return;
        if (this.corners.some(function (c) { return c === null; })) {
            console.warn("PnPCalib: not all corners placed");
            return;
        }
        if (this.points_3d.some(function (p) { return p === null; })) {
            console.warn("PnPCalib: not all 3D points set");
            this.errorEl.textContent = "请先放置4个角点";
            return;
        }

        // 检查 3D 点分布
        var distWarning = this._check3DDistribution();
        if (distWarning) {
            console.warn("PnPCalib: " + distWarning);
            this.errorEl.textContent = "⚠ " + distWarning;
        }

        var sceneMeta = this.data.meta[this.data.world.frameInfo.scene];
        var camName = this._getActiveCameraName();

        try {
            var response = await fetch("/solve_pnp", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    scene: this.data.world.frameInfo.scene,
                    camera: camName,
                    points_3d: this.points_3d,
                    points_2d: this.corners.map(function (c) { return [c.u, c.v]; }),
                }),
            });
            this.result = await response.json();
        } catch (e) {
            this.errorEl.textContent = "网络错误";
            console.error("PnPCalib: network error", e);
            return;
        }

        if (this.result.success) {
            var calibData = sceneMeta.calib.camera[camName];
            calibData.extrinsic = this.result.extrinsic;

            this.errorEl.textContent = this.result.reprojection_error.toFixed(2);
            this.saveBtn.disabled = false;

            this.editor.imageContextManager.render_2d_image();

            // 在弹窗图像上显示重投影点（小十字），用于对比拖拽位置
            this._renderReprojected();

            // 重投影所有检测框，便于验证外参效果
            this._renderAllBoxes();

            // Debug: 输出详细的标定信息
            console.log("=== PnP Solve 详细日志 ===");
            console.log("3D 点:", JSON.stringify(this.points_3d));
            var userPts = this.corners.map(function(c){return [c.u,c.v];});
            console.log("2D 点(用户拖拽):", JSON.stringify(userPts));
            console.log("求解外参:", JSON.stringify(this.result.extrinsic));
            console.log("rvec:", this.result.rvec, "tvec:", this.result.tvec);
            if (this.result.reprojected_points) {
                console.log("--- 逐点重投影对比 ---");
                for (var i = 0; i < 4; i++) {
                    var rp = this.result.reprojected_points[i];
                    var up = userPts[i];
                    var err = this.result.per_point_errors ? this.result.per_point_errors[i] : "?";
                    console.log("  P" + i + ": 用户=(" + up[0].toFixed(1) + "," + up[1].toFixed(1) +
                        ") 重投影=(" + rp[0].toFixed(1) + "," + rp[1].toFixed(1) +
                        ") 误差=" + (typeof err === "number" ? err.toFixed(2) : err) + "px");
                }
            }
            console.log("平均重投影误差: " + this.result.reprojection_error.toFixed(2) + "px");
            console.log("========================");
        } else {
            this.errorEl.textContent =
                "求解失败: " + (this.result.error || "未知错误");
            console.error("PnPCalib: solve failed", this.result);
        }
    };

    /** 持久化当前外参到磁盘 */
    this.save = async function () {
        if (!this.result || !this.result.success) return;

        var sceneMeta = this.data.meta[this.data.world.frameInfo.scene];
        var camName = this._getActiveCameraName();
        var calibData = sceneMeta.calib.camera[camName];

        try {
            var response = await fetch("/calib_save", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    scene: this.data.world.frameInfo.scene,
                    camera: camName,
                    extrinsic: calibData.extrinsic,
                }),
            });
            var saveResult = await response.json();
        } catch (e) {
            this.errorEl.textContent = "保存失败: 网络错误";
            console.error("PnPCalib: save network error", e);
            return;
        }

        if (saveResult.success) {
            console.log("PnPCalib: saved successfully");
            this.exit();
        } else {
            this.errorEl.textContent =
                "保存失败: " + (saveResult.error || "未知错误");
            console.error("PnPCalib: save failed", saveResult);
        }
    };

    /** 穷举 24 种点序排列（后端一次性完成），展示全部结果 */
    this.bruteForce = async function () {
        if (!this.active) return;
        if (this.corners.some(function (c) { return c === null; })) {
            console.warn("PnPCalib: not all corners placed");
            return;
        }

        var scene = this.data.world.frameInfo.scene;
        var camName = this._getActiveCameraName();

        this.errorEl.textContent = "测试中...";
        this.bruteBtn.disabled = true;
        console.log("=== PnP 穷举测试 (后端 24 种点序) ===");

        // 保存原始外参用于恢复
        var sceneMeta = this.data.meta[scene];
        if (!this._savedExtrinsic) {
            this._savedExtrinsic = sceneMeta.calib.camera[camName].extrinsic;
        }
        // 保存原始 2D 角点顺序
        if (!this._originalCorners2d) {
            this._originalCorners2d = this.corners.slice();
        }

        try {
            var resp = await fetch("/brute_force_pnp", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    scene: scene,
                    camera: camName,
                    points_3d: this.points_3d,
                    points_2d: this.corners.map(function (c) { return [c.u, c.v]; }),
                }),
            });
            var data = await resp.json();
        } catch (e) {
            this.errorEl.textContent = "网络错误";
            this.bruteBtn.disabled = false;
            console.error("PnPCalib: brute force network error", e);
            return;
        }

        if (!data.success) {
            this.errorEl.textContent = "失败: " + (data.error || "未知");
            this.bruteBtn.disabled = false;
            return;
        }

        // 按误差排序，展示全部结果
        var results = data.results;
        results.sort(function (a, b) { return a.error - b.error; });
        this._bruteForceResults = results;

        for (var i = 0; i < results.length; i++) {
            console.log("  #" + (i+1) + " [" + results[i].perm.join(",") + "] error=" + results[i].error.toFixed(2) + " px");
        }

        // 渲染全部结果列表
        this._renderBruteForceList(results, scene, camName);

        // 默认预览最优结果
        this._previewBruteResult(results[0], scene, camName);

        this.bruteBtn.disabled = false;
    };

    /** 渲染穷举结果列表（全部 24 种） */
    this._renderBruteForceList = function (results, scene, camName) {
        var container = this.wrapper.querySelector("#pnp-top4-results");
        if (!container) return;
        var list = container.querySelector(".pnp-top4-list");
        list.innerHTML = "";

        var pnp = this;
        for (var i = 0; i < results.length; i++) {
            (function (idx) {
                var r = results[idx];
                var item = document.createElement("div");
                item.className = "pnp-top4-item" + (idx === 0 ? " selected" : "");
                item.innerHTML =
                    '<span class="pnp-top4-rank">#' + (idx + 1) + '</span>' +
                    '<span class="pnp-top4-perm">[' + r.perm.join(",") + ']</span>' +
                    '<span class="pnp-top4-error">' + r.error.toFixed(2) + ' px</span>' +
                    '<button class="pnp-top4-save-btn" title="保存此外参">保存</button>';

                // 点击条目 → 实时预览（更新所有重投影）
                item.addEventListener("click", function (e) {
                    if (e.target.classList.contains("pnp-top4-save-btn")) return;
                    list.querySelectorAll(".pnp-top4-item").forEach(function (el) {
                        el.classList.remove("selected");
                    });
                    item.classList.add("selected");
                    pnp._previewBruteResult(results[idx], scene, camName);
                });

                // 保存按钮 → 直接保存此外参到文件
                var saveBtn = item.querySelector(".pnp-top4-save-btn");
                saveBtn.addEventListener("click", function (e) {
                    e.stopPropagation();
                    pnp._saveBruteResult(results[idx], scene, camName);
                });

                list.appendChild(item);
            })(i);
        }
        container.style.display = "block";
    };

    /** 预览某个穷举结果：应用外参 + 重排角点 + 更新所有重投影 */
    this._previewBruteResult = function (result, scene, camName) {
        // 按 result.perm 重排 2D 角点
        var orig2d = this._originalCorners2d;
        for (var i = 0; i < 4; i++) {
            this.corners[i] = orig2d[result.perm[i]];
        }

        // 应用外参
        var sceneMeta = this.data.meta[scene];
        sceneMeta.calib.camera[camName].extrinsic = result.extrinsic;

        this.errorEl.textContent = result.error.toFixed(2);
        this.result = { success: true, extrinsic: result.extrinsic, reprojection_error: result.error };
        this.saveBtn.disabled = false;

        // 更新侧边栏 UI
        this._updateUI();
        this._renderHandles();
        this._renderReprojected();
        this._renderAllBoxes();

        // 更新主图像面板（box editor 重投影 + PnP 手柄）
        this.editor.imageContextManager.renderPnpHandles(this.corners);
        this.editor.imageContextManager.render_2d_image();

        // 更新 box editor 焦点 canvas
        var selBox = this.editor.selected_box;
        if (selBox && selBox.boxEditor) {
            selBox.boxEditor.focusImageContext.updateFocusedImageContext(selBox);
        }
    };

    /** 保存穷举结果中的某个外参到文件 */
    this._saveBruteResult = async function (result, scene, camName) {
        if (!result || !result.extrinsic) {
            this.errorEl.textContent = "保存失败: 无有效外参";
            return;
        }

        try {
            var response = await fetch("/calib_save", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    scene: scene,
                    camera: camName,
                    extrinsic: result.extrinsic,
                }),
            });
            var saveResult = await response.json();
        } catch (e) {
            this.errorEl.textContent = "保存失败: 网络错误";
            console.error("PnPCalib: save network error", e);
            return;
        }

        if (saveResult.success) {
            this.errorEl.textContent = "已保存外参 (perm=[" + result.perm.join(",") + "], err=" + result.error.toFixed(2) + "px)";
            console.log("PnPCalib: brute result saved", result.perm);
        } else {
            this.errorEl.textContent = "保存失败: " + (saveResult.error || "未知错误");
        }
    };

    /** 在侧边栏 SVG 上绘制所有检测框的重投影（用于验证外参效果） */
    this._renderAllBoxes = function () {
        var old = this.svg.querySelector("#pnp-all-boxes-group");
        if (old) old.remove();

        var calib = this._getActiveCalib();
        if (!calib) return;

        var boxes = this.data.world.annotation.boxes;
        if (!boxes || boxes.length === 0) return;

        var selectedBox = this.editor.selected_box;

        var g = document.createElementNS("http://www.w3.org/2000/svg", "g");
        g.setAttribute("id", "pnp-all-boxes-group");

        for (var i = 0; i < boxes.length; i++) {
            var box = boxes[i];
            var pts = box_to_2d_points(box, calib);
            if (!pts) continue;

            var isSelected = (box === selectedBox);
            var color = isSelected ? "#00ff00" : "#ff8800";
            var opacity = isSelected ? "0.9" : "0.4";
            var strokeWidth = isSelected ? "2" : "1";

            // 画前面 (+X 面): indices 0-3 → pts[0..7]
            var frontPts = pts.slice(0, 8);
            var polygon = document.createElementNS("http://www.w3.org/2000/svg", "polygon");
            polygon.setAttribute("points", frontPts.join(","));
            polygon.setAttribute("fill", color);
            polygon.setAttribute("fill-opacity", opacity);
            polygon.setAttribute("stroke", color);
            polygon.setAttribute("stroke-width", strokeWidth);
            g.appendChild(polygon);

            // 画后面 (-X 面): indices 4-7 → pts[8..15]
            var rearPts = pts.slice(8, 16);
            var polygon2 = document.createElementNS("http://www.w3.org/2000/svg", "polygon");
            polygon2.setAttribute("points", rearPts.join(","));
            polygon2.setAttribute("fill", "none");
            polygon2.setAttribute("stroke", color);
            polygon2.setAttribute("stroke-width", strokeWidth);
            polygon2.setAttribute("stroke-dasharray", "4,4");
            g.appendChild(polygon2);

            // 连接前后面对应边
            for (var j = 0; j < 4; j++) {
                var line = document.createElementNS("http://www.w3.org/2000/svg", "line");
                line.setAttribute("x1", pts[j * 2]); line.setAttribute("y1", pts[j * 2 + 1]);
                line.setAttribute("x2", pts[(j + 4) * 2]); line.setAttribute("y2", pts[(j + 4) * 2 + 1]);
                line.setAttribute("stroke", color);
                line.setAttribute("stroke-width", "1");
                line.setAttribute("stroke-opacity", opacity);
                g.appendChild(line);
            }

            // 标注 track id
            if (box.obj_track_id) {
                var cx = (pts[0] + pts[8]) / 2;
                var cy = (pts[1] + pts[9]) / 2;
                var label = document.createElementNS("http://www.w3.org/2000/svg", "text");
                label.setAttribute("x", cx);
                label.setAttribute("y", cy - 6);
                label.setAttribute("fill", color);
                label.setAttribute("font-size", "11");
                label.setAttribute("font-family", "monospace");
                label.setAttribute("text-anchor", "middle");
                label.appendChild(document.createTextNode(box.obj_type + " " + box.obj_track_id));
                g.appendChild(label);
            }
        }

        this.svg.appendChild(g);
    };

    /** 重置所有角点（重新选择 3D 角点） */
    this.reset = function () {
        if (!this.active) return;
        if (this.errorEl) this.errorEl.textContent = "--";
        if (this.saveBtn) this.saveBtn.disabled = true;
        this.corners = [null, null, null, null];
        this.points_3d = [null, null, null, null];
        this.result = null;
        this._originalCorners2d = null;
        this._bruteForceResults = null;
        this._manualPickIndex = 0;

        // 停止测量模式并清理回调
        if (this.editor.measureTool) {
            this.editor.measureTool.stop();
            this.editor.measureTool.onPick = null;
        }

        // 恢复手动选点按钮状态
        if (this.manual3dBtn) {
            this.manual3dBtn.textContent = "手动选点";
            this.manual3dBtn.disabled = false;
        }

        // 隐藏 top-4 结果
        var top4 = this.wrapper ? this.wrapper.querySelector("#pnp-top4-results") : null;
        if (top4) top4.style.display = "none";

        // 重新创建 3D 角点标记（重置选择状态）
        var box = this.editor.selected_box;
        if (box) {
            try { this._create3DCornerMarkers(box); } catch (e) { /* ignore */ }
        }

        this._updateUI();
        this._renderHandles();
        this.editor.imageContextManager.clearPnpHandles();
        this.editor.imageContextManager.render_2d_image();

        // 清除重投影标记
        var reproj = this.svg ? this.svg.querySelector("#pnp-reprojected-group") : null;
        if (reproj) reproj.remove();
        var allBoxes = this.svg ? this.svg.querySelector("#pnp-all-boxes-group") : null;
        if (allBoxes) allBoxes.remove();
    };

    // ── 内部工具 ──────────────────────────────────────────────────────────

    this._getActiveCalib = function () {
        try {
            var sceneMeta = this.data.meta[this.data.world.frameInfo.scene];
            var camName = this._getActiveCameraName();
            return sceneMeta.calib.camera[camName];
        } catch (e) {
            return null;
        }
    };

    this._updateUI = function () {
        var allPlaced = true;
        for (let i = 0; i < 4; i++) {
            var c = this.corners[i];
            if (c) {
                this.cornerEls[i].textContent =
                    "(" + c.u.toFixed(0) + ", " + c.v.toFixed(0) + ")";
            } else {
                this.cornerEls[i].textContent = "--, --";
                allPlaced = false;
            }
        }
        this.solveBtn.disabled = !allPlaced;
    };

    // ── 3D 角点选择 ──────────────────────────────────────────────────────────

    /** 在 3D 场景中创建标记：4 个边中点（默认）+ 8 个角点，用户点击选择 P0-P3 */
    this._create3DCornerMarkers = function (box) {
        this._remove3DCornerMarkers();
        this._selectedCornerIndices = [];

        var box3d = psr_to_xyz(box.position, box.scale, box.rotation);
        // 8 个角点的 3D 坐标
        var corners = [];
        for (var i = 0; i < 8; i++) {
            corners.push([box3d[i * 4], box3d[i * 4 + 1], box3d[i * 4 + 2]]);
        }

        // 4 个边中点 = (前面[i] + 后面[i]) / 2，都在 x=0 平面上
        var midpoints = [];
        for (var i = 0; i < 4; i++) {
            midpoints.push([
                (corners[i][0] + corners[i + 4][0]) / 2,
                (corners[i][1] + corners[i + 4][1]) / 2,
                (corners[i][2] + corners[i + 4][2]) / 2
            ]);
        }

        // 统一存储：idx 0-3 边中点，4-7 前角点，8-11 后角点
        this._allMarkers3d = midpoints.concat(corners);

        // 球体半径：取 box 最大维度的 5%，至少 0.1
        var maxScale = Math.max(box.scale.x, box.scale.y, box.scale.z);
        var sphereRadius = Math.max(maxScale * 0.05, 0.1);
        var midRadius = sphereRadius * 0.75;

        var group = new THREE.Group();
        group.name = "pnp-corner-markers";

        // 颜色：中点(青/洋红/橙/紫), 前角(红/绿/蓝/黄), 后角(浅色)
        var midColors = [0x00cccc, 0xcc00cc, 0xcc8800, 0x8800cc];
        var frontColors = [0xff4444, 0x44ff44, 0x4444ff, 0xffff44];
        var backColors = [0xff8888, 0x88ff88, 0x8888ff, 0xffff88];
        var allColors = midColors.concat(frontColors).concat(backColors);
        var labels = ["M0", "M1", "M2", "M3", "C0", "C1", "C2", "C3", "C4", "C5", "C6", "C7"];

        // 保存 mesh 引用，方便选中时更新样式
        this._markerMeshes = [];

        for (var i = 0; i < 12; i++) {
            var isMid = i < 4;
            var geo = isMid
                ? new THREE.OctahedronGeometry(midRadius)
                : new THREE.SphereGeometry(sphereRadius);
            var mat = new THREE.MeshBasicMaterial({
                color: allColors[i],
                depthTest: false
            });
            var mesh = new THREE.Mesh(geo, mat);
            var pt = this._allMarkers3d[i];
            mesh.position.set(pt[0], pt[1], pt[2]);
            mesh.userData.pnpCornerIdx = i;
            mesh.renderOrder = 999;
            group.add(mesh);
            this._markerMeshes.push(mesh);

            // 标签
            var canvas = document.createElement("canvas");
            canvas.width = 64; canvas.height = 32;
            var ctx = canvas.getContext("2d");
            ctx.fillStyle = "#" + allColors[i].toString(16).padStart(6, "0");
            ctx.font = "bold 22px monospace";
            ctx.textAlign = "center";
            ctx.fillText(labels[i], 32, 24);
            var tex = new THREE.CanvasTexture(canvas);
            var spriteMat = new THREE.SpriteMaterial({ map: tex, depthTest: false });
            var sprite = new THREE.Sprite(spriteMat);
            sprite.position.set(pt[0], pt[1] + sphereRadius * 1.5, pt[2]);
            sprite.scale.set(sphereRadius * 2, sphereRadius, 1);
            sprite.renderOrder = 998;
            group.add(sprite);
        }

        this.editor.scene.add(group);
        this._cornerMarkerGroup = group;

        // 用 pointerdown 而非 click，避免 OrbitControls setPointerCapture 吞掉事件
        var pnp = this;
        var renderer = this.editor.renderer;
        var camera = this.editor.viewManager.mainView.camera;
        var raycaster = new THREE.Raycaster();
        var mouse = new THREE.Vector2();

        this._cornerClickHandler = function (event) {
            if (!pnp.active) return;
            var rect = renderer.domElement.getBoundingClientRect();
            mouse.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
            mouse.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;
            raycaster.setFromCamera(mouse, camera);
            // 只检测 Mesh（跳过 Sprite 标签）
            var meshes = group.children.filter(function (c) { return c.isMesh; });
            var intersects = raycaster.intersectObjects(meshes);
            if (intersects.length > 0) {
                var idx = intersects[0].object.userData.pnpCornerIdx;
                pnp._selectCorner(idx);
                event.stopPropagation();
            }
        };
        renderer.domElement.addEventListener("pointerdown", this._cornerClickHandler);

        console.log("PnPCalib: markers created — 4 midpoints + 8 corners");
        console.log("  midpoints (x≈0):", JSON.stringify(midpoints));
    };

    /** 用户点击了一个 3D 标记，分配为下一个 P 点 */
    this._selectCorner = function (cornerIdx) {
        // 检查是否已选过
        if (this._selectedCornerIndices.indexOf(cornerIdx) >= 0) {
            console.log("PnPCalib: marker " + cornerIdx + " already selected");
            return;
        }

        var pIdx = this._selectedCornerIndices.length;
        if (pIdx >= 4) return;

        this._selectedCornerIndices.push(cornerIdx);
        var pt = this._allMarkers3d[cornerIdx];
        this.points_3d[pIdx] = pt;

        // 更新标记颜色为对应 P 点颜色
        var pColors = [0xff0000, 0x00ff00, 0x0000ff, 0xffff00];
        var marker = this._markerMeshes[cornerIdx];
        if (marker) {
            marker.material.color.setHex(pColors[pIdx]);
            marker.scale.multiplyScalar(1.3); // 选中的稍大一些
        }

        // 尝试投影到 2D；若失败则给默认位置，确保手柄始终可见
        var calib = this._getActiveCalib();
        var placed = false;
        if (calib && calib.extrinsic) {
            var homo = [pt[0], pt[1], pt[2], 1];
            var proj = points3d_homo_to_image2d(homo, calib);
            console.log("PnPCalib: P" + pIdx + " proj=", proj, "calib.extrinsic=", calib.extrinsic ? "ok" : "MISSING");
            if (proj && isFinite(proj[0]) && isFinite(proj[1])) {
                // proj 是图片像素坐标，viewBox = 图片尺寸，直接使用
                var vb = this.svg.viewBox.baseVal;
                var maxU = vb.width > 0 ? vb.width - 20 : 2028;
                var maxV = vb.height > 0 ? vb.height - 20 : 1516;
                this.corners[pIdx] = {
                    u: Math.max(20, Math.min(maxU, proj[0])),
                    v: Math.max(20, Math.min(maxV, proj[1]))
                };
                placed = true;
            }
        } else {
            console.warn("PnPCalib: no calib or extrinsic, calib=", calib);
        }
        if (!placed) {
            // 无有效外参或投影失败，给一个散布的默认位置
            var defaults = [
                { u: 400, v: 400 }, { u: 1600, v: 400 },
                { u: 1600, v: 1100 }, { u: 400, v: 1100 }
            ];
            this.corners[pIdx] = defaults[pIdx];
            console.warn("PnPCalib: projection failed for P" + pIdx + ", using default pos=", defaults[pIdx]);
        }

        this._updateUI();
        this._renderHandles();
        this.editor.imageContextManager.renderPnpHandles(this.corners);
        this._updateZViewLabels();

        console.log("PnPCalib: selected corner " + cornerIdx + " as P" + pIdx +
            " 3D=(" + pt[0].toFixed(3) + "," + pt[1].toFixed(3) + "," + pt[2].toFixed(3) + ")" +
            " 2D=(" + this.corners[pIdx].u.toFixed(0) + "," + this.corners[pIdx].v.toFixed(0) + ")");

        if (pIdx === 3) {
            console.log("PnPCalib: all 4 corners selected, ready to solve");
            for (var k = 0; k < 4; k++) {
                console.log("  P" + k + ": cornerIdx=" + this._selectedCornerIndices[k] +
                    " 3D=(" + this.points_3d[k][0].toFixed(3) + "," + this.points_3d[k][1].toFixed(3) + "," + this.points_3d[k][2].toFixed(3) + ")" +
                    " 2D=(" + this.corners[k].u.toFixed(0) + "," + this.corners[k].v.toFixed(0) + ")");
            }
        }
    };

    /** 移除 3D 角点标记 */
    this._remove3DCornerMarkers = function () {
        if (this._cornerClickHandler && this.editor && this.editor.renderer) {
            this.editor.renderer.domElement.removeEventListener("pointerdown", this._cornerClickHandler);
            this._cornerClickHandler = null;
        }
        if (this._cornerMarkerGroup && this.editor && this.editor.scene) {
            this.editor.scene.remove(this._cornerMarkerGroup);
            // 释放几何体和材质
            this._cornerMarkerGroup.traverse(function (child) {
                if (child.geometry) child.geometry.dispose();
                if (child.material) child.material.dispose();
            });
            this._cornerMarkerGroup = null;
        }
        this._markerMeshes = [];
        this._selectedCornerIndices = [];
        this._allMarkers3d = null;
    };

    // ── Z-view 角点标签 ──────────────────────────────────────────────────────

    /** 在 Z-view SVG 上创建 P0-P3 角点标签 */
    this._createZViewLabels = function () {
        var zView = document.getElementById("z-view-manipulator");
        if (!zView) return;
        var svg = zView.querySelector("#view-svg");
        if (!svg) return;

        // 移除旧标签
        this._removeZViewLabels();

        this._zViewLabels = [];
        var colors = ["#ff4444", "#44ff44", "#4444ff", "#ffff44"];
        var texts = ["P0", "P1", "P2", "P3"];

        for (var i = 0; i < 4; i++) {
            var t = document.createElementNS("http://www.w3.org/2000/svg", "text");
            t.setAttribute("fill", colors[i]);
            t.setAttribute("font-size", "12");
            t.setAttribute("font-weight", "bold");
            t.setAttribute("font-family", "monospace");
            t.appendChild(document.createTextNode(texts[i]));
            svg.appendChild(t);
            this._zViewLabels.push(t);
        }

        this._updateZViewLabels();

        // 监听 SVG 线条位置变化以同步标签位置
        var pnp = this;
        var lines = svg.querySelectorAll(".svg-line");
        this._zViewObserver = new MutationObserver(function () {
            pnp._updateZViewLabels();
        });
        lines.forEach(function (line) {
            pnp._zViewObserver.observe(line, { attributes: true, attributeFilter: ["x1", "y1", "x2", "y2"] });
        });
    };

    /** 更新 Z-view 标签位置（将实际 3D 角点投影到 Z-view SVG 坐标系） */
    this._updateZViewLabels = function () {
        if (!this._zViewLabels) return;
        if (!this.points_3d || this.points_3d.some(function(p){ return p === null; })) return;

        var zView = document.getElementById("z-view-manipulator");
        if (!zView) return;
        var svg = zView.querySelector("#view-svg");
        if (!svg) return;

        var left   = svg.querySelector("#line-left");
        var top    = svg.querySelector("#line-top");
        var right  = svg.querySelector("#line-right");
        var bottom = svg.querySelector("#line-bottom");
        if (!left || !top || !right || !bottom) return;

        // 矩形边界 (SVG 像素坐标)
        var l = parseFloat(left.getAttribute("x1"));
        var r = parseFloat(right.getAttribute("x1"));
        var t = parseFloat(top.getAttribute("y1"));
        var b = parseFloat(bottom.getAttribute("y1"));

        // 矩形中心和半尺寸
        var cx = (l + r) / 2, cy = (t + b) / 2;
        var hw = (r - l) / 2, hh = (b - t) / 2;

        // 获取 box 旋转角 (Z-view 旋转 = -rotation.z)
        var box = this.editor.selected_box;
        var heading = box ? (box.rotation.z || 0) : 0;
        var cosR = Math.cos(-heading), sinR = Math.sin(-heading);

        // 3D→Z-view SVG 投影:
        //   SVG 右 = -3D Y,  SVG 下 = -3D X
        //   矩形宽 = scale.y * 1.5,  矩形高 = scale.x * 1.5
        //   3D 坐标需绕 box 中心旋转 heading 角后映射

        var boxCenter = box ? box.position : { x: 0, y: 0, z: 0 };
        var scaleX = box ? box.scale.x : 1;
        var scaleY = box ? box.scale.y : 1;

        for (var i = 0; i < 4; i++) {
            var pt = this.points_3d[i];
            if (!pt) continue;

            // 相对于 box 中心的 3D 偏移
            var dx = pt[0] - boxCenter.x;
            var dy = pt[1] - boxCenter.y;

            // 旋转到 box 局部坐标系
            var lx = dx * cosR - dy * sinR;
            var ly = dx * sinR + dy * cosR;

            // 局部坐标 → SVG 坐标 (无旋转)
            // SVG x = -ly / (scaleY * 1.5) * 2 * hw + cx
            // SVG y = -lx / (scaleX * 1.5) * 2 * hh + cy
            var sx = cx - (ly / scaleY) * (2 * hw / 1.5);
            var sy = cy - (lx / scaleX) * (2 * hh / 1.5);

            // 标签偏移避免遮挡
            var offX = 4, offY = -4;
            if (sx > cx) offX = -20; else offX = 4;
            if (sy > cy) offY = 14; else offY = -4;

            this._zViewLabels[i].setAttribute("x", sx + offX);
            this._zViewLabels[i].setAttribute("y", sy + offY);
        }
    };

    /** 移除 Z-view 角点标签 */
    this._removeZViewLabels = function () {
        if (this._zViewObserver) {
            this._zViewObserver.disconnect();
            this._zViewObserver = null;
        }
        if (this._zViewLabels) {
            for (var i = 0; i < this._zViewLabels.length; i++) {
                if (this._zViewLabels[i].parentNode) {
                    this._zViewLabels[i].parentNode.removeChild(this._zViewLabels[i]);
                }
            }
            this._zViewLabels = null;
        }
    };
}

export { PnPCalib };
