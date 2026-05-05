
import * as THREE from './lib/three.module.js';

class MeasureTool {
    constructor(editor) {
        this.editor = editor;
        this.measuring = false;
        this.points = [];
        this.markers = [];
        this.line = null;
        this.labelEl = null;
        this.raycaster = new THREE.Raycaster();
        // 初始化 Points threshold
        if (!this.raycaster.params.Points) {
            this.raycaster.params.Points = {};
        }
        this.raycaster.params.Points.threshold = 1.0;
    }

    start() {
        this.measuring = true;
        this.clear();
        this._updateUI();
        this._showStatus("测量模式：点击点云拾取点，ESC 退出");
        console.log("MeasureTool: started");
    }

    stop() {
        this.measuring = false;
        this._updateUI();
        this._showStatus("");
        console.log("MeasureTool: stopped");
    }

    toggle() {
        if (this.measuring) {
            this.stop();
        } else {
            this.start();
        }
    }

    addPoint(screenPos) {
        if (!this.measuring) return;

        var world = this.editor.data.world;
        if (!world) {
            this._showStatus("无世界数据");
            return;
        }

        var point = this._pickPoint(screenPos);
        if (!point) {
            this._showStatus("未拾取到点，请重试");
            return;
        }

        console.log("MeasureTool: addPoint", point, "current count:", this.points.length);

        this.points.push(point);
        this._addMarker(point);

        if (this.points.length === 1) {
            this._showPointInfo(1, point);
        } else if (this.points.length === 2) {
            try {
                this._drawLine();
                this._showDistance();
            } catch (e) {
                console.error("MeasureTool: error drawing line", e);
                this._calcDistance();
            }
        } else {
            // 超过2个点，重新开始
            this.clear();
            this.points.push(point);
            this._addMarker(point);
            this._showPointInfo(1, point);
        }

        this.editor.render();
    }

    clear() {
        var world = this.editor.data.world;
        if (!world) return;

        // 清除标记点
        for (var marker of this.markers) {
            world.scene.remove(marker);
        }
        this.markers = [];

        // 清除连线
        if (this.line) {
            world.scene.remove(this.line);
            this.line = null;
        }

        this.points = [];
        this.editor.render();
    }

    _pickPoint(screenPos) {
        var world = this.editor.data.world;

        // screenPos 已经是 NDC 坐标 (-1 到 1)
        var mouse = new THREE.Vector2(screenPos.x, screenPos.y);

        // 尝试从点云拾取
        if (world.lidar && world.lidar.points) {
            this.raycaster.setFromCamera(mouse, this.editor.viewManager.mainView.camera);

            // 使用较小的 threshold 提高精度
            this.raycaster.params.Points.threshold = 0.3;
            var intersects = this.raycaster.intersectObjects([world.lidar.points], false);

            if (intersects.length > 0) {
                // 选择距离相机最近的点（第一个交点）
                var p = intersects[0].point;
                console.log("MeasureTool: picked from cloud, distance:", intersects[0].distance.toFixed(2));
                return { x: p.x, y: p.y, z: p.z };
            }
        }

        // 备选：使用鼠标在 z=0 平面上的投影
        console.log("MeasureTool: using z=0 plane fallback");
        return this.editor.mouse.get_mouse_location_in_world();
    }

    _calcDistance() {
        if (this.points.length < 2) return;

        var p1 = this.points[0];
        var p2 = this.points[1];
        var dx = p2.x - p1.x;
        var dy = p2.y - p1.y;
        var dz = p2.z - p1.z;
        var dist = Math.sqrt(dx * dx + dy * dy + dz * dz);

        this._showStatus("距离: " + dist.toFixed(3) + " m  (dx=" + dx.toFixed(3) + " dy=" + dy.toFixed(3) + " dz=" + dz.toFixed(3) + ")");
    }

    _addMarker(p) {
        var world = this.editor.data.world;

        // 使用更小的标记点
        var geometry = new THREE.SphereGeometry(0.05, 8, 8);
        var material = new THREE.MeshBasicMaterial({ color: 0xffff00 });
        var sphere = new THREE.Mesh(geometry, material);
        sphere.position.set(p.x, p.y, p.z);

        world.scene.add(sphere);
        this.markers.push(sphere);
    }

    _drawLine() {
        var world = this.editor.data.world;
        if (!world || !world.scene) {
            console.error("MeasureTool: world or scene not available");
            return;
        }

        var p1 = this.points[0];
        var p2 = this.points[1];

        try {
            this.line = world.new_line(
                [p1.x, p1.y, p1.z],
                [p2.x, p2.y, p2.z],
                0xffff00
            );
            world.scene.add(this.line);
        } catch (e) {
            console.error("MeasureTool: error in new_line", e);
            // 备用方案：手动创建线
            this._drawLineFallback(p1, p2, world);
        }
    }

    _drawLineFallback(p1, p2, world) {
        var geometry = new THREE.BufferGeometry();
        var vertices = new Float32Array([p1.x, p1.y, p1.z, p2.x, p2.y, p2.z]);
        geometry.setAttribute('position', new THREE.BufferAttribute(vertices, 3));
        var material = new THREE.LineBasicMaterial({ color: 0xffff00, linewidth: 2 });
        this.line = new THREE.LineSegments(geometry, material);
        world.scene.add(this.line);
    }

    _showDistance() {
        var p1 = this.points[0];
        var p2 = this.points[1];
        var dx = p2.x - p1.x;
        var dy = p2.y - p1.y;
        var dz = p2.z - p1.z;
        var dist = Math.sqrt(dx * dx + dy * dy + dz * dz);

        var msg = "P1:(" + p1.x.toFixed(2) + ", " + p1.y.toFixed(2) + ", " + p1.z.toFixed(2) + ")  " +
                  "P2:(" + p2.x.toFixed(2) + ", " + p2.y.toFixed(2) + ", " + p2.z.toFixed(2) + ")  " +
                  "距离: " + dist.toFixed(3) + " m";
        this._showStatus(msg);
    }

    _showPointInfo(index, point) {
        var msg = "P" + index + ": (" + point.x.toFixed(2) + ", " + point.y.toFixed(2) + ", " + point.z.toFixed(2) + ")";
        this._showStatus(msg);
    }

    _updateUI() {
        var btn = document.getElementById("measure-button");
        if (btn) {
            btn.classList.toggle("active", this.measuring);
        }
    }

    _showStatus(msg) {
        var el = document.getElementById("measure-status");
        if (el) {
            el.textContent = msg;
            el.style.display = msg ? "block" : "none";
        }
    }
}

export { MeasureTool };
