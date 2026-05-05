
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
        this.raycaster.params.Points.threshold = 0.3;
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
        if (!world || !world.lidar || !world.lidar.points) {
            this._showStatus("无点云数据");
            return;
        }

        var point = this._pickPoint(screenPos);
        if (!point) {
            this._showStatus("未拾取到点，请重试");
            return;
        }

        this.points.push(point);
        this._addMarker(point);

        if (this.points.length === 1) {
            this._showStatus("已拾取第1个点，继续点击拾取第2个点");
        } else if (this.points.length === 2) {
            this._drawLine();
            this._showDistance();
        } else {
            this.clear();
            this.points.push(point);
            this._addMarker(point);
            this._showStatus("已拾取第1个点，继续点击拾取第2个点");
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
        var points = world.lidar.points;

        // 转换屏幕坐标到 NDC
        var mouse = new THREE.Vector2();
        mouse.x = screenPos.x;
        mouse.y = screenPos.y;

        this.raycaster.setFromCamera(mouse, this.editor.viewManager.mainView.camera);
        var intersects = this.raycaster.intersectObjects([points], false);

        if (intersects.length > 0) {
            var p = intersects[0].point;
            // 转换回 LiDAR 坐标系
            return world.scenePosToLidar(p);
        }

        // 备选：使用 z=0 平面交点
        return this.editor.mouse.get_mouse_location_in_world();
    }

    _addMarker(p) {
        var world = this.editor.data.world;
        var sceneP = world.lidarPosToScene(p);

        var geometry = new THREE.SphereGeometry(0.15, 16, 16);
        var material = new THREE.MeshBasicMaterial({ color: 0xffff00 });
        var sphere = new THREE.Mesh(geometry, material);
        sphere.position.set(sceneP.x, sceneP.y, sceneP.z);

        world.scene.add(sphere);
        this.markers.push(sphere);
    }

    _drawLine() {
        var world = this.editor.data.world;
        var p1 = world.lidarPosToScene(this.points[0]);
        var p2 = world.lidarPosToScene(this.points[1]);

        this.line = world.new_line(
            [p1.x, p1.y, p1.z],
            [p2.x, p2.y, p2.z],
            0xffff00
        );
        world.scene.add(this.line);
    }

    _showDistance() {
        var p1 = this.points[0];
        var p2 = this.points[1];
        var dx = p2.x - p1.x;
        var dy = p2.y - p1.y;
        var dz = p2.z - p1.z;
        var dist = Math.sqrt(dx * dx + dy * dy + dz * dz);

        this._showStatus("距离: " + dist.toFixed(3) + " m  (dx=" + dx.toFixed(3) + " dy=" + dy.toFixed(3) + " dz=" + dz.toFixed(3) + ")");
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
