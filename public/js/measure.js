
import * as THREE from './lib/three.module.js';
import { psr_to_xyz } from './util.js';

class MeasureTool {
    constructor(editor) {
        this.editor = editor;
        this.measuring = false;
        this.points = [];
        this.markers = [];
        this.line = null;
        this.labelEl = null;
        this.raycaster = new THREE.Raycaster();
        this.onPick = null; // 回调函数，用于 PnP 等外部模块接收选择的点
        this.onlyBoxCorners = false; // 是否只捕捉 box 角点
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

        // 清除角点高亮
        if (this._cornerHighlight) {
            this.editor.scene.remove(this._cornerHighlight);
            this._cornerHighlight = null;
        }

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

        // 如果有回调函数（PnP 模式），调用回调
        if (this.onPick) {
            this.onPick(point);
            return;
        }

        // 正常测量模式
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

        // 清除角点高亮
        if (this._cornerHighlight) {
            this.editor.scene.remove(this._cornerHighlight);
            this._cornerHighlight = null;
        }

        this.points = [];
        this.editor.render();
    }

    _pickPoint(screenPos) {
        var world = this.editor.data.world;

        // screenPos 已经是 NDC 坐标 (-1 到 1)
        var mouse = new THREE.Vector2(screenPos.x, screenPos.y);
        this.raycaster.setFromCamera(mouse, this.editor.viewManager.mainView.camera);

        // 如果设置了只捕捉 box 角点
        if (this.onlyBoxCorners) {
            var cornerPoint = this._pickBoxCorner(world);
            if (cornerPoint) {
                console.log("MeasureTool: picked box corner (onlyBoxCorners mode)");
                return cornerPoint;
            }
            return null; // 没有找到角点则返回 null
        }

        // 正常模式：优先 box 角点，其次点云
        // 1. 尝试拾取 box 角点
        var cornerPoint = this._pickBoxCorner(world);
        if (cornerPoint) {
            console.log("MeasureTool: picked box corner");
            return cornerPoint;
        }

        // 2. 尝试从点云拾取
        if (world.lidar && world.lidar.points) {
            this.raycaster.params.Points.threshold = 0.3;
            var intersects = this.raycaster.intersectObjects([world.lidar.points], false);

            if (intersects.length > 0) {
                var p = intersects[0].point;
                console.log("MeasureTool: picked from cloud, distance:", intersects[0].distance.toFixed(2));
                return { x: p.x, y: p.y, z: p.z };
            }
        }

        // 3. 备选：使用鼠标在 z=0 平面上的投影
        console.log("MeasureTool: using z=0 plane fallback");
        return this.editor.mouse.get_mouse_location_in_world();
    }

    _pickBoxCorner(world) {
        if (!world.annotation || !world.annotation.boxes) return null;

        var boxes = world.annotation.boxes;
        var minDist = Infinity;
        var closestCorner = null;
        var threshold = 2.0; // 增大角点拾取阈值（米）

        for (var i = 0; i < boxes.length; i++) {
            var box = boxes[i];
            var corners = psr_to_xyz(box.position, box.scale, box.rotation);

            // 检查 8 个角点
            for (var j = 0; j < 8; j++) {
                var cx = corners[j * 4];
                var cy = corners[j * 4 + 1];
                var cz = corners[j * 4 + 2];

                // 计算角点到射线的距离
                var dist = this._pointToRayDistance(cx, cy, cz);
                if (dist < threshold && dist < minDist) {
                    minDist = dist;
                    closestCorner = { x: cx, y: cy, z: cz, boxIdx: i, cornerIdx: j };
                }
            }
        }

        // 高亮最近的角点
        this._highlightCorner(closestCorner);

        return closestCorner;
    }

    _highlightCorner(corner) {
        // 移除之前的高亮
        if (this._cornerHighlight) {
            this.editor.scene.remove(this._cornerHighlight);
            this._cornerHighlight = null;
        }

        if (!corner) return;

        // 创建小的高亮球体
        var geometry = new THREE.SphereGeometry(0.03, 8, 8);
        var material = new THREE.MeshBasicMaterial({
            color: 0x00ff00,
            transparent: true,
            opacity: 0.8
        });
        this._cornerHighlight = new THREE.Mesh(geometry, material);
        this._cornerHighlight.position.set(corner.x, corner.y, corner.z);
        this.editor.scene.add(this._cornerHighlight);
    }

    _pointToRayDistance(px, py, pz) {
        var ray = this.raycaster.ray;
        var origin = ray.origin;
        var direction = ray.direction;

        // 计算点到射线的距离
        var dx = px - origin.x;
        var dy = py - origin.y;
        var dz = pz - origin.z;

        // 投影到射线方向
        var dot = dx * direction.x + dy * direction.y + dz * direction.z;

        // 如果在射线后方，返回大距离
        if (dot < 0) return Infinity;

        // 计算最近点
        var closestX = origin.x + direction.x * dot;
        var closestY = origin.y + direction.y * dot;
        var closestZ = origin.z + direction.z * dot;

        // 计算距离
        var distX = px - closestX;
        var distY = py - closestY;
        var distZ = pz - closestZ;

        return Math.sqrt(distX * distX + distY * distY + distZ * distZ);
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
