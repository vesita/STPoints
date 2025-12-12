
import {vector4to3, vector3_nomalize, psr_to_xyz, matmul} from "./util.js"
import {globalObjectCategory, } from './obj_cfg.js';
import { MovableView } from "./popup_dialog.js";

/**
 * 盒子图像上下文类，用于处理3D盒子在图像上的投影显示
 * @param {HTMLElement} ui - 画布UI元素
 */
function BoxImageContext(ui){

    this.ui = ui;
    
    /**
     * 更新焦点盒子的图像上下文，在图像上绘制高亮选中的3D盒子
     * @param {Object} box - 要绘制的3D盒子对象
     * @returns {void} 如果无法绘制则直接返回
     */
    this.updateFocusedImageContext = function(box){
        // 获取场景元数据
        var scene_meta = box.world.frameInfo.sceneMeta;

        // 选择最适合显示该盒子的相机视角
        let bestImage = choose_best_camera_for_point(
            scene_meta,
            box.position);

        if (!bestImage){
            return;           
        }

        if (!scene_meta.calib.camera){
            return;
        }
        
        var calib = scene_meta.calib.camera[bestImage]
        if (!calib){
            return;
        }
        
        if (calib){
            var img = box.world.cameras.getImageByName(bestImage);
            if (img && (img.naturalWidth > 0)){

                this.clear_canvas();

                var imgfinal = box_to_2d_points(box, calib)

                if (imgfinal != null){  // if projection is out of range of the image, stop drawing.
                    var ctx = this.ui.getContext("2d");
                    ctx.lineWidth = 0.5;

                    // note: 320*240 should be adjustable
                    var crop_area = crop_image(img.naturalWidth, img.naturalHeight, ctx.canvas.width, ctx.canvas.height, imgfinal);

                    ctx.drawImage(img, crop_area[0], crop_area[1],crop_area[2], crop_area[3], 0, 0, ctx.canvas.width, ctx.canvas.height);// ctx.canvas.clientHeight);
                    //ctx.drawImage(img, 0,0,img.naturalWidth, img.naturalHeight, 0, 0, 320, 180);// ctx.canvas.clientHeight);
                    var imgfinal = vectorsub(imgfinal, [crop_area[0],crop_area[1]]);
                    var trans_ratio = {
                        x: ctx.canvas.height/crop_area[3],
                        y: ctx.canvas.height/crop_area[3],
                    }

                    draw_box_on_image(ctx, box, imgfinal, trans_ratio, true);
                }
            }
        }
    }

    this.clear_canvas = function(){
        var c = this.ui;
        var ctx = c.getContext("2d");
        ctx.clearRect(0, 0, c.width, c.height);
    }


    function vectorsub(vs, v){
        var ret = [];
        var vl = v.length;

        for (var i = 0; i<vs.length/vl; i++){
            for (var j=0; j<vl; j++)
                ret[i*vl+j] = vs[i*vl+j]-v[j];
        }

        return ret;
    }


    function crop_image(imgWidth, imgHeight, clientWidth, clientHeight, corners)
    {
        var maxx=0, maxy=0, minx=imgWidth, miny=imgHeight;

        for (var i = 0; i < corners.length/2; i++){
            var x = corners[i*2];
            var y = corners[i*2+1];

            if (x>maxx) maxx=x;
            else if (x<minx) minx=x;

            if (y>maxy) maxy=y;
            else if (y<miny) miny=y;        
        }

        var targetWidth= (maxx-minx)*1.5;
        var targetHeight= (maxy-miny)*1.5;

        if (targetWidth/targetHeight > clientWidth/clientHeight){
            //increate height
            targetHeight = targetWidth*clientHeight/clientWidth;        
        }
        else{
            targetWidth = targetHeight*clientWidth/clientHeight;
        }

        var centerx = (maxx+minx)/2;
        var centery = (maxy+miny)/2;

        return [
            centerx - targetWidth/2,
            centery - targetHeight/2,
            targetWidth,
            targetHeight
        ];
    }

    function draw_box_on_image(ctx, box, box_corners, trans_ratio, selected){
        var imgfinal = box_corners;

        if (!selected){
                let target_color = null;
                if (box.world.data.cfg.color_obj == "category")
                {
                    target_color = globalObjectCategory.get_color_by_category(box.obj_type);
                }
                else // by id
                {
                    let idx = (box.obj_track_id)?parseInt(box.obj_track_id): box.obj_local_id;
                    target_color = globalObjectCategory.get_color_by_id(idx);
                }



                //ctx.strokeStyle = get_obj_cfg_by_type(box.obj_type).color;

                //var c = get_obj_cfg_by_type(box.obj_type).color;
                var r ="0x"+(target_color.x*256).toString(16);
                var g ="0x"+(target_color.y*256).toString(16);;
                var b ="0x"+(target_color.z*256).toString(16);;

            ctx.fillStyle="rgba("+parseInt(r)+","+parseInt(g)+","+parseInt(b)+",0.2)";
        }
        else{
            ctx.strokeStyle="#ff00ff";        
            ctx.fillStyle="rgba(255,0,255,0.2)";
        }

        // front panel
        ctx.beginPath();
        ctx.moveTo(imgfinal[3*2]*trans_ratio.x,imgfinal[3*2+1]*trans_ratio.y);

        for (var i=0; i < imgfinal.length/2/2; i++)
        {
            ctx.lineTo(imgfinal[i*2+0]*trans_ratio.x, imgfinal[i*2+1]*trans_ratio.y);
        }

        ctx.closePath();
        ctx.fill();
        
        // frame
        ctx.beginPath();

        ctx.moveTo(imgfinal[3*2]*trans_ratio.x,imgfinal[3*2+1]*trans_ratio.y);

        for (var i=0; i < imgfinal.length/2/2; i++)
        {
            ctx.lineTo(imgfinal[i*2+0]*trans_ratio.x, imgfinal[i*2+1]*trans_ratio.y);
        }
        //ctx.stroke();


        //ctx.strokeStyle="#ff00ff";
        //ctx.beginPath();

        ctx.moveTo(imgfinal[7*2]*trans_ratio.x,imgfinal[7*2+1]*trans_ratio.y);

        for (var i=4; i < imgfinal.length/2; i++)
        {
            ctx.lineTo(imgfinal[i*2+0]*trans_ratio.x, imgfinal[i*2+1]*trans_ratio.y);
        }
        
        ctx.moveTo(imgfinal[0*2]*trans_ratio.x,imgfinal[0*2+1]*trans_ratio.y);
        ctx.lineTo(imgfinal[4*2+0]*trans_ratio.x, imgfinal[4*2+1]*trans_ratio.y);
        ctx.moveTo(imgfinal[1*2]*trans_ratio.x,imgfinal[1*2+1]*trans_ratio.y);
        ctx.lineTo(imgfinal[5*2+0]*trans_ratio.x, imgfinal[5*2+1]*trans_ratio.y);
        ctx.moveTo(imgfinal[2*2]*trans_ratio.x,imgfinal[2*2+1]*trans_ratio.y);
        ctx.lineTo(imgfinal[6*2+0]*trans_ratio.x, imgfinal[6*2+1]*trans_ratio.y);
        ctx.moveTo(imgfinal[3*2]*trans_ratio.x,imgfinal[3*2+1]*trans_ratio.y);
        ctx.lineTo(imgfinal[7*2+0]*trans_ratio.x, imgfinal[7*2+1]*trans_ratio.y);


        ctx.stroke();
    }
}



class ImageContext extends MovableView{

    /**
     * 图像上下文类，用于处理相机图像的显示和交互
     * @param {HTMLElement} parentUi - 父级UI元素
     * @param {string} name - 图像名称
     * @param {boolean} autoSwitch - 是否自动切换
     * @param {Object} cfg - 配置对象
     * @param {Function} on_img_click - 图像点击回调函数
     */
    constructor(parentUi, name, autoSwitch, cfg, on_img_click){

        // 创建UI
        let template = document.getElementById("image-wrapper-template");
        let tool = template.content.cloneNode(true);
        // this.boxEditorHeaderUi.appendChild(tool);
        // return this.boxEditorHeaderUi.lastElementChild;

        parentUi.appendChild(tool);
        let ui = parentUi.lastElementChild;
        let handle = ui.querySelector("#move-handle");
        super(handle, ui);

        this.ui = ui;
        this.cfg = cfg;
        this.on_img_click = on_img_click;
        this.autoSwitch = autoSwitch;
        this.setImageName(name);
    }
    
    // 移除UI元素
    remove(){
        this.ui.remove();    
    }


    // 设置图像名称
    setImageName(name)
    {
        this.name = name;
        this.ui.querySelector("#header").innerText = (this.autoSwitch?"auto-":"")+name;
    }

    
    get_selected_box = null;


    // 初始化图像操作
    init_image_op(func_get_selected_box){
        this.ui.onclick = (e)=>this.on_click(e);
        this.get_selected_box = func_get_selected_box;
        // var h = parentUi.querySelector("#resize-handle");
        // h.onmousedown = resize_mouse_down;
        
        // c.onresize = on_resize;
    }
    
    // 清空主画布
    clear_main_canvas(){

        var boxes = this.ui.querySelector("#svg-boxes").children;
        
        if (boxes.length>0){
            for (var c=boxes.length-1; c >= 0; c--){
                boxes[c].remove();                    
            }
        }

        var points = this.ui.querySelector("#svg-points").children;
        
        if (points.length>0){
            for (var c=points.length-1; c >= 0; c--){
                points[c].remove();                    
            }
        }
    }






    world = null;
    img = null;

    // 关联世界对象
    attachWorld(world){
            this.world = world;
    };

    // 隐藏图像
    hide (){
        this.ui.style.display="none";
    };

    // 检查是否隐藏
    hidden(){
            this.ui.style.display=="none";
        };

    // 显示图像
    show(){
            this.ui.style.display="";
        };
    
    
    
    drawing = false;
    points = [];
    polyline;

    // 根据距离确定点的颜色
    img_lidar_point_map = {};

    point_color_by_distance(x,y)
    {
        // x,y 是图像坐标
        let p = this.img_lidar_point_map[y*this.img.width+x];

        // 计算点到传感器的距离
        let distance = Math.sqrt(p[1]*p[1] + p[2]*p[2] + p[3]*p[3] );

        // 限制距离范围在10-60米之间
        if (distance > 60.0)
            distance = 60.0;
        else if (distance < 10.0)
            distance = 10.0;
        
        // 根据距离生成颜色（近为绿色，远为红色）
        return [(distance-10)/50.0, 1- (distance-10)/50.0, 0].map(c=>{
            let hex = Math.floor(c*255).toString(16);
            if (hex.length == 1)
                hex = "0"+hex;
            return hex;
        }).reduce((a,b)=>a+b,"#"); 

    }

    // 将点列表转换为折线属性字符串
    to_polyline_attr(points){
        return points.reduce(function(x,y){
            return String(x)+","+y;
        }
        )
    }

    
    // 转换为视图框坐标
    to_viewbox_coord(x,y){
        var div = this.ui.querySelector("#maincanvas-svg");
        
        x = Math.round(x*2048/div.clientWidth);
        y = Math.round(y*1536/div.clientHeight);
        return [x,y];

    }

    // 处理点击事件
    on_click(e){
        var p= this.to_viewbox_coord(e.layerX, e.layerY);
        var x=p[0];
        var y=p[1];
        
        console.log("clicked",x,y);

        
        if (!this.drawing){

            if (e.ctrlKey){
                // 开始绘制模式
                this.drawing = true;
                var svg = this.ui.querySelector("#maincanvas-svg");
                //svg.style.position = "absolute";
                
                this.polyline = document.createElementNS("http://www.w3.org/2000/svg", 'polyline');
                svg.appendChild(this.polyline);
                this.points.push(x);
                this.points.push(y);

                
                this.polyline.setAttribute("class", "maincanvas-line")
                this.polyline.setAttribute("points", this.to_polyline_attr(this.points));

                var c = this.ui;
                c.onmousemove = on_move;
                c.ondblclick = on_dblclick;   
                c.onkeydown = on_key;    
            
            }
            else{
                // 非绘制模式
                //this is a test
                if (false){
                    let nearest_x = 100000;
                    let nearest_y = 100000;
                    let selected_pts = [];
                    
                    // 查找最近的点
                    for (let i =x-100; i<x+100; i++){
                        if (i < 0 || i >= this.img.width)
                            continue;

                        for (let j = y-100; j<y+100; j++){
                            if (j < 0 || j >= this.img.height)
                                continue;

                            let lidarpoint = this.img_lidar_point_map[j*this.img.width+i];
                            if (lidarpoint){
                                //console.log(i,j, lidarpoint);
                                selected_pts.push(lidarpoint); //激光雷达点的索引

                                if (((i-x) * (i-x) + (j-y)*(j-y)) < ((nearest_x-x)*(nearest_x-x) + (nearest_y-y)*(nearest_y-y))){
                                    nearest_x = i;
                                    nearest_y = j;                                
                                }
                            }
                                
                        }
                    }
                    console.log("nearest", nearest_x, nearest_y);
                    this.draw_point(nearest_x, nearest_y);
                    if (nearest_x < 100000)
                    {
                        this.on_img_click([this.img_lidar_point_map[nearest_y*this.img.width+nearest_x][0]]);
                    }
                }
                
            }

        } else {
            if (this.points[this.points.length-2]!=x || this.points[this.points.length-1]!=y){
                this.points.push(x);
                this.points.push(y);
                this.polyline.setAttribute("points", this.to_polyline_attr(this.points));
            }
            
        }


        function on_move(e){
            var p= to_viewbox_coord(e.layerX, e.layerY);
            var x=p[0];
            var y=p[1];

            console.log(x,y);
            this.polyline.setAttribute("points", this.to_polyline_attr(this.points) + ',' + x + ',' + y);
        }

        function on_dblclick(e){
            
            this.points.push(this.points[0]);
            this.points.push(this.points[1]);
            
            this.polyline.setAttribute("points", this.to_polyline_attr(this.points));
            console.log(this.points)
            
            all_lines.push(this.points);

            this.drawing = false;
            this.points = [];

            var c = this.ui;
            c.onmousemove = null;
            c.ondblclick = null;
            c.onkeypress = null;
            c.blur();
        }

        function cancel(){
                
                polyline.remove();

                this.drawing = false;
                this.points = [];
                var c = this.ui;
                c.onmousemove = null;
                c.ondblclick = null;
                c.onkeypress = null;

                c.blur();
        }

        function on_key(e){
            if (e.key == "Escape"){
                cancel();
                
            }
        }
    }


    // all boxes
    


    /**
     * 获取相机标定参数
     * @returns {Object|null} 相机标定参数对象或null（如果不存在）
     */
    getCalib(){
        // 获取场景元数据
        var scene_meta = this.world.sceneMeta;
           
        // 检查是否存在相机标定数据
        if (!scene_meta.calib.camera){
            return null;
        }

        // 获取当前相机的标定参数
        //var active_camera_name = this.world.cameras.active_name;
        var calib = scene_meta.calib.camera[this.name];

        return calib;
    }




    /**
     * 获取坐标变换比例
     * @returns {Object|null} 变换比例对象或null（如果没有图像）
     */
    get_trans_ratio(){
        // 获取相机图像
        var img = this.world.cameras.getImageByName(this.name);       

        // 如果没有图像或图像宽度为0，返回null
        if (!img || img.width==0){
            return null;
        }

        var clientWidth, clientHeight;

        // 设置客户端宽高
        clientWidth = 2048;
        clientHeight = 1536;

        // 计算变换比例
        var trans_ratio ={
            x: clientWidth/img.naturalWidth,
            y: clientHeight/img.naturalHeight,
        };

        return trans_ratio;
    }

    /**
     * 显示图像
     */
    show_image(){
        var svgimage = this.ui.querySelector("#svg-image");

        // 获取相机图像（全局设置，有时可能未设置）
        var img = this.world.cameras.getImageByName(this.name);
        if (img){
            svgimage.setAttribute("xlink:href", img.src);
        }

        this.img = img;


    }


    /**
     * 将点转换为SVG格式
     * @param {Array} points - 点坐标数组
     * @param {Object} trans_ratio - 坐标变换比例
     * @param {string} cssclass - CSS类名
     * @param {number} radius - 点的半径，默认为2
     * @returns {Element} SVG元素
     */
    points_to_svg(points, trans_ratio, cssclass, radius=2){
        // 根据变换比例调整点坐标
        var ptsFinal = points.map(function(x, i){
            if (i%2==0){
                return Math.round(x * trans_ratio.x);
            }else {
                return Math.round(x * trans_ratio.y);
            }
        });

        // 创建SVG组元素
        var svg = document.createElementNS("http://www.w3.org/2000/svg", 'g');
        
        // 设置CSS类
        if (cssclass)
        {
            svg.setAttribute("class", cssclass);
        }
        
        // 为每个点创建圆形元素
        for (let i = 0; i < ptsFinal.length; i+=2){

            
            let x = ptsFinal[i];
            let y = ptsFinal[i+1];

            let p = document.createElementNS("http://www.w3.org/2000/svg", 'circle');
            p.setAttribute("cx", x);
            p.setAttribute("cy", y);
            p.setAttribute("r", 2);
            p.setAttribute("stroke-width", "1");            

            // 如果没有指定CSS类，则根据距离设置颜色
            if (! cssclass){
                let image_x = points[i];
                let image_y = points[i+1];
                let color = point_color_by_distance(image_x, image_y);
                color += "24"; // 透明度
                p.setAttribute("stroke", color);
                p.setAttribute("fill", color);
            }
            
            svg.appendChild(p);
        }
        
        return svg;
    }

    /**
     * 绘制点
     * @param {number} x - X坐标
     * @param {number} y - Y坐标
     */
    draw_point(x,y){
        let trans_ratio = this.get_trans_ratio();
        let svg = this.ui.querySelector("#svg-points");
        let pts_svg = this.points_to_svg([x,y], trans_ratio, "radar-points");
        svg.appendChild(pts_svg);
    };




    /**
     * 渲染2D图像
     */
    render_2d_image (){

        
        // 如果禁用了主图像上下文，则返回
        if (this.cfg.disableMainImageContext)
            return;
        // 清空画布
        this.clear_main_canvas();

        // 显示图像
        this.show_image();
        // 绘制SVG元素
        this.draw_svg();
    }


    /**
     * 隐藏画布
     */
    hide_canvas(){
        //document.getElementsByClassName("ui-wrapper")[0].style.display="none";
        this.ui.style.display="none";
    }

    /**
     * 显示画布
     */
    show_canvas(){
        this.ui.style.display="inline";
    }


    /**
     * 绘制SVG元素
     */
    draw_svg(){
        // 绘制图像
        var img = this.world.cameras.getImageByName(this.name);

        // 如果没有图像或图像宽度为0，隐藏画布并返回
        if (!img || img.width==0){
            this.hide_canvas();
            return;
        }

        // 显示画布
        this.show_canvas();

        // 获取坐标变换比例
        var trans_ratio = this.get_trans_ratio();

        // 获取相机标定参数
        var calib = this.getCalib();
        if (!calib){
            return;
        }

        let svg = this.ui.querySelector("#svg-boxes");

        // 绘制3D框
        this.world.annotation.boxes.forEach((box)=>{
            // 将3D框投影到2D图像上
            var imgfinal = box_to_2d_points(box, calib);
            if (imgfinal){
                // 转换为SVG元素
                var box_svg = this.box_to_svg (box, imgfinal, trans_ratio, this.get_selected_box() == box);
                svg.appendChild(box_svg);
            }

        });

        svg = this.ui.querySelector("#svg-points");

        // 绘制雷达点
        if (this.cfg.projectRadarToImage)
        {
            this.world.radars.radarList.forEach(radar=>{
                // 获取未偏移的雷达点
                let pts = radar.get_unoffset_radar_points();
                // 将3D点投影到2D图像上
                let ptsOnImg = points3d_to_image2d(pts, calib);

                // 投影后可能没有点
                if (ptsOnImg && ptsOnImg.length>0){
                    // 转换为SVG元素
                    let pts_svg = this.points_to_svg(ptsOnImg, trans_ratio, radar.cssStyleSelector);
                    svg.appendChild(pts_svg);
                }
            });
        }



        // 将激光雷达点投影到相机图像上   
        if (this.cfg.projectLidarToImage){
            // 获取所有激光雷达点
            let pts = this.world.lidar.get_all_points();
            // 将3D点投影到2D图像上
            let ptsOnImg = points3d_to_image2d(pts, calib, true, this.img_lidar_point_map, img.width, img.height);

            // 投影后可能没有点
            if (ptsOnImg && ptsOnImg.length>0){
                // 转换为SVG元素
                let pts_svg = this.points_to_svg(ptsOnImg, trans_ratio);
                svg.appendChild(pts_svg);
            }
        }

    }

    /**
     * 将3D框转换为SVG元素
     * @param {Object} box - 3D框对象
     * @param {Array} box_corners - 框角点坐标
     * @param {Object} trans_ratio - 坐标变换比例
     * @param {boolean} selected - 是否选中
     * @returns {Element} SVG元素
     */
    box_to_svg(box, box_corners, trans_ratio, selected){
        
        // 根据变换比例调整角点坐标
        var imgfinal = box_corners.map(function(x, i){
            if (i%2==0){
                return Math.round(x * trans_ratio.x);
            }else {
                return Math.round(x * trans_ratio.y);
            }
        })

        // 创建SVG组元素
        var svg = document.createElementNS("http://www.w3.org/2000/svg", 'g');
        svg.setAttribute("id", "svg-box-local-"+box.obj_local_id);

        // 设置CSS类
        if (selected){
            svg.setAttribute("class", box.obj_type+" box-svg box-svg-selected");
        } else{
            if (box.world.data.cfg.color_obj == "id")
            {
                svg.setAttribute("class", "color-"+box.obj_track_id%33);
            }
            else // 按类别着色
            {
                svg.setAttribute("class", box.obj_type + " box-svg");
            }
        }

                
        // 创建前面板多边形
        var front_panel =  document.createElementNS("http://www.w3.org/2000/svg", 'polygon');
        svg.appendChild(front_panel);
        front_panel.setAttribute("points",
            imgfinal.slice(0, 4*2).reduce(function(x,y){            
                return String(x)+","+y;
            })
        )

        /*
        var back_panel =  document.createElementNS("http://www.w3.org/2000/svg", 'polygon');
        svg.appendChild(back_panel);
        back_panel.setAttribute("points",
            imgfinal.slice(4*2).reduce(function(x,y){            
                return String(x)+","+y;
            })
        )
        */

        // 绘制后面板的边
        for (var i = 0; i<4; ++i){
            var line =  document.createElementNS("http://www.w3.org/2000/svg", 'line');
            svg.appendChild(line);
            line.setAttribute("x1", imgfinal[(4+i)*2]);
            line.setAttribute("y1", imgfinal[(4+i)*2+1]);
            line.setAttribute("x2", imgfinal[(4+(i+1)%4)*2]);
            line.setAttribute("y2", imgfinal[(4+(i+1)%4)*2+1]);
        }


        // 绘制连接线（前面板到后面板）
        for (var i = 0; i<4; ++i){
            var line =  document.createElementNS("http://www.w3.org/2000/svg", 'line');
            svg.appendChild(line);
            line.setAttribute("x1", imgfinal[i*2]);
            line.setAttribute("y1", imgfinal[i*2+1]);
            line.setAttribute("x2", imgfinal[(i+4)*2]);
            line.setAttribute("y2", imgfinal[(i+4)*2+1]);
        }

        return svg;
    }


    // 盒子管理器
    boxes_manager = {
        // 显示图像
        display_image: ()=>{
            if (!this.cfg.disableMainImageContext)
                this.render_2d_image();
        },

        // 添加盒子
        add_box: (box)=>{
            var calib = this.getCalib();
            if (!calib){
                return;
            }
            var trans_ratio = this.get_trans_ratio();
            if (trans_ratio){
                var imgfinal = box_to_2d_points(box, calib);
                if (imgfinal){
                    var imgfinal = imgfinal.map(function(x, i){
                        if (i%2==0){
                            return Math.round(x * trans_ratio.x);
                        }else {
                            return Math.round(x * trans_ratio.y);
                        }
                    })

                    var svg_box = this.box_to_svg(box, imgfinal, trans_ratio);
                    var svg = this.ui.querySelector("#svg-boxes");
                    svg.appendChild(svg_box);
                }
            }
        },


        // 当盒子被选中时
        onBoxSelected: (box_obj_local_id, obj_type)=>{
            var b = this.ui.querySelector("#svg-box-local-"+box_obj_local_id);
            if (b){
                b.setAttribute("class", "box-svg-selected");
            }
        },


        // 当盒子取消选中时
        onBoxUnselected: (box_obj_local_id, obj_type)=>{
            var b = this.ui.querySelector("#svg-box-local-"+box_obj_local_id);

            if (b)
                b.setAttribute("class", obj_type);
        },

        remove_box: (box_obj_local_id)=>{
            var b = this.ui.querySelector("#svg-box-local-"+box_obj_local_id);

            if (b)
                b.remove();
        },

        update_obj_type: (box_obj_local_id, obj_type)=>{
            this.onBoxSelected(box_obj_local_id, obj_type);
        },
        
        update_box: (box)=>{
            var b = this.ui.querySelector("#svg-box-local-"+box.obj_local_id);
            if (!b){
                return;
            }

            var children = b.childNodes;

            var calib = this.getCalib();
            if (!calib){
                return;
            }

            var trans_ratio = this.get_trans_ratio();
            var imgfinal = box_to_2d_points(box, calib);

            if (!imgfinal){
                //box may go out of image
                return;
            }
            var imgfinal = imgfinal.map(function(x, i){
                if (i%2==0){
                    return Math.round(x * trans_ratio.x);
                }else {
                    return Math.round(x * trans_ratio.y);
                }
            })

            if (imgfinal){
                var front_panel = children[0];
                front_panel.setAttribute("points",
                    imgfinal.slice(0, 4*2).reduce(function(x,y){            
                        return String(x)+","+y;
                    })
                )



                for (var i = 0; i<4; ++i){
                    var line =  children[1+i];
                    line.setAttribute("x1", imgfinal[(4+i)*2]);
                    line.setAttribute("y1", imgfinal[(4+i)*2+1]);
                    line.setAttribute("x2", imgfinal[(4+(i+1)%4)*2]);
                    line.setAttribute("y2", imgfinal[(4+(i+1)%4)*2+1]);
                }


                for (var i = 0; i<4; ++i){
                    var line =  children[5+i];
                    line.setAttribute("x1", imgfinal[i*2]);
                    line.setAttribute("y1", imgfinal[i*2+1]);
                    line.setAttribute("x2", imgfinal[(i+4)*2]);
                    line.setAttribute("y2", imgfinal[(i+4)*2+1]);
                }
            }

        }
    }

}


class ImageContextManager {
    constructor(parentUi, selectorUi, cfg, on_img_click){
        this.parentUi = parentUi;
        this.selectorUi = selectorUi;
        this.cfg = cfg;
        this.on_img_click = on_img_click;

        this.addImage("", true);


        this.selectorUi.onmouseenter=function(event){
            if (this.timerId)
            {
                clearTimeout(this.timerId);
                this.timerId = null;
            }
            
            event.target.querySelector("#camera-list").style.display="";

        };


        this.selectorUi.onmouseleave=function(event){
            let ui = event.target.querySelector("#camera-list");

            this.timerId = setTimeout(()=>{
                ui.style.display="none";
                this.timerId = null;
            },
            200);           

        };

        this.selectorUi.querySelector("#camera-list").onclick = (event)=>{
            let cameraName = event.target.innerText;
            
            if (cameraName == "auto"){

                let existed = this.images.find(x=>x.autoSwitch);

                if (existed)
                {
                    this.removeImage(existed);
                }
                else
                {
                    this.addImage("", true);
                }

            }
            else{
                let existed = this.images.find(x=>!x.autoSwitch && x.name == cameraName);

                if (existed)
                {
                    this.removeImage(existed);
                    
                }
                else
                {
                    this.addImage(cameraName);
                }
            }
        };

    }

    updateCameraList(cameras){

        let autoCamera = '<div class="camera-item" id="camera-item-auto">auto</div>';

        if (this.images.find(i=>i.autoSwitch)){
            autoCamera = '<div class="camera-item camera-selected" id="camera-item-auto">auto</div>';
        }

        let camera_selector_str = cameras.map(c=>{

                let existed = this.images.find(i=>i.name == c && !i.autoSwitch);
                let className = existed?"camera-item camera-selected":"camera-item";

                return `<div class="${className}" id="camera-item-${c}">${c}</div>`;
            }).reduce((x,y)=>x+y,  autoCamera);

        let ui = this.selectorUi.querySelector("#camera-list");
        ui.innerHTML = camera_selector_str;
        ui.style.display="none";

        this.setDefaultBestCamera(cameras[0]);
    }

    setDefaultBestCamera(c){

        if (!this.bestCamera)
        {
            let existed = this.images.find(x=>x.autoSwitch);
            if (existed)
            {
                existed.setImageName(c);
            }

            this.bestCamera = c;
        }
    }

    images = [];
    addImage(name, autoSwitch)
    {

        if (autoSwitch && this.bestCamera && !name)
            name = this.bestCamera;
            
        let image = new ImageContext(this.parentUi, name, autoSwitch, this.cfg, this.on_img_click);

        this.images.push(image);

        if (this.init_image_op_para){
            image.init_image_op(this.init_image_op_para);
        }

        if (this.world){
            image.attachWorld(this.world);
            image.render_2d_image();
        }


        let selectorName = autoSwitch?"auto":name;
            
        let ui = this.selectorUi.querySelector("#camera-item-"+selectorName);
        if (ui)
            ui.className = "camera-item camera-selected";


        return image;
    }

    removeImage(image){

        let selectorName = image.autoSwitch?"auto":image.name;
        this.selectorUi.querySelector("#camera-item-"+selectorName).className = "camera-item";
        this.images = this.images.filter(x=>x!=image);
        image.remove();


    }

    setBestCamera(camera)
    {
        this.images.filter(i=>i.autoSwitch).forEach(i=>{
            i.setImageName(camera);
            i.boxes_manager.display_image();
        });

        this.bestCamera = camera;
    }

    render_2d_image(){
        this.images.forEach(i=>i.render_2d_image());
    }

    attachWorld(world){

        this.world = world;
        this.images.forEach(i=>i.attachWorld(world));
    }

    hide(){
        this.images.forEach(i=>i.hide());
    }


    show(){
        this.images.forEach(i=>i.show());
    }

    clear_main_canvas(){
        this.images.forEach(i=>i.clear_main_canvas());
    }

    init_image_op(op){
        this.init_image_op_para = op;
        this.images.forEach(i=>i.init_image_op(op));
    }
    hidden(){
        return false;
    }

    choose_best_camera_for_point = choose_best_camera_for_point;

    self = this;

    boxes_manager = {
        
        display_image: ()=>{
            if (!this.cfg.disableMainImageContext)
                this.render_2d_image();
        },

        add_box: (box)=>{
            this.images.forEach(i=>i.boxes_manager.add_box(box));
        },


        onBoxSelected: (box_obj_local_id, obj_type)=>{
            this.images.forEach(i=>i.boxes_manager.onBoxSelected(box_obj_local_id, obj_type));
        },


        onBoxUnselected: (box_obj_local_id, obj_type)=>{
            this.images.forEach(i=>i.boxes_manager.onBoxUnselected(box_obj_local_id, obj_type));
        },

        remove_box: (box_obj_local_id)=>{
            this.images.forEach(i=>i.boxes_manager.remove_box(box_obj_local_id));
        },

        update_obj_type: (box_obj_local_id, obj_type)=>{
            this.images.forEach(i=>i.boxes_manager.update_obj_type(box_obj_local_id, obj_type));
        },
        
        update_box: (box)=>{
            this.images.forEach(i=>i.boxes_manager.update_box(box));
        }
    }
    

}

/**
 * 将3D框转换为2D图像上的点
 * @param {Object} box - 3D框对象
 * @param {Object} calib - 相机标定参数
 * @returns {Array} 2D点坐标数组
 */
function box_to_2d_points(box, calib){
    var scale = box.scale;
    var pos = box.position;
    var rotation = box.rotation;

    // 将位置、尺寸和旋转转换为3D坐标
    var box3d = psr_to_xyz(pos, scale, rotation);

    //console.log(box.obj_track_id, box3d.slice(8*4));

    box3d = box3d.slice(0,8*4);
    // 将3D点投影到2D图像上
    return points3d_homo_to_image2d(box3d, calib);
}   

/**
 * 将齐次坐标系的3D点投影到2D图像上
 * @param {Array} points3d - 3D点坐标（齐次坐标，长度为4的行向量）
 * @param {Object} calib - 相机标定参数
 * @param {boolean} accept_partial - 是否接受部分点（默认为false）
 * @param {Object} save_map - 保存映射关系的对象
 * @param {number} img_dx - 图像宽度
 * @param {number} img_dy - 图像高度
 * @returns {Array} 2D点坐标数组
 */
function points3d_homo_to_image2d(points3d, calib, accept_partial=false,save_map, img_dx, img_dy){
    // 使用外参矩阵将点从世界坐标系变换到相机坐标系
    var imgpos = matmul(calib.extrinsic, points3d, 4);
    
    // 对于KITTI数据集，需要应用rect矩阵
    if (calib.rect){
        imgpos = matmul(calib.rect, imgpos, 4);
    }

    // 将4维向量转换为3维向量
    var imgpos3 = vector4to3(imgpos);

    var imgpos2;
    // 根据内参矩阵的长度选择不同的处理方式
    if (calib.intrinsic.length>9) {
        imgpos2 = matmul(calib.intrinsic, imgpos, 4);
    }
    else
        imgpos2 = matmul(calib.intrinsic, imgpos3, 3);

    // 归一化坐标
    let imgfinal = vector3_nomalize(imgpos2);
    let imgfinal_filterd = [];
     
    // 如果接受部分点
    if (accept_partial){
        let temppos=[];
        let p = imgpos3;
        for (var i = 0; i<p.length/3; i++){
            // 只处理Z坐标大于0的点（在相机前方）
            if (p[i*3+2]>0){
                let x = imgfinal[i*2];
                let y = imgfinal[i*2+1];
                
                x = Math.round(x);
                y = Math.round(y);
                // 检查点是否在图像范围内
                if (x > 0 && x < img_dx && y > 0 && y < img_dy){
                    if (save_map){
                        // 保存点的映射关系
                        save_map[img_dx*y+x] = [i, points3d[i*4+0], points3d[i*4+1], points3d[i*4+2]];  //save index? a little dangerous! //[points3d[i*4+0], points3d[i*4+1], points3d[i*4+2]];
                    }

                    imgfinal_filterd.push(x);
                    imgfinal_filterd.push(y);

                }
                else{
                    // console.log("points outside of image",x,y);
                }

            }
        }        

        imgfinal = imgfinal_filterd;
        //warning: what if calib.intrinsic.length
        //todo: this function need clearance
        //imgpos2 = matmul(calib.intrinsic, temppos, 3);
    }
    // 如果不接受部分点且有点不在图像范围内，返回null
    else  if (!accept_partial && !all_points_in_image_range(imgpos3)){
            return null;
    }

    return imgfinal;
}

/**
 * 将3D点转换为齐次坐标
 * @param {Array} points - 3D点坐标
 * @returns {Array} 齐次坐标
 */
function point3d_to_homo(points){
    let homo=[];
    for (let i =0; i<points.length; i+=3){
        homo.push(points[i]);
        homo.push(points[i+1]);
        homo.push(points[i+2]);
        homo.push(1);
    }

    return homo;
}

/**
 * 将3D点投影到2D图像上
 * @param {Array} points - 3D点坐标
 * @param {Object} calib - 相机标定参数
 * @param {boolean} accept_partial - 是否接受部分点（默认为false）
 * @param {Object} save_map - 保存映射关系的对象
 * @param {number} img_dx - 图像宽度
 * @param {number} img_dy - 图像高度
 * @returns {Array} 2D点坐标数组
 */
function points3d_to_image2d(points, calib, accept_partial=false, save_map, img_dx, img_dy){
    // 
    return points3d_homo_to_image2d(point3d_to_homo(points), calib, accept_partial, save_map, img_dx, img_dy);
}

/**
 * 检查所有点是否都在图像范围内（Z坐标大于0）
 * @param {Array} p - 3D点坐标
 * @returns {boolean} 是否所有点都在图像范围内
 */
function all_points_in_image_range(p){
    for (var i = 0; i<p.length/3; i++){
        if (p[i*3+2]<0){
            return false;
        }
    }
    
    return true;
}


/**
 * 为点选择最佳相机
 * @param {Object} scene_meta - 场景元数据
 * @param {Object} center - 中心点坐标
 * @returns {string|null} 最佳相机名称或null
 */
function  choose_best_camera_for_point(scene_meta, center){
        
    // 检查是否存在标定数据
    if (!scene_meta.calib){
        return null;
    }

    var proj_pos = [];
    // 遍历所有相机，计算点在各相机坐标系中的位置
    for (var i in scene_meta.calib.camera){
        var imgpos = matmul(scene_meta.calib.camera[i].extrinsic, [center.x,center.y,center.z,1], 4);
        proj_pos.push({calib: i, pos: vector4to3(imgpos)});
    }

    // 过滤出在图像范围内的投影点
    var valid_proj_pos = proj_pos.filter(function(p){
        return all_points_in_image_range(p.pos);
    });
    
    // 计算各点到图像中心的距离
    valid_proj_pos.forEach(function(p){
        p.dist_to_center = p.pos[0]*p.pos[0] + p.pos[1]*p.pos[1];
    });

    // 按距离排序
    valid_proj_pos.sort(function(x,y){
        return x.dist_to_center - y.dist_to_center;
    });

    //console.log(valid_proj_pos);

    // 返回距离最近的相机
    if (valid_proj_pos.length>0){
        return valid_proj_pos[0].calib;
    }

    return null;

}


export {ImageContextManager, BoxImageContext};
