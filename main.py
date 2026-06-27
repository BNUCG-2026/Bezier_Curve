import taichi as ti
import numpy as np
import math

# 使用 gpu 后端
ti.init(arch=ti.gpu)

WIDTH = 800
HEIGHT = 800
MAX_CONTROL_POINTS = 100
NUM_SEGMENTS = 1000 # 曲线采样点数量

# 像素缓冲区
pixels = ti.Vector.field(3, dtype=ti.f32, shape=(WIDTH, HEIGHT))

# GUI 绘制数据缓冲池
gui_points = ti.Vector.field(2, dtype=ti.f32, shape=MAX_CONTROL_POINTS)
gui_indices = ti.field(dtype=ti.i32, shape=MAX_CONTROL_POINTS * 2)

# 用于存放曲线坐标的 GPU 缓冲区
curve_points_field = ti.Vector.field(2, dtype=ti.f32, shape=NUM_SEGMENTS + 1)

def de_casteljau(points, t):
    """纯 Python 递归实现 De Casteljau 算法"""
    if len(points) == 1:
        return points[0]
    next_points = []
    for i in range(len(points) - 1):
        p0 = points[i]
        p1 = points[i+1]
        x = (1.0 - t) * p0[0] + t * p1[0]
        y = (1.0 - t) * p0[1] + t * p1[1]
        next_points.append([x, y])
    return de_casteljau(next_points, t)

def cubic_b_spline(control_points, num_segments):
    """均匀三次 B 样条曲线计算 (矩阵形式)"""
    n = len(control_points)
    if n < 4:
        return np.zeros((num_segments + 1, 2), dtype=np.float32)
    
    # 三次 B 样条基矩阵
    M_B = np.array([
        [-1.0,  3.0, -3.0, 1.0],
        [ 3.0, -6.0,  3.0, 0.0],
        [-3.0,  0.0,  3.0, 0.0],
        [ 1.0,  4.0,  1.0, 0.0]
    ]) / 6.0

    curve_points = []
    num_curves = n - 3 # 分段数
    
    # 为了保证总采样点为 num_segments + 1，对每一段进行均匀分配
    for idx in range(num_curves):
        # 取出当前段的 4 个控制点
        p = np.array(control_points[idx:idx+4])
        
        # 计算当前段的采样点数
        start_seg = int(idx * num_segments / num_curves)
        end_seg = int((idx + 1) * num_segments / num_curves)
        if idx == num_curves - 1:
            end_seg = num_segments # 确保包含终点
            
        for s in range(start_seg, end_seg + 1):
            # 将全局段数归一化到局部段的 [0, 1] 之间
            if end_seg == start_seg:
                t = 0.0
            else:
                t = (s - start_seg) / (end_seg - start_seg) if idx != num_curves - 1 else (s - start_seg) / (end_seg - start_seg)
            # 处理终点闭合重合，这里直接按总步长计算
            t_global = (s - int(idx * num_segments / num_curves)) / (int((idx + 1) * num_segments / num_curves) - int(idx * num_segments / num_curves))
            
            T = np.array([t_global**3, t_global**2, t_global, 1.0])
            pt = T @ M_B @ p
            curve_points.append(pt)
            
    # 过滤可能因为分段首尾重叠产生的多余点，确保刚好满足长度
    return np.array(curve_points[:num_segments + 1], dtype=np.float32)

@ti.kernel
def clear_pixels():
    """并行清空像素缓冲区"""
    for i, j in pixels:
        pixels[i, j] = ti.Vector([0.0, 0.0, 0.0])

@ti.kernel
def draw_curve_kernel(n: ti.i32, anti_aliasing: ti.i32):
    """GPU 并行点亮像素内核 (支持 3x3 邻域的反走样)"""
    for i in range(n):
        pt = curve_points_field[i]
        x_exact = pt[0] * WIDTH
        y_exact = pt[1] * HEIGHT
        
        if anti_aliasing == 1:
            # 反走样逻辑：考察中心点周围 3x3 像素邻域
            x_center = ti.cast(ti.round(x_exact), ti.i32)
            y_center = ti.cast(ti.round(y_exact), ti.i32)
            
            for offset_x in range(-1, 2):
                for offset_y in range(-1, 2):
                    nx = x_center + offset_x
                    ny = y_center + offset_y
                    
                    if 0 <= nx < WIDTH and 0 <= ny < HEIGHT:
                        # 计算像素中心到精确几何坐标的欧氏距离
                        dist = ti.sqrt((ti.cast(nx, ti.f32) + 0.5 - x_exact)**2 + (ti.cast(ny, ti.f32) + 0.5 - y_exact)**2)
                        # 距离衰减模型 (使用高斯衰减)
                        sigma = 0.8
                        weight = ti.exp(-(dist**2) / (2.0 * sigma**2))
                        
                        # 混合颜色，使用原子最大值或累加防止多点覆盖时变暗/过载
                        # 曲线颜色为绿色 [0.0, 1.0, 0.0]
                        ti.atomic_max(pixels[nx, ny][1], weight)
        else:
            # 基础光栅化：直接截断硬点亮
            x_pixel = ti.cast(x_exact, ti.i32)
            y_pixel = ti.cast(y_exact, ti.i32)
            if 0 <= x_pixel < WIDTH and 0 <= y_pixel < HEIGHT:
                pixels[x_pixel, y_pixel] = ti.Vector([0.0, 1.0, 0.0])

def main():
    window = ti.ui.Window("Bezier & B-Spline Curve (with Anti-Aliasing)", (WIDTH, HEIGHT))
    canvas = window.get_canvas()
    control_points = []
    
    # 状态变量
    mode = 0  # 0: 贝塞尔曲线模式, 1: B样条曲线模式
    anti_aliasing = 0 # 0: 关闭反走样, 1: 开启反走样
    
    print("=======================================================")
    print("操作指南:")
    print("  [鼠标左键] : 添加控制点")
    print("  [C 键]     : 清空画布")
    print("  [M 键]     : 切换模式 (当前: Bezier 贝塞尔)")
    print("  [A 键]     : 开关反走样抗锯齿 (当前: 关闭)")
    print("=======================================================")

    while window.running:
        for e in window.get_events(ti.ui.PRESS):
            if e.key == ti.ui.LMB: 
                if len(control_points) < MAX_CONTROL_POINTS:
                    pos = window.get_cursor_pos()
                    control_points.append(pos)
                    print(f"添加控制点: {pos}")
            elif e.key == 'c': 
                control_points = []
                print("画布已清空。")
            elif e.key == 'm':
                mode = 1 - mode
                mode_str = "B-Spline (B样条)" if mode == 1 else "Bezier (贝塞尔)"
                print(f"模式切换 -> {mode_str}")
            elif e.key == 'a':
                anti_aliasing = 1 - anti_aliasing
                aa_str = "开启" if anti_aliasing == 1 else "关闭"
                print(f"反走样抗锯齿 -> {aa_str}")
        
        clear_pixels()
        
        current_count = len(control_points)
        
        # 渲染曲线逻辑
        if mode == 0 and current_count >= 2:
            # 贝塞尔曲线模式
            curve_points_np = np.zeros((NUM_SEGMENTS + 1, 2), dtype=np.float32)
            for t_int in range(NUM_SEGMENTS + 1):
                t = t_int / NUM_SEGMENTS
                curve_points_np[t_int] = de_casteljau(control_points, t)
            curve_points_field.from_numpy(curve_points_np)
            draw_curve_kernel(NUM_SEGMENTS + 1, anti_aliasing)
            
        elif mode == 1 and current_count >= 4:
            # B样条曲线模式 (至少需要4个点)
            curve_points_np = cubic_b_spline(control_points, NUM_SEGMENTS)
            curve_points_field.from_numpy(curve_points_np)
            draw_curve_kernel(NUM_SEGMENTS + 1, anti_aliasing)
                    
        canvas.set_image(pixels)
        
        # 绘制交互控制点与控制多边形连线
        if current_count > 0:
            np_points = np.full((MAX_CONTROL_POINTS, 2), -10.0, dtype=np.float32)
            np_points[:current_count] = np.array(control_points, dtype=np.float32)
            gui_points.from_numpy(np_points)
            
            # 控制点用红色表示
            canvas.circles(gui_points, radius=0.006, color=(1.0, 0.0, 0.0))
            
            if current_count >= 2:
                np_indices = np.zeros(MAX_CONTROL_POINTS * 2, dtype=np.int32)
                indices = []
                for i in range(current_count - 1):
                    indices.extend([i, i + 1])
                np_indices[:len(indices)] = np.array(indices, dtype=np.int32)
                gui_indices.from_numpy(np_indices)
                # 灰连线表示控制多边形
                canvas.lines(gui_points, width=0.002, indices=gui_indices, color=(0.5, 0.5, 0.5))
        
        window.show()

if __name__ == '__main__':
    main()
