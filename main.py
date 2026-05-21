import taichi as ti
import numpy as np

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

# --- 【性能优化核心 1】：新增一个用于存放曲线坐标的 GPU 缓冲区 ---
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

@ti.kernel
def clear_pixels():
    """并行清空像素缓冲区"""
    for i, j in pixels:
        pixels[i, j] = ti.Vector([0.0, 0.0, 0.0])

# --- 【性能优化核心 2】：将“点亮像素”的工作交给 GPU 并行执行 ---
@ti.kernel
def draw_curve_kernel(n: ti.i32):
    # 这个 for 循环在 kernel 中，Taichi 会自动将其在 GPU 上极速执行
    for i in range(n):
        pt = curve_points_field[i]
        x_pixel = ti.cast(pt[0] * WIDTH, ti.i32)
        y_pixel = ti.cast(pt[1] * HEIGHT, ti.i32)
        if 0 <= x_pixel < WIDTH and 0 <= y_pixel < HEIGHT:
            pixels[x_pixel, y_pixel] = ti.Vector([0.0, 1.0, 0.0])

def main():
    window = ti.ui.Window("Bezier Curve (60 FPS Restored)", (WIDTH, HEIGHT))
    canvas = window.get_canvas()
    control_points = []
    
    while window.running:
        for e in window.get_events(ti.ui.PRESS):
            if e.key == ti.ui.LMB: 
                if len(control_points) < MAX_CONTROL_POINTS:
                    pos = window.get_cursor_pos()
                    control_points.append(pos)
                    print(f"Added control point: {pos}")
            elif e.key == 'c': 
                control_points = []
                print("Canvas cleared.")
        
        clear_pixels()
        
        current_count = len(control_points)
        if current_count >= 2:
            # 1. 在 CPU 端 (Python) 把所有点的坐标算好，存进 numpy 数组
            curve_points_np = np.zeros((NUM_SEGMENTS + 1, 2), dtype=np.float32)
            for t_int in range(NUM_SEGMENTS + 1):
                t = t_int / NUM_SEGMENTS
                curve_points_np[t_int] = de_casteljau(control_points, t)
            
            # 2. 一次性打包发送给 GPU (只发生 1 次内存通信，而不是 1000 次)
            curve_points_field.from_numpy(curve_points_np)
            
            # 3. 呼叫 GPU：数据已经提供，去显存里将对应像素涂绿
            draw_curve_kernel(NUM_SEGMENTS + 1)
                    
        canvas.set_image(pixels)
        
        if current_count > 0:
            np_points = np.full((MAX_CONTROL_POINTS, 2), -10.0, dtype=np.float32)
            np_points[:current_count] = np.array(control_points, dtype=np.float32)
            gui_points.from_numpy(np_points)
            canvas.circles(gui_points, radius=0.006, color=(1.0, 0.0, 0.0))
            
            if current_count >= 2:
                np_indices = np.zeros(MAX_CONTROL_POINTS * 2, dtype=np.int32)
                indices = []
                for i in range(current_count - 1):
                    indices.extend([i, i + 1])
                np_indices[:len(indices)] = np.array(indices, dtype=np.int32)
                gui_indices.from_numpy(np_indices)
                canvas.lines(gui_points, width=0.002, indices=gui_indices, color=(0.5, 0.5, 0.5))
        
        window.show()

if __name__ == '__main__':
    main()