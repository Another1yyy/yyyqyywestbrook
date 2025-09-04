import math
import numpy as np

# =========================
# 基础向量与几何工具
# =========================
def unit(v):
    n = np.linalg.norm(v)
    if n == 0:
        return v
    return v / n

def dist_point_to_segment(P, A, B):
    """
    返回点 P 到线段 AB 的距离，以及投影参数 s_clamped ∈ [0, 1]
    s_clamped=0/1 表示最近点在 A/B；中间表示最近点在线段内部。
    """
    BA = B - A
    l2 = float(np.dot(BA, BA))
    if l2 == 0.0:
        return float(np.linalg.norm(P - A)), 0.0
    s = float(np.dot(P - A, BA) / l2)
    s_clamped = max(0.0, min(1.0, s))
    Q = A + s_clamped * BA
    return float(np.linalg.norm(P - Q)), s_clamped

def bisect_root(f, a, b, tol=1e-10, maxiter=200):
    """在 [a,b] 上用二分法求 f(t)=0（要求 f(a)*f(b) <= 0）"""
    fa, fb = f(a), f(b)
    if fa == 0.0:
        return a
    if fb == 0.0:
        return b
    if fa * fb > 0:
        return None
    lo, hi = a, b
    for _ in range(maxiter):
        mid = 0.5 * (lo + hi)
        fm = f(mid)
        if abs(fm) < tol or (hi - lo) < tol:
            return mid
        if fa * fm <= 0:
            hi = mid
            fb = fm
        else:
            lo = mid
            fa = fm
    return 0.5 * (lo + hi)

def find_cover_intervals(f, t0, t1, dt=0.02):
    """
    在 [t0, t1] 以步长 dt 扫描 f(t)=D(t)-R_cloud，求 D(t)<=R_cloud 的时间区间。
    返回 [(tin, tout), ...]
    """
    ts, vs = [], []
    t = t0
    while t <= t1 + 1e-12:
        ts.append(t)
        vs.append(f(t))
        t += dt

    roots = []
    # 扫描相邻采样段，寻找变号以二分细化
    for i in range(1, len(ts)):
        a, b = ts[i - 1], ts[i]
        fa, fb = vs[i - 1], vs[i]
        if fa == 0.0:
            roots.append(a)
        if fa * fb < 0.0:
            r = bisect_root(f, a, b)
            if r is not None:
                roots.append(r)

    # 将根排序并配对成进入/离开区间；若端点处即满足，也做边界处理
    roots = sorted(roots)
    intervals = []

    # 辅助：判定端点是否在覆盖内
    def inside(t):
        return f(t) <= 0.0

    # 处理可能在 t0 或 t1 就处于覆盖内的情形
    cur_in = inside(t0)
    cursor = t0
    for r in roots:
        if cur_in:
            # 从内到外
            intervals.append((cursor, r))
            cur_in = False
        else:
            # 从外到内
            cursor = r
            cur_in = True

    # 若扫描结束时仍在覆盖内，补上尾段
    if cur_in:
        intervals.append((cursor, t1))

    # 过滤掉可能的极短/数值误差段
    intervals = [(a, b) for (a, b) in intervals if b > a + 1e-8]
    return intervals

# =========================
# 题目“第一问”参数（可改）
# =========================
g = 9.8              # 重力加速度 (m/s^2)
v_m = 300.0          # 导弹速度 (m/s)
R_cloud = 10.0       # 云团有效半径 (m)
sink_v = 3.0         # 云团下沉速度 (m/s)
effective_span = 20.0  # 起爆后有效 20 s

# 真目标与导弹初始
T = np.array([0.0, 200.0, 0.0])          # 真目标
M0 = np.array([20000.0, 0.0, 2000.0])    # 导弹 M1 初始位置，指向原点

# FY1 无人机：初始、航向、速度、投放与起爆设置
F0 = np.array([17800.0, 0.0, 1800.0])    # FY1 初始
h = np.array([-1.0, 0.0, 0.0])           # 航向：沿 -x 指向假目标方向（题意）
v_u = 120.0                               # FY1 速度
t_drop = 1.5                              # 受令后投放时刻 (s)
tau = 3.6                                  # 引信延时 (s)

t_e = t_drop + tau                         # 起爆时刻


# =========================
# 轨迹与距离函数
# =========================
u_m = unit(-M0)  # 导弹单位方向（指向假目标原点）

def missile_pos(t):
    """导弹位置 M(t)"""
    return M0 + v_m * t * u_m

def drone_pos(t):
    """无人机水平等高直线飞行位置 F(t)"""
    return F0 + v_u * t * h

# 投放点 R 与起爆点 E
R = drone_pos(t_drop)
E = R + v_u * tau * h + 0.5 * np.array([0.0, 0.0, -g]) * (tau ** 2)

def cloud_center(t):
    """
    起爆后云团中心：以 3 m/s 竖直下沉
    仅在 t >= t_e 有意义；调用者会限制时间窗口
    """
    return E + np.array([0.0, 0.0, -sink_v]) * (t - t_e)

def D_minus_R(t):
    """
    f(t) = 点到线段距离 - 云团半径
    线段为 [M(t), T]，点为 C(t)
    返回 f(t), D(t), s*(投影参数，便于诊断)
    """
    A = missile_pos(t)
    B = T
    P = cloud_center(t)
    D, s = dist_point_to_segment(P, A, B)
    return D - R_cloud, D, s

# =========================
# 求解：进入/离开遮蔽时刻与时长
# =========================
if __name__ == "__main__":
    t0 = t_e
    t1 = t_e + effective_span

    # 包装成标量函数供根查找
    f = lambda t: D_minus_R(t)[0]

    intervals = find_cover_intervals(f, t0, t1, dt=0.02)

    # 汇总与打印
    total_cover = sum(b - a for (a, b) in intervals)
    print("=== 第一问：FY1 单次投放 对导弹 M1 的遮蔽判定 ===")
    print(f"起爆时刻 t_e = {t_e:.6f} s；有效窗口 [{t0:.3f}, {t1:.3f}] s")
    print(f"起爆点 E = ({E[0]:.3f}, {E[1]:.3f}, {E[2]:.3f}) m")

    if intervals:
        for k, (a, b) in enumerate(intervals, 1):
            # 可选：在区间内取若干采样点，找最小距离/最近点类型（诊断）
            ts = np.linspace(a, b, 2001)
            Ds = []
            ss = []
            for tt in ts:
                _, D, s = D_minus_R(tt)
                Ds.append(D); ss.append(s)
            imin = int(np.argmin(Ds))
            print(f"- 区间{k}: 进入 {a:.6f} s, 离开 {b:.6f} s, 时长 {b-a:.6f} s")
            print(f"  · 区间内最小距离 ≈ {Ds[imin]:.3f} m @ t ≈ {ts[imin]:.6f} s, "
                  f"s* ≈ {ss[imin]:.6e} (≈0 表示最近点在导弹端)")
        print(f"===> 有效遮蔽总时长: {total_cover:.6f} s")
    else:
        print("未发生遮蔽（D(t) 始终 > 10 m）")