import math
import numpy as np
import pandas as pd
from dataclasses import dataclass

# =========================
# 常量与基础数据（题面）
# =========================
G = 9.81                 # m/s^2
SMOKE_R = 10.0           # 有效遮蔽半径
SMOKE_SINK = 3.0         # 云团下沉速度 m/s
SMOKE_TTL = 20.0         # 起爆后有效时长 s
MISSILE_SPEED = 300.0    # m/s
DT = 0.02                # 仿真时间步长（可调：0.01~0.05）
T_RELEASE_MAX = 60.0     # 释放时间上限（可调）
FUSE_MAX = 30.0          # 延时引信上限（可调）

# 导弹 M1 初始/方向（指向原点）
M1_START = np.array([20000.0, 0.0, 2000.0], dtype=float)
TARGET_CENTER = np.array([0.0, 200.0, 0.0], dtype=float)  # 圆柱中心点近似

# 计算 M1 到达原点的时间（仿真上限）
vec_to_origin = -M1_START
dist_to_origin = np.linalg.norm(vec_to_origin)
m1_dir = vec_to_origin / dist_to_origin
T_END = dist_to_origin / MISSILE_SPEED  # 仿真截止
# 也可适度延长窗口以覆盖过飞：T_END += 5.0

# 3 架 UAV 初始状态（x, y, z）
UAVS = {
    "FY1": np.array([17800.0,    0.0, 1800.0], dtype=float),
    "FY2": np.array([12000.0, 1400.0, 1400.0], dtype=float),
    "FY3": np.array([ 6000.0,-3000.0,  700.0], dtype=float),
}

# 速度界
V_MIN, V_MAX = 70.0, 140.0

# =========================
# 几何/判定工具
# =========================
def segment_point_distance(a: np.ndarray, b: np.ndarray, p: np.ndarray):
    """
    线段 ab 与点 p 的最小距离；返回 (距离, 投影参数 t∈[0,1])
    """
    ab = b - a
    ab2 = np.dot(ab, ab)
    if ab2 == 0.0:
        return np.linalg.norm(p - a), 0.0
    t = np.dot(p - a, ab) / ab2
    t_clamped = max(0.0, min(1.0, t))
    closest = a + t_clamped * ab
    return np.linalg.norm(p - closest), t_clamped

def missile_pos(t: float) -> np.ndarray:
    return M1_START + m1_dir * (MISSILE_SPEED * t)

# =========================
# 烟幕云团轨迹
# =========================
@dataclass
class BombPlan:
    uav_name: str
    theta: float          # 航向（平面方位角），弧度
    v: float              # 速度
    t_release: float      # 释放时刻
    fuse: float           # 延时引信（起爆相对释放的延迟）

def uav_xy_unit(theta: float) -> np.ndarray:
    # UAV 等高度直线飞行，平面航向角
    return np.array([math.cos(theta), math.sin(theta), 0.0], dtype=float)

def bomb_release_point(uav_start: np.ndarray, theta: float, v: float, t_release: float) -> np.ndarray:
    # UAV 等高度匀速直线：z 不变
    dir3 = uav_xy_unit(theta)
    return uav_start + dir3 * (v * t_release)

def bomb_detonation_point(release_pt: np.ndarray, theta: float, v: float, fuse: float) -> np.ndarray:
    # 平抛：水平方向随 UAV 速度匀速，竖直自由落体
    dir3 = uav_xy_unit(theta)
    horiz = dir3 * (v * fuse)
    dz = -0.5 * G * (fuse ** 2)
    return release_pt + horiz + np.array([0.0, 0.0, dz], dtype=float)

def smoke_center_at(t: float, det_t: float, det_pt: np.ndarray) -> np.ndarray:
    # t >= det_t 后，球心以 3 m/s 匀速下沉
    dt = t - det_t
    return det_pt + np.array([0.0, 0.0, -SMOKE_SINK * dt], dtype=float)

# =========================
# 遮蔽评估（离散时间）
# =========================
def evaluate_plans(plans: list) -> dict:
    """
    计算在 [0, T_END] 内的遮蔽总时长（秒），并提供细节。
    """
    # 准备每个烟幕实例的 (det_t, det_pt)
    smokes = []
    for plan in plans:
        # 释放点
        rel_pt = bomb_release_point(UAVS[plan.uav_name], plan.theta, plan.v, plan.t_release)
        # 起爆时刻/点
        det_t = plan.t_release + plan.fuse
        det_pt = bomb_detonation_point(rel_pt, plan.theta, plan.v, plan.fuse)
        # 云团有效窗口 [det_t, det_t + SMOKE_TTL]
        smokes.append({
            "uav": plan.uav_name,
            "release_pt": rel_pt,
            "det_t": det_t,
            "det_pt": det_pt,
        })

    # 时间网格
    ts = np.arange(0.0, T_END, DT)
    covered = np.zeros_like(ts, dtype=bool)

    for i, t in enumerate(ts):
        mpos = missile_pos(t)
        # 线段：导弹位置 -> 真目标中心
        for S in smokes:
            # 若不在云团寿命窗，跳过
            if not (S["det_t"] <= t <= S["det_t"] + SMOKE_TTL):
                continue
            c = smoke_center_at(t, S["det_t"], S["det_pt"])
            d, _ = segment_point_distance(mpos, TARGET_CENTER, c)
            if d <= SMOKE_R:
                covered[i] = True
                break

    total_cover = covered.sum() * DT
    # 计算每个 UAV 贡献时长（粗略：统计若只有该烟幕时的覆盖）
    indiv = {}
    for uav in UAVS.keys():
        mask = np.zeros_like(ts, dtype=bool)
        for i, t in enumerate(ts):
            mpos = missile_pos(t)
            for S in smokes:
                if S["uav"] != uav:
                    continue
                if not (S["det_t"] <= t <= S["det_t"] + SMOKE_TTL):
                    continue
                c = smoke_center_at(t, S["det_t"], S["det_pt"])
                d, _ = segment_point_distance(mpos, TARGET_CENTER, c)
                if d <= SMOKE_R:
                    mask[i] = True
                    break
        indiv[uav] = mask.sum() * DT

    return {
        "total_cover": total_cover,
        "details": smokes,
        "indiv_cover": indiv
    }

# =========================
# 差分进化（DE, rand/1/bin）
# =========================
class DE:
    def __init__(self, dim, bounds, pop_size=60, F=0.6, CR=0.9, max_gen=300, seed=42):
        self.dim = dim
        self.bounds = np.array(bounds, dtype=float)  # [(lo, hi), ...]
        self.pop_size = pop_size
        self.F = F
        self.CR = CR
        self.max_gen = max_gen
        self.rng = np.random.default_rng(seed)

    def ask(self):
        lo = self.bounds[:, 0]
        hi = self.bounds[:, 1]
        return self.rng.uniform(lo, hi, size=(self.pop_size, self.dim))

    def clip(self, X):
        lo = self.bounds[:, 0]
        hi = self.bounds[:, 1]
        return np.clip(X, lo, hi)

    def run(self, fobj):
        X = self.ask()
        fitness = np.array([fobj(x) for x in X])
        best_idx = np.argmax(fitness)
        Xbest = X[best_idx].copy()
        fbest = fitness[best_idx]

        for gen in range(self.max_gen):
            for i in range(self.pop_size):
                # 选择 r1,r2,r3
                idxs = np.arange(self.pop_size)
                idxs = idxs[idxs != i]
                r1, r2, r3 = self.rng.choice(idxs, size=3, replace=False)
                mutant = X[r1] + self.F * (X[r2] - X[r3])

                # 二进制交叉
                cross_mask = self.rng.random(self.dim) < self.CR
                # 确保至少有一个维度来自变异向量
                if not np.any(cross_mask):
                    cross_mask[self.rng.integers(0, self.dim)] = True
                trial = np.where(cross_mask, mutant, X[i])
                trial = self.clip(trial)

                f_trial = fobj(trial)
                if f_trial >= fitness[i]:  # 最大化
                    X[i], fitness[i] = trial, f_trial
                    if f_trial > fbest:
                        fbest, Xbest = f_trial, trial.copy()

        return Xbest, fbest

# =========================
# 决策变量与目标函数封装
# =========================
UAV_ORDER = ["FY1", "FY2", "FY3"]

def vec_to_plans(vec):
    """
    vec: 长度 12，按 [FY1 θ, FY1 v, FY1 trel, FY1 fuse, FY2 θ, ...] 排列
    """
    plans = []
    k = 0
    for uav in UAV_ORDER:
        theta = vec[k + 0]
        v     = vec[k + 1]
        trel  = vec[k + 2]
        fuse  = vec[k + 3]
        k += 4
        plans.append(BombPlan(uav, theta, v, trel, fuse))
    return plans

def objectives(vec):
    """
    返回需要最大化的指标：遮蔽总时长（秒）
    加轻微正则：惩罚过晚起爆（超出仿真窗）或无效配置（隐式会得到 0 覆盖）。
    """
    plans = vec_to_plans(vec)
    # 若起爆均在仿真窗外，几乎 0 覆盖，直接返回极小值
    det_times = [p.t_release + p.fuse for p in plans]
    if all(dt > T_END + SMOKE_TTL for dt in det_times):
        return -1e6
    res = evaluate_plans(plans)
    return res["total_cover"]

# 决策空间：每架 UAV 4 维
# θ ∈ [-π, π]，v ∈ [70, 140]，t_release ∈ [0, 60]，fuse ∈ [0, 30]
BDS_ONE = [(-math.pi, math.pi), (V_MIN, V_MAX), (0.0, T_RELEASE_MAX), (0.0, FUSE_MAX)]
BOUNDS = BDS_ONE * 3  # 复制三次

# =========================
# 主流程
# =========================
def main():
    de = DE(dim=len(BOUNDS), bounds=BOUNDS, pop_size=80, F=0.7, CR=0.9, max_gen=250, seed=2025)
    xbest, fbest = de.run(objectives)

    best_plans = vec_to_plans(xbest)
    eval_best = evaluate_plans(best_plans)

    # 汇总输出表（保存 result2.xlsx）
    rows = []
    for bp, det in zip(best_plans, eval_best["details"]):
        release_pt = det["release_pt"]
        det_pt = det["det_pt"]
        rows.append({
            "UAV": bp.uav_name,
            "heading_deg": np.degrees(bp.theta),
            "speed_mps": bp.v,
            "t_release_s": bp.t_release,
            "fuse_s": bp.fuse,
            "drop_x": release_pt[0], "drop_y": release_pt[1], "drop_z": release_pt[2],
            "det_x": det_pt[0], "det_y": det_pt[1], "det_z": det_pt[2],
            "individual_cover_s": eval_best["indiv_cover"][bp.uav_name],
        })
    df = pd.DataFrame(rows)
    df_total = pd.DataFrame([{"Total_cover_time_s": eval_best["total_cover"]}])

    # 若 openpyxl 可用则写 xlsx；否则退化为 csv
    out_path = "result2.xlsx"
    try:
        with pd.ExcelWriter(out_path, engine="openpyxl") as w:
            df.to_excel(w, index=False, sheet_name="plan")
            df_total.to_excel(w, index=False, sheet_name="summary")
        print(f"[OK] 写出 {out_path}；总遮蔽时长 = {eval_best['total_cover']:.3f} s")
    except Exception as e:
        print(f"[WARN] 写 xlsx 失败（{e}），改写 CSV。")
        df.to_csv("result2_plan.csv", index=False)
        df_total.to_csv("result2_summary.csv", index=False)
        print(f"[OK] 写出 result2_plan.csv / result2_summary.csv；总遮蔽时长 = {eval_best['total_cover']:.3f} s")

    # 终端打印简报
    print("\n=== Best Plan (DE) ===")
    for r in rows:
        print(
            f"{r['UAV']}: heading={r['heading_deg']:.2f}°, v={r['speed_mps']:.2f} m/s, "
            f"release={r['t_release_s']:.2f}s, fuse={r['fuse_s']:.2f}s, "
            f"cover≈{r['individual_cover_s']:.3f}s"
        )
    print(f"TOTAL COVER ≈ {eval_best['total_cover']:.3f} s")

if __name__ == "__main__":
    main()
