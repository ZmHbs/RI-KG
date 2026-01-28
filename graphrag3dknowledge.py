# /home/hbs/projects/ljx/utils/graphrag3dknowledge_v8_fullscreen.py
# -*- coding: utf-8 -*-
"""
[Nature-Style] 3D Knowledge Graph V8.6 - Full Screen & Responsive
-------------------------------------------------------------
Changes:
1. Removed fixed resolution (900x1600).
2. Enabled 'autosize=True' to fill the browser window.
3. Preserved all features (Mode Filtering, Color Hierarchy, Text Sizing).
"""

import os
import json
import argparse
import networkx as nx
import plotly.graph_objects as go
import plotly.io as pio
import numpy as np

# ================= 配置区 =================
JSON_PATH = "/home/hbs/projects/ljx/data/records_v3_9room_v2.json"
pio.renderers.default = "browser"
FONT_FAMILY = "Times New Roman"

VALID_ROOMS = [
    "kitchen", "bedroom", "living room", "office room",
    "bathroom", "dining room", "laundry room", "gym", "lounge"
]


# ================= 逻辑层 (ETL & Filter) =================

def build_graph(json_path: str, style_config: dict) -> nx.DiGraph:
    """ETL 数据构建流水线"""
    if not os.path.exists(json_path):
        json_path = os.path.basename(json_path)

    with open(json_path, "r", encoding="utf-8-sig") as f:
        records = json.load(f)

    G = nx.DiGraph()
    for rec in records:
        if not isinstance(rec, dict): continue
        n = rec.get("n", {})
        m = rec.get("m", {})
        r = rec.get("r", {})
        if not n or not m: continue

        def get_node(node_data):
            props = node_data.get("properties", {})
            labels = node_data.get("labels", [])
            name = props.get("name", "").strip() or props.get("chinese_name", "").strip() or str(
                node_data.get("identity"))

            # --- 节点类型判定 ---
            is_room = "Room" in labels
            is_tool = "Category" in labels or "Tool" in labels or "Furniture" in labels

            if is_room:
                node_type = "Room"
                color = style_config["node_color_room"]
                size = style_config["node_size_room"]
                text_size = style_config["text_size_room"]
            elif is_tool:
                node_type = "Tool"
                color = style_config["node_color_tool"]
                size = style_config["node_size_tool"]
                text_size = style_config["text_size_tool"]
            else:
                node_type = "Entity"
                color = style_config["node_color_entity"]
                size = style_config["node_size_entity"]
                text_size = style_config["text_size_entity"]

            return f"{node_type}:{name}", {
                "name": name,
                "type": node_type,
                "color": color,
                "size": size,
                "text_size": text_size
            }

        u, u_attr = get_node(n)
        v, v_attr = get_node(m)

        G.add_node(u, **u_attr)
        G.add_node(v, **v_attr)

        rel_type = r.get("type", "RELATED")
        G.add_edge(u, v, type=rel_type)

    return G


def filter_graph_by_mode(G: nx.DiGraph, mode: str) -> nx.DiGraph:
    """根据模式过滤图谱"""
    mode = mode.lower().strip()

    if mode == "all":
        print("�� 模式 'all': 显示完整图谱")
        return G

    target_node = None
    for node in G.nodes():
        if G.nodes[node]["type"] == "Room" and G.nodes[node]["name"].lower() == mode:
            target_node = node
            break

    if not target_node:
        print(f"⚠️ 警告: 未找到房间 '{mode}'，显示完整图谱。可选房间: {VALID_ROOMS}")
        return G

    print(f"�� 模式 '{mode}': 正在提取子图...")
    kept_nodes = set([target_node])
    related_tools = set()

    for neighbor in G.successors(target_node):
        kept_nodes.add(neighbor)
        related_tools.add(neighbor)
    for neighbor in G.predecessors(target_node):
        kept_nodes.add(neighbor)
        related_tools.add(neighbor)

    for tool in related_tools:
        for neighbor in G.predecessors(tool):
            kept_nodes.add(neighbor)
        for neighbor in G.successors(tool):
            kept_nodes.add(neighbor)

    print(f"✅ 提取完成: {len(kept_nodes)} 个节点")
    return G.subgraph(kept_nodes)


def compute_layout(G):
    print("⚙️ 计算 3D 布局 (Spring Model)...")
    k_val = 0.5 if len(G.nodes) < 50 else 0.25
    pos = nx.spring_layout(G, dim=3, seed=42, iterations=100, k=k_val)
    return pos


# ================= 可视化层 (Responsive Fix) =================

def determine_edge_style(u, v, G, config):
    source_type = G.nodes[u]["type"]
    target_type = G.nodes[v]["type"]

    if source_type == "Room" and target_type == "Room":
        return "Topological Adjacency", config["edge_color_connected"]
    if source_type == "Room":
        return "Spatial Constraint Flow", config["edge_color_flow_room"]
    if source_type == "Tool":
        return "Tool Affordance Flow", config["edge_color_flow_tool"]
    return "Attribute Association Flow", config["edge_color_flow_entity"]


def create_traces(G, pos, config):
    print("⚙️ 生成可视化对象...")
    traces = []

    categories = {
        "Topological Adjacency": {"color": config["edge_color_connected"], "data": _init_data()},
        "Spatial Constraint Flow": {"color": config["edge_color_flow_room"], "data": _init_data()},
        "Tool Affordance Flow": {"color": config["edge_color_flow_tool"], "data": _init_data()},
        "Attribute Association Flow": {"color": config["edge_color_flow_entity"], "data": _init_data()},
    }

    for u, v in G.edges():
        if u not in pos or v not in pos: continue

        x0, y0, z0 = pos[u]
        x1, y1, z1 = pos[v]

        cat_name, color = determine_edge_style(u, v, G, config)
        if cat_name not in categories: cat_name = "Attribute Association Flow"

        d = categories[cat_name]["data"]
        d["lx"].extend([x0, x1, None])
        d["ly"].extend([y0, y1, None])
        d["lz"].extend([z0, z1, None])

        ratio = 0.7
        cx = x0 + (x1 - x0) * ratio
        cy = y0 + (y1 - y0) * ratio
        cz = z0 + (z1 - z0) * ratio
        vx, vy, vz = x1 - x0, y1 - y0, z1 - z0
        norm = np.sqrt(vx ** 2 + vy ** 2 + vz ** 2) + 1e-9

        d["cx"].append(cx);
        d["cy"].append(cy);
        d["cz"].append(cz)
        d["cu"].append(vx / norm);
        d["cv"].append(vy / norm);
        d["cw"].append(vz / norm)

    for cat_name, info in categories.items():
        d = info["data"]
        color = info["color"]
        if not d["lx"]: continue

        traces.append(go.Scatter3d(
            x=d["lx"], y=d["ly"], z=d["lz"],
            mode="lines",
            line=dict(width=config["edge_width"], color=color),
            opacity=0.6, hoverinfo="none", showlegend=False
        ))

        if d["cx"]:
            traces.append(go.Cone(
                x=d["cx"], y=d["cy"], z=d["cz"],
                u=d["cu"], v=d["cv"], w=d["cw"],
                colorscale=[[0, color], [1, color]],
                showscale=False, sizemode="absolute", sizeref=0.08,
                anchor="center", hoverinfo="none", showlegend=False
            ))
    return traces


def _init_data():
    return {"lx": [], "ly": [], "lz": [], "cx": [], "cy": [], "cz": [], "cu": [], "cv": [], "cw": []}


def visualize(G, config, mode_title):
    if len(G.nodes) == 0:
        print("❌ 错误: 当前模式下没有节点可显示。")
        return

    pos = compute_layout(G)
    graph_traces = create_traces(G, pos, config)

    # 节点属性
    node_x, node_y, node_z = [], [], []
    node_colors, node_sizes, node_texts, node_text_sizes = [], [], [], []

    for node in G.nodes():
        x, y, z = pos[node]
        node_x.append(x);
        node_y.append(y);
        node_z.append(z)
        attrs = G.nodes[node]
        node_colors.append(attrs["color"])
        node_sizes.append(attrs["size"])
        node_texts.append(attrs["name"])
        node_text_sizes.append(attrs["text_size"])

    node_trace = go.Scatter3d(
        x=node_x, y=node_y, z=node_z,
        mode="markers+text",
        text=node_texts,
        textposition="top center",
        marker=dict(
            size=node_sizes, color=node_colors,
            line=dict(width=config["border_width"], color=config["border_color"]),
            opacity=1.0
        ),
        textfont=dict(family=FONT_FAMILY, size=node_text_sizes, color="black"),
        hoverinfo="text", showlegend=False
    )

    fig = go.Figure()
    for trace in graph_traces: fig.add_trace(trace)
    fig.add_trace(node_trace)

    axis_3d = dict(showbackground=False, showticklabels=False, title="", showgrid=False, zeroline=False,
                   showspikes=False)

    # --- [核心修改] 布局设置为全屏响应式 ---
    fig.update_layout(
        title=f"3D Knowledge Graph - Mode: {mode_title}",
        scene=dict(
            xaxis=axis_3d, yaxis=axis_3d, zaxis=axis_3d,
            aspectmode="cube",
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
        ),
        template="plotly_white",
        font=dict(family=FONT_FAMILY, size=12),
        margin=dict(l=0, r=0, t=30, b=0),  # 最小化边距 (仅保留顶部标题空间)
        autosize=True,  # 开启自动适应窗口大小
        # height=900, width=1600,         # [已删除] 移除固定分辨率
        showlegend=False,
        hovermode="closest"
    )

    print(f"�� 启动可视化: {mode_title} (全屏自适应模式)...")
    fig.show()


# ================= 主函数 =================

def main():
    parser = argparse.ArgumentParser(description="3D 知识图谱 (V8.6 全屏版)")

    # --- 核心模式 ---
    parser.add_argument("--mode", type=str, default="kitchen",
                        help=f"显示模式: 'all' 或 指定房间名. 可选: {VALID_ROOMS}")

    # --- 文字大小参数 ---
    parser.add_argument("--text_size_room", type=int, default=18, help="[文字] 房间名称大小")
    parser.add_argument("--text_size_tool", type=int, default=12, help="[文字] 工具/类别名称大小")
    parser.add_argument("--text_size_entity", type=int, default=8, help="[文字] 实体名称大小")

    # --- 节点样式 ---
    parser.add_argument("--node_color_room", type=str, default="#DE4640", help="房间颜色(红)")
    parser.add_argument("--node_size_room", type=int, default=15)
    parser.add_argument("--node_color_tool", type=str, default="#FF9F43", help="工具颜色(橙)")
    parser.add_argument("--node_size_tool", type=int, default=13)
    parser.add_argument("--node_color_entity", type=str, default="#2187CB", help="实体颜色(蓝)")
    parser.add_argument("--node_size_entity", type=int, default=10)

    parser.add_argument("--border_color", type=str, default="rgba(50, 50, 50, 0.5)")
    parser.add_argument("--border_width", type=float, default=2)
    parser.add_argument("--edge_width", type=float, default=2)

    # --- 连线颜色 ---
    parser.add_argument("--edge_color_flow_room", type=str, default="#DE4640", help="Room->Tool (Red)")
    parser.add_argument("--edge_color_flow_tool", type=str, default="#2187CB", help="Tool->Entity (Blue)")
    parser.add_argument("--edge_color_flow_entity", type=str, default="#2187CB", help="Entity->Tool (Blue)")
    parser.add_argument("--edge_color_connected", type=str, default="#F4A261", help="Room<->Room (Orange)")

    parser.add_argument("--data", type=str, default=JSON_PATH)

    args = parser.parse_args()
    config = vars(args)

    print(f"�� 启动参数: Mode={args.mode}, AutoSize=True")

    if os.path.exists(args.data) or os.path.exists(os.path.basename(args.data)):
        G_full = build_graph(args.data, config)
        G_filtered = filter_graph_by_mode(G_full, args.mode)
        visualize(G_filtered, config, args.mode)
    else:
        print(f"❌ 错误: 找不到数据文件 {args.data}")


if __name__ == "__main__":
    main()