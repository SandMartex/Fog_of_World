#!/usr/bin/env python3
"""
世界迷雾 (.fwss) → 美国县级地图生成器
用法: python fow_county_map.py your_snapshot.fwss

依赖安装:
    pip install geopandas matplotlib shapely requests
"""

import os, sys, math, zlib, struct, hashlib, zipfile
import argparse

# ─────────────────────────────────────────────
# Part 1: .fwss 解析器 (inline from CaviarChen/Fog-of-World-Data-Parser)
# ─────────────────────────────────────────────

FILENAME_MASK1 = "olhwjsktri"
FILENAME_MASK2 = "eizxdwknmo"
FILENAME_ENCODING = {k: v for v, k in enumerate(FILENAME_MASK1)}

MAP_WIDTH      = 512
TILE_WIDTH     = 128
TILE_HEADER_LEN  = TILE_WIDTH ** 2
TILE_HEADER_SIZE = TILE_HEADER_LEN * 2
BLOCK_BITMAP_SIZE = 512
BLOCK_EXTRA_DATA  = 3
BLOCK_SIZE        = BLOCK_BITMAP_SIZE + BLOCK_EXTRA_DATA
BITMAP_WIDTH      = 64

NNZ_FOR_BYTE = bytes(bin(x).count("1") for x in range(256))

def _nnz(data):
    return sum(NNZ_FOR_BYTE[b] for b in data)

class Block:
    def __init__(self, x, y, data):
        self.x = x
        self.y = y
        self.bitmap = data[:BLOCK_BITMAP_SIZE]

    def is_visited(self, x, y):
        return self.bitmap[(x // 8) + y * 8] & (1 << (7 - x % 8))

    def has_any_visit(self):
        return any(self.bitmap)


class Tile:
    def __init__(self, sync_folder, filename):
        self.id = 0
        for v in [FILENAME_ENCODING[c] for c in filename[4:-2]]:
            self.id = self.id * 10 + v
        self.x = self.id % MAP_WIDTH
        self.y = self.id // MAP_WIDTH

        with open(os.path.join(sync_folder, filename), "rb") as f:
            raw = zlib.decompress(f.read())

        header = struct.unpack(f"{TILE_HEADER_LEN}H", raw[:TILE_HEADER_SIZE])
        self.blocks = {}
        for i, block_idx in enumerate(header):
            if block_idx > 0:
                bx, by = i % TILE_WIDTH, i // TILE_WIDTH
                off = TILE_HEADER_SIZE + (block_idx - 1) * BLOCK_SIZE
                block = Block(bx, by, raw[off: off + BLOCK_SIZE])
                if block.has_any_visit():
                    self.blocks[(bx, by)] = block


def pixel_to_lng_lat(px, py):
    total = MAP_WIDTH * TILE_WIDTH * BITMAP_WIDTH   # 4,194,304
    lng = px / total * 360 - 180
    lat = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * py / total))))
    return lng, lat


def extract_visited_points(fwss_path, region="USA"):
    """解析 .fwss，返回所有 visited block 中心点 (lng, lat) 列表
    直接从 zip 内存读取，绕过 Windows 不允许 * 号目录名的限制。
    """
    # 美国 bbox (含 AK/HI)
    if region == "USA":
        lng_min, lng_max = -180, -60
        lat_min, lat_max = 15, 75
    else:
        lng_min, lng_max = -180, 180
        lat_min, lat_max = -90, 90

    with zipfile.ZipFile(fwss_path) as z:
        all_names = z.namelist()

        # 找 Model/*/ 目录下的 tile 文件（zip 内路径用 / 分隔）
        # 示例路径: Model/*/0092lowoooee
        tile_entries = {}  # filename -> zip entry name
        for name in all_names:
            parts = name.replace("\\", "/").split("/")
            # 格式: Model / * / <tilefile>
            if len(parts) == 3 and parts[0] == "Model" and parts[1] == "*":
                fname = parts[2]
                if len(fname) > 6:
                    tile_entries[fname] = name

        if not tile_entries:
            # 打印所有条目帮助调试
            print("zip 内容示例:", all_names[:10])
            raise FileNotFoundError(
                "找不到 Model/* 目录下的 tile 文件\n"
                f"zip 内容: {all_names[:5]}..."
            )

        print(f"total {len(tile_entries)} tiles, parsing...")
        points = []

        for fname, zip_entry in tile_entries.items():
            try:
                raw_bytes = z.read(zip_entry)
                raw = zlib.decompress(raw_bytes)
            except Exception:
                continue

            # 解析 tile id from filename
            try:
                tile_id = 0
                for c in fname[4:-2]:
                    tile_id = tile_id * 10 + FILENAME_ENCODING[c]
            except (KeyError, IndexError):
                continue

            tx = tile_id % MAP_WIDTH
            ty = tile_id // MAP_WIDTH

            # 解析 header
            if len(raw) < TILE_HEADER_SIZE:
                continue
            header = struct.unpack(f"{TILE_HEADER_LEN}H", raw[:TILE_HEADER_SIZE])

            for i, block_idx in enumerate(header):
                if block_idx == 0:
                    continue
                bx = i % TILE_WIDTH
                by = i // TILE_WIDTH
                off = TILE_HEADER_SIZE + (block_idx - 1) * BLOCK_SIZE
                if off + BLOCK_SIZE > len(raw):
                    continue
                bitmap = raw[off: off + BLOCK_BITMAP_SIZE]
                if not any(bitmap):
                    continue
                # Block 中心的全局像素坐标
                gpx = tx * TILE_WIDTH * BITMAP_WIDTH + bx * BITMAP_WIDTH + BITMAP_WIDTH // 2
                gpy = ty * TILE_WIDTH * BITMAP_WIDTH + by * BITMAP_WIDTH + BITMAP_WIDTH // 2
                lng, lat = pixel_to_lng_lat(gpx, gpy)
                if lng_min <= lng <= lng_max and lat_min <= lat <= lat_max:
                    points.append((lng, lat))

    print(f"Visited blocks in USA: {len(points)}")
    return points


# ─────────────────────────────────────────────
# Part 2: 下载县级边界 + 空间叠加
# ─────────────────────────────────────────────

def _topojson_to_geojson(topo, layer="counties"):
    """轻量 TopoJSON → GeoJSON（无需第三方包），过滤退化环"""
    transform = topo.get("transform", {})
    scale = transform.get("scale", [1, 1])
    translate = transform.get("translate", [0, 0])

    def decode_arc(arc):
        x = y = 0
        coords = []
        for dx, dy in arc:
            x += dx; y += dy
            coords.append([x * scale[0] + translate[0], y * scale[1] + translate[1]])
        return coords

    arcs = [decode_arc(a) for a in topo["arcs"]]

    def stitch(arc_indices):
        ring = []
        for idx in arc_indices:
            seg = arcs[idx] if idx >= 0 else arcs[~idx][::-1]
            ring.extend(seg if not ring else seg[1:])
        # 确保环闭合
        if ring and ring[0] != ring[-1]:
            ring.append(ring[0])
        return ring

    def valid_ring(ring):
        # LinearRing 需要至少 4 个点（首尾相同）
        return len(ring) >= 4

    def geom_to_geojson(g):
        t = g["type"]
        if t == "Polygon":
            rings = [stitch(r) for r in g["arcs"]]
            rings = [r for r in rings if valid_ring(r)]
            if not rings:
                return None
            return {"type": "Polygon", "coordinates": rings}
        if t == "MultiPolygon":
            polys = []
            for poly_arcs in g["arcs"]:
                rings = [stitch(r) for r in poly_arcs]
                rings = [r for r in rings if valid_ring(r)]
                if rings:
                    polys.append(rings)
            if not polys:
                return None
            return {"type": "MultiPolygon", "coordinates": polys}
        return None

    features = []
    for g in topo["objects"][layer]["geometries"]:
        geo = geom_to_geojson(g)
        if geo:
            features.append({
                "type": "Feature",
                "id": g.get("id"),
                "properties": g.get("properties") or {"GEOID": str(g.get("id", ""))},
                "geometry": geo,
            })
    return {"type": "FeatureCollection", "features": features}


def download_counties_geojson(cache_path="us_counties.geojson"):
    """下载美国县级数据（带缓存）。优先 TopoJSON，内置转换无需额外依赖。"""
    if os.path.exists(cache_path):
        print(f"使用缓存的县级数据: {cache_path}")
        return cache_path

    import requests, json

    # 方案 A：直接 GeoJSON
    geojson_urls = [
        "https://raw.githubusercontent.com/holtzy/D3-graph-gallery/master/DATA/us_states.geojson",  # placeholder
    ]

    # 方案 B：us-atlas TopoJSON（最可靠，CDN 多备份）
    topo_urls = [
        "https://cdn.jsdelivr.net/npm/us-atlas@3/counties-10m.json",
        "https://unpkg.com/us-atlas@3/counties-10m.json",
        "https://raw.githubusercontent.com/topojson/us-atlas/master/counties-10m.json",
    ]
    for url in topo_urls:
        try:
            print(f"下载县级边界数据: {url}")
            r = requests.get(url, timeout=120)
            r.raise_for_status()
            topo = r.json()
            print("  转换 TopoJSON → GeoJSON ...")
            geojson = _topojson_to_geojson(topo, "counties")
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(geojson, f)
            print(f"  已保存到 {cache_path}")
            return cache_path
        except Exception as e:
            print(f"  失败: {e}")

    # 方案 C：直接 GeoJSON 兜底（plotly 数据集，含 FIPS，最稳）
    direct_geojson_urls = [
        "https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json",
    ]
    for url in direct_geojson_urls:
        try:
            print(f"下载县级边界数据(GeoJSON): {url}")
            r = requests.get(url, timeout=120)
            r.raise_for_status()
            geojson = r.json()
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(geojson, f)
            print(f"  已保存到 {cache_path}")
            return cache_path
        except Exception as e:
            print(f"  失败: {e}")

    raise RuntimeError(
        "所有下载源均失败，请手动下载并保存为 us_counties.geojson:\n"
        "  https://cdn.jsdelivr.net/npm/us-atlas@3/counties-10m.json\n"
        "（脚本支持直接读取该 topojson 文件，改名为 us_counties.geojson 即可）"
    )


def find_visited_counties(points, counties_gdf):
    """空间叠加：找出包含 visited points 的县"""
    from shapely.geometry import MultiPoint
    import geopandas as gpd

    print("Running spatial join (point-in-polygon)...")
    pts_gdf = gpd.GeoDataFrame(
        geometry=gpd.points_from_xy([p[0] for p in points],
                                     [p[1] for p in points]),
        crs="EPSG:4326"
    )
    counties_4326 = counties_gdf.to_crs("EPSG:4326")

    joined = gpd.sjoin(pts_gdf, counties_4326[["geometry"]], how="left", predicate="within")
    visited_idx = set(joined["index_right"].dropna().astype(int))
    print(f"Matched {len(visited_idx)} counties")
    return visited_idx


# ─────────────────────────────────────────────
# Part 3: 绘图
# ─────────────────────────────────────────────

# 州 FIPS → 州名（含阿拉斯加 02 / 夏威夷 15）
STATE_NAMES = {
    "01":"Alabama","02":"Alaska","04":"Arizona","05":"Arkansas","06":"California",
    "08":"Colorado","09":"Connecticut","10":"Delaware","11":"Washington DC",
    "12":"Florida","13":"Georgia","15":"Hawaii","16":"Idaho","17":"Illinois",
    "18":"Indiana","19":"Iowa","20":"Kansas","21":"Kentucky",
    "22":"Louisiana","23":"Maine","24":"Maryland","25":"Massachusetts",
    "26":"Michigan","27":"Minnesota","28":"Mississippi","29":"Missouri",
    "30":"Montana","31":"Nebraska","32":"Nevada","33":"New Hampshire",
    "34":"New Jersey","35":"New Mexico","36":"New York","37":"North Carolina",
    "38":"North Dakota","39":"Ohio","40":"Oklahoma","41":"Oregon",
    "42":"Pennsylvania","44":"Rhode Island","45":"South Carolina",
    "46":"South Dakota","47":"Tennessee","48":"Texas","49":"Utah",
    "50":"Vermont","51":"Virginia","53":"Washington","54":"West Virginia",
    "55":"Wisconsin","56":"Wyoming",
}


def _state_fips(row):
    # try common column names for county FIPS / state FIPS
    # 优先直接的州字段，再退回到县级 FIPS（取前两位）
    for col in ("STATEFP", "STATE", "GEOID", "geoid", "id", "GEO_ID"):
        v = row.get(col)
        if v is None:
            continue
        s = str(v).strip()
        if not s:
            continue
        # 处理 Census 风格的 GEO_ID，如 "0500000US01001"
        if "US" in s:
            s = s.split("US")[-1]
        # 长度 <= 2 视为州级 FIPS（如 STATE="01"）；否则是县级 FIPS（取前两位）
        if len(s) <= 2:
            return s.zfill(2)
        return s.zfill(5)[:2]   # first 2 digits = state FIPS
    return "00"


def _setup_fonts():
    """注册随仓库附带的 Source Sans 3 字体；找不到则优雅退回。
    返回 (regular_family, semibold_family)。"""
    from matplotlib import font_manager as fm
    here = os.path.dirname(os.path.abspath(__file__))
    font_dir = os.path.join(here, "fonts")
    reg = semi = None
    files = {
        "SourceSans3-Regular.otf": "reg",
        "SourceSans3-Semibold.otf": "semi",
        "SourceSans3-Bold.otf": "bold",
    }
    for fname in files:
        fpath = os.path.join(font_dir, fname)
        if os.path.exists(fpath):
            try:
                fm.fontManager.addfont(fpath)
            except Exception:
                pass
    names = {f.name for f in fm.fontManager.ttflist}
    if "Source Sans 3" in names:
        # Source Sans 3 的 Regular/Semibold/Bold 同名，靠 weight 区分
        return "Source Sans 3", "Source Sans 3"
    # 退回到系统里较干净的无衬线
    for cand in ("Liberation Sans", "FreeSans", "DejaVu Sans"):
        if cand in names:
            return cand, cand
    return "DejaVu Sans", "DejaVu Sans"


def plot_map(counties_gdf, visited_idx, output_path="fow_usa_counties.png"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.path import Path
    import numpy as np
    import pyproj
    from shapely.ops import transform as shp_transform

    # ── fonts ─────────────────────────────────────────────
    FONT_REG, FONT_SEMI = _setup_fonts()
    plt.rcParams["font.family"] = FONT_REG
    plt.rcParams["svg.fonttype"] = "none"

    # ── projections ──────────────────────────────────────
    wgs84      = pyproj.CRS("EPSG:4326")
    albers_us  = pyproj.CRS("ESRI:102003")   # USA Contiguous Albers Equal Area
    albers_ak  = pyproj.CRS("ESRI:102006")   # Alaska Albers Equal Area
    albers_hi  = pyproj.CRS("ESRI:102007")   # Hawaii Albers Equal Area

    print("Classifying counties (CONUS / AK / HI)...")
    # classify using original WGS84 centroid
    counties_wgs = counties_gdf.to_crs(wgs84)

    conus_idx, ak_idx, hi_idx = [], [], []
    for idx, row in counties_wgs.iterrows():
        g = row.geometry
        if g is None or g.is_empty:
            continue
        c = g.centroid
        lon, lat = c.x, c.y
        # 先判夏威夷：其经度 ~-155 也 < -141，必须在阿拉斯加之前判断
        if -161 < lon < -154 and 18 < lat < 23:
            hi_idx.append(idx)
        elif lat > 51 or lon < -141:
            ak_idx.append(idx)
        # Strict CONUS bbox: lon -125..-65, lat 24..50
        elif -125 <= lon <= -65 and 24 <= lat <= 50:
            conus_idx.append(idx)
        # else: overseas territory — skip entirely

    conus_wgs = counties_wgs.loc[conus_idx]
    conus_alb = conus_wgs.to_crs(albers_us)
    ak_wgs    = counties_wgs.loc[ak_idx]
    ak_alb    = ak_wgs.to_crs(albers_ak)
    hi_wgs    = counties_wgs.loc[hi_idx]
    hi_alb    = hi_wgs.to_crs(albers_hi)
    print(f"  CONUS: {len(conus_alb)} counties | AK: {len(ak_alb)} | HI: {len(hi_alb)}")

    # ── palette（旧地图纸 / 探索档案质感）─────────────────────
    BG       = "#F4F0E8"   # 米白纸感底
    C_VIS    = "#B24A3B"   # 锈红 / 砖红（旅行档案感，非警报红）
    C_UNVIS  = "#D8D5CC"   # 未访问县：暖灰
    C_EDGE   = "#ECE8DF"   # 县界：极浅
    C_STATE  = "#A8A39A"   # 州界：稍深暖灰
    C_INK    = "#3A352E"   # 主文字：墨棕
    C_MUTED  = "#8A8275"   # 次要文字：灰褐
    FONT_MONO = "DejaVu Sans Mono"

    def _draw_poly(ax, poly, fc, lw):
        def ring_path(ring):
            c = np.array(ring.coords)
            codes = [Path.MOVETO] + [Path.LINETO]*(len(c)-2) + [Path.CLOSEPOLY]
            return c, codes
        verts, codes = [], []
        c, cd = ring_path(poly.exterior); verts.append(c); codes.extend(cd)
        for interior in poly.interiors:
            c, cd = ring_path(interior); verts.append(c); codes.extend(cd)
        path  = Path(np.concatenate(verts), codes)
        patch = mpatches.PathPatch(path, fc=fc, ec=C_EDGE, lw=lw, antialiased=True)
        ax.add_patch(patch)

    def draw_gdf(ax, gdf, idx_set, lw=0.35):
        for idx, row in gdf.iterrows():
            g = row.geometry
            if g is None or g.is_empty: continue
            fc = C_VIS if idx in idx_set else C_UNVIS
            if g.geom_type == "Polygon":
                _draw_poly(ax, g, fc, lw)
            elif g.geom_type == "MultiPolygon":
                for p in g.geoms:
                    _draw_poly(ax, p, fc, lw)

    def setup_ax(ax, gdf):
        ax.axis("off")
        ax.set_facecolor("white")
        if gdf is None or len(gdf) == 0:
            return False
        # drop invalid geometries before computing bounds
        valid = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty]
        if len(valid) == 0:
            return False
        bounds = valid.total_bounds   # [xmn, ymn, xmx, ymx]
        if np.any(~np.isfinite(bounds)):
            return False
        xmn, ymn, xmx, ymx = bounds
        px = max((xmx-xmn)*0.03, 1)
        py = max((ymx-ymn)*0.03, 1)
        ax.set_xlim(xmn-px, xmx+px)
        ax.set_ylim(ymn-py, ymx+py)
        ax.set_aspect("equal")
        return True

    # ── canvas（横向矩形，纸感底）────────────────────────────
    W, H, DPI = 7680, 5040, 500
    fig = plt.figure(figsize=(W/DPI, H/DPI), dpi=DPI, facecolor=BG)

    # ── build state outlines by dissolving counties ─────
    print("Building state outlines...")

    conus_alb2 = conus_alb.copy()
    conus_alb2["_sfips"] = conus_alb2.apply(_state_fips, axis=1)
    states_alb = conus_alb2.dissolve(by="_sfips")

    def draw_state_borders(ax, states_gdf, lw=1.6, color="#1a1a1a"):
        """Draw only interior state-to-state borders (not coastline/national border).
        Uses shapely: for each state, intersect its boundary with all other states.
        Only shared edges get drawn — outer coastline is excluded.
        """
        if states_gdf is None or len(states_gdf) == 0:
            return
        from shapely.ops import unary_union

        geoms = list(states_gdf.geometry)

        shared_lines = []
        for i, g1 in enumerate(geoms):
            if g1 is None or g1.is_empty:
                continue
            # boundary of this state
            b1 = g1.boundary
            # intersect with boundaries of all OTHER states
            for j, g2 in enumerate(geoms):
                if i >= j or g2 is None or g2.is_empty:
                    continue
                shared = b1.intersection(g2.boundary)
                if not shared.is_empty:
                    shared_lines.append(shared)

        def plot_line(line):
            if line.geom_type == "LineString":
                xs, ys = line.xy
                ax.plot(xs, ys, color=color, lw=lw, solid_capstyle="butt",
                        solid_joinstyle="round", zorder=5)
            elif line.geom_type in ("MultiLineString", "GeometryCollection"):
                for part in line.geoms:
                    if not part.is_empty:
                        plot_line(part)

        for line in shared_lines:
            plot_line(line)

    # ── 去过的州 + 州总数 ─────────────────────────────────
    import pandas as pd, math, numpy as np
    all_regions = pd.concat([conus_wgs, ak_wgs, hi_wgs]).copy()
    all_regions["_sfips"] = all_regions.apply(_state_fips, axis=1)
    visited_states = {all_regions.loc[i, "_sfips"] for i in visited_idx
                      if i in all_regions.index
                      and all_regions.loc[i, "_sfips"] in STATE_NAMES}
    visited_state_names = sorted(STATE_NAMES[sf] for sf in visited_states)
    total_states = len({sf for sf in all_regions["_sfips"] if sf in STATE_NAMES})
    total = len(counties_gdf); nvis = len(visited_idx); nst = len(visited_state_names)

    def spaced(s, gap=" "):
        return gap.join(list(s))

    # ── CONUS（左侧主体）────────────────────────────────────
    ax_main = fig.add_axes([0.0, 0.175, 0.72, 0.695])
    ax_main.axis("off"); ax_main.set_facecolor("none")
    ax_main.set_xlim(-2_400_000, 2_500_000); ax_main.set_ylim(-1_400_000, 1_650_000)
    ax_main.set_aspect("equal")
    draw_gdf(ax_main, conus_alb, visited_idx, lw=0.3)
    draw_state_borders(ax_main, states_alb, lw=0.6, color=C_STATE)

    # ── Alaska / Hawaii：带细边框的 inset box（右侧上下两格）─────
    def _bounds(gdf):
        b = gdf.total_bounds
        return b if (np.all(np.isfinite(b)) and b[2] > b[0]) else None
    ak_frame = None
    if len(ak_alb) > 0:
        core = ak_wgs[(ak_wgs.geometry.representative_point().x > -156)
                      | (ak_wgs.index.isin(visited_idx))]
        ak_frame = _bounds((core if len(core) else ak_wgs).to_crs(albers_ak))
    hi_frame = _bounds(hi_alb) if len(hi_alb) > 0 else None

    def _inset_box(frame, gdf, box, label, lw):
        x0, y0, bw, bh = box
        # 边框
        fr = fig.add_axes([x0, y0, bw, bh])
        fr.set_xticks([]); fr.set_yticks([]); fr.set_facecolor("none")
        for s in fr.spines.values():
            s.set_edgecolor(C_STATE); s.set_linewidth(1.1)
        fr.text(0.5, 0.90, spaced(label.upper()), transform=fr.transAxes,
                ha="center", va="center", fontsize=12, color=C_MUTED,
                fontfamily=FONT_REG)
        # 内部地图（留出标题空间），等比不变形
        inner = fig.add_axes([x0+bw*0.08, y0+bh*0.06, bw*0.84, bh*0.72])
        inner.axis("off"); inner.patch.set_visible(False)
        draw_gdf(inner, gdf, visited_idx, lw=lw)
        inner.set_xlim(frame[0], frame[2]); inner.set_ylim(frame[1], frame[3])
        inner.set_aspect("equal")

    if ak_frame is not None:
        _inset_box(ak_frame, ak_alb, [0.745, 0.50, 0.235, 0.305], "Alaska", 0.22)
    if hi_frame is not None:
        _inset_box(hi_frame, hi_alb, [0.745, 0.185, 0.235, 0.275], "Hawaii", 0.3)

    # ── 标题（左上，间隔大写）+ 副标题 ──────────────────────────
    fig.text(0.025, 0.955, spaced("AMERICAN") + "    " + spaced("FOOTPRINT"),
             ha="left", va="top", fontsize=33, fontweight="semibold",
             color=C_INK, fontfamily=FONT_SEMI)
    fig.text(0.027, 0.895, f"{nvis:,} counties   ·   {nst} states",
             ha="left", va="top", fontsize=17, color=C_MUTED, fontfamily=FONT_REG)

    # ── 铭牌式计数（右上，等宽，数字按列对齐）────────────────────
    np_line1 = f"{'COUNTIES':<9}{nvis:>5,} / {total:,}"
    np_line2 = f"{'STATES':<9}{nst:>5} / {total_states}"
    fig.text(0.745, 0.952, np_line1, ha="left", va="top",
             fontsize=18, color=C_INK, fontfamily=FONT_MONO)
    fig.text(0.745, 0.905, np_line2, ha="left", va="top",
             fontsize=18, color=C_INK, fontfamily=FONT_MONO)

    # ── 图例（克制：小色块 + 小写注释）──────────────────────────
    ax_leg = fig.add_axes([0.03, 0.115, 0.30, 0.045]); ax_leg.axis("off")
    ax_leg.set_xlim(0, 1); ax_leg.set_ylim(0, 1)
    ax_leg.add_patch(mpatches.Rectangle((0.0, 0.3), 0.045, 0.42, fc=C_VIS, ec="none"))
    ax_leg.text(0.065, 0.5, "visited", va="center", ha="left",
                fontsize=14, color=C_MUTED, fontfamily=FONT_REG)
    ax_leg.add_patch(mpatches.Rectangle((0.27, 0.3), 0.045, 0.42, fc=C_UNVIS, ec="none"))
    ax_leg.text(0.335, 0.5, "not yet", va="center", ha="left",
                fontsize=14, color=C_MUTED, fontfamily=FONT_REG)

    # ── 州名注释块（左下，小字灰，像地图注记）────────────────────
    if visited_state_names:
        per_line = 9
        nlines = max(1, math.ceil(nst / per_line))
        per_line = math.ceil(nst / nlines)
        sep = "  ·  "
        lines = [sep.join(visited_state_names[i:i+per_line]) for i in range(0, nst, per_line)]
        body = "\n".join(lines)
        fig.text(0.03, 0.088, spaced("STATES VISITED"), ha="left", va="top",
                 fontsize=12, color=C_MUTED, fontfamily=FONT_REG)
        fig.text(0.03, 0.057, body, ha="left", va="top",
                 fontsize=13, linespacing=1.65, color=C_MUTED, fontfamily=FONT_REG)

    print(f"Saving image ({W}x{H} @ {DPI}dpi)...")
    plt.savefig(output_path, dpi=DPI,
                facecolor=BG, edgecolor="none")
    plt.close()
    print(f"\n✅ Saved: {output_path}")
    return output_path

# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="世界迷雾 .fwss → 美国县级探索地图"
    )
    parser.add_argument("fwss", help=".fwss 快照文件路径")
    parser.add_argument("-o", "--output", default="fow_usa_counties.png",
                        help="输出图片路径（默认 fow_usa_counties.png）")
    parser.add_argument("--counties", default="us_counties.geojson",
                        help="县级 GeoJSON 缓存路径（第一次运行会自动下载）")
    args = parser.parse_args()

    if not os.path.exists(args.fwss):
        print(f"❌ 找不到文件: {args.fwss}")
        sys.exit(1)

    # 1. 解析 .fwss
    points = extract_visited_points(args.fwss)
    if not points:
        print("❌ 没有解析到任何 visited 坐标点")
        sys.exit(1)

    # 2. 获取县级边界
    geojson_path = download_counties_geojson(args.counties)

    import geopandas as gpd, json
    print("Loading county boundaries...")
    # 用二进制读取，自动检测编码，支持 GeoJSON 和 TopoJSON
    with open(geojson_path, "rb") as f:
        raw_bytes = f.read()
    # 尝试常见编码
    for enc in ("utf-8", "utf-8-sig", "latin-1", "cp1252"):
        try:
            raw = json.loads(raw_bytes.decode(enc))
            break
        except (UnicodeDecodeError, json.JSONDecodeError):
            continue
    else:
        raise RuntimeError(f"无法解析 {geojson_path}，请删除该文件后重新运行")

    if raw.get("type") == "Topology":
        print("  检测到 TopoJSON 格式，自动转换...")
        raw = _topojson_to_geojson(raw, "counties")
        tmp_geojson = geojson_path + ".converted.geojson"
        with open(tmp_geojson, "w", encoding="utf-8") as f:
            json.dump(raw, f)
        geojson_path = tmp_geojson
    counties = gpd.read_file(geojson_path)
    # 修复任何残余的无效几何
    counties["geometry"] = counties["geometry"].buffer(0)
    counties = counties[counties["geometry"].notna() & ~counties["geometry"].is_empty].reset_index(drop=True)
    print(f"  {len(counties)} counties, CRS: {counties.crs}")

    # 3. 空间叠加
    visited_idx = find_visited_counties(points, counties)

    # 4. 绘图
    plot_map(counties, visited_idx, args.output)


if __name__ == "__main__":
    main()
