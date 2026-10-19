"""Vector A4 tags, camera target, dimensions and assembly instructions.

Requires the print and vision extras. All tag dimensions come from the same
definition as PnP and simulation. No raster scaling or browser printing needed.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas

from orca_feetech.objects import DICTIONARY, OBJECT_VERSION, marker_layout, specification, pen_dimensions


def generate(output):
    d = pen_dimensions()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    font_path = Path("/System/Library/Fonts/Supplemental/Arial Unicode.ttf")
    if font_path.exists():
        pdfmetrics.registerFont(TTFont("Guide", str(font_path)))
    else:
        pdfmetrics.registerFont(UnicodeCIDFont("STSong-Light"))
        pdfmetrics.registerFontFamily("Guide", normal="STSong-Light")
    face = "Guide" if font_path.exists() else "STSong-Light"
    c = canvas.Canvas(str(output / "print-kit-a4.pdf"), pagesize=A4, pageCompression=1)
    c.setTitle(f"ORCA {OBJECT_VERSION} / A1 mini PLA / 1:1 print kit")
    dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, DICTIONARY))
    manifest = dict(version=OBJECT_VERSION, units="mm", pdf="print-kit-a4.pdf",
                    page_sizes_mm=[], tags=[], print_scale_percent=100,
                    chessboard=dict(page=4, inner_corners=[9, 6], square_mm=20))

    def text(x, y, content, size=10):
        c.setFillColorRGB(0, 0, 0)
        c.setFont(face, size)
        c.drawString(x*mm, y*mm, content)

    def header(number, title, subtitle, wide=False):
        w, h = landscape(A4) if wide else A4
        c.setPageSize((w, h))
        manifest["page_sizes_mm"].append([w/mm, h/mm])
        text(12, h/mm-16, title, 17)
        text(12, h/mm-24, subtitle, 9)
        text(12, 9, f"ORCA {OBJECT_VERSION} | {number}/6 | A4 | 100% Actual size | NO Fit/Shrink", 8)

    def ruler(x=15, y=26):
        c.setStrokeColorRGB(0, 0, 0)
        c.setLineWidth(.2*mm)
        c.line(x*mm, y*mm, (x+100)*mm, y*mm)
        for i in range(11):
            height = 3 if i % 5 == 0 else 1.5
            c.line((x+i*10)*mm, (y-height)*mm, (x+i*10)*mm, (y+height)*mm)
        text(x, y+6, "打印后用直尺确认此线正好 100 mm；不符时不要粘贴。", 9)

    def tag(kind, item, page, x, y, caption):
        size, paper = item["black_size_m"]*1000, item["paper_size_m"]*1000
        bits = cv2.aruco.generateImageMarker(dictionary, item["id"], 8)
        c.setFillColorRGB(1, 1, 1)
        c.rect(x*mm, y*mm, paper*mm, paper*mm, fill=1, stroke=0)
        c.setStrokeColorRGB(.65, .65, .65)
        c.setDash(1*mm, 1*mm)
        c.setLineWidth(.15*mm)
        c.rect(x*mm, y*mm, paper*mm, paper*mm, fill=0, stroke=1)
        c.setDash()
        offset = (paper-size)/2
        c.setFillColorRGB(0, 0, 0)
        for iy, ix in zip(*np.where(bits == 0)):
            c.rect((x+offset+ix*size/8)*mm, (y+offset+(7-iy)*size/8)*mm,
                   size/8*mm, size/8*mm, fill=1, stroke=0)
        text(x, y+paper+6, f"ID {item['id']} | {caption}", 8)
        # Arrow lies outside the crop boundary and indicates the decoded top.
        ax, ay = x+paper/2, y+paper+2
        c.line(ax*mm, ay*mm, ax*mm, (ay+2.5)*mm)
        c.line(ax*mm, (ay+2.5)*mm, (ax-1)*mm, (ay+1.5)*mm)
        c.line(ax*mm, (ay+2.5)*mm, (ax+1)*mm, (ay+1.5)*mm)
        manifest["tags"].append(dict(kind=kind, id=item["id"], page=page,
                                     paper_lower_left_mm=[x, y], paper_size_mm=paper,
                                     black_size_mm=size, black_lower_left_mm=[x+offset, y+offset]))

    header(1, "01 / 方块标签：55 mm 方块", "AprilTag 36h11 | 黑框 32 mm | 含白边纸片 40 mm | 每面中央一张")
    names = ("+X RIGHT", "-X LEFT", "+Y BACK", "-Y FRONT", "+Z TOP", "-Z BOTTOM")
    for i, item in enumerate(marker_layout("cube")):
        tag("cube", item, 1, 20+(i % 3)*60, 190-(i//3)*65, names[i])
    text(15, 94, "沿虚线裁切，保留完整白边；编号、箭头和说明留在纸外。")
    text(15, 83, "使用哑光白纸或不透底的哑光贴纸，不覆反光透明胶。")
    text(15, 72, "粘贴方向见第 6 页；黑框外边长才是位姿求解使用的尺寸。")
    ruler()
    c.showPage()

    header(2, "02 / 训练笔两端标签", "总长 210 mm | 杆径 16 mm | 黑框 16 mm | 含白边纸片 20 mm")
    by_id = {item["id"]: item for item in marker_layout("pen")}
    for row, ids in enumerate(((10, 12, 14, 16, 18), (11, 13, 15, 17, 19))):
        text(15, 243-row*65, "负端 Y-" if row == 0 else "正端 Y+", 12)
        for col, ident in enumerate(ids):
            caption = ("+X", "+Z", "-X", "-Z", "END")[col]
            tag("pen", by_id[ident], 2, 15+col*37, 204-row*65, caption)
    text(15, 101, "侧面码顶边统一朝物体 +Y；不是各自朝外，也不是都朝圆杆。")
    text(15, 90, "端面码顶边统一朝 +Z。先定义坐标方向，再按第 6 页贴码。")
    text(15, 79, "纸片在 24 x 22 mm 侧面中央：宽方向各留 2 mm，轴向各留 1 mm。")
    text(15, 68, "端面纸片居中，四边各留 2 mm。圆杆与渐变肩部均不贴码。")
    ruler()
    c.showPage()

    header(3, "03 / 固定世界参考码", "AprilTag 36h11 | ID 40 | 黑框 50 mm | 白纸 66 mm")
    tag("world", dict(id=40, black_size_m=.05, paper_size_m=.066), 3, 72, 153, "WORLD +Y UP")
    text(15, 120, "贴在平整、刚性的独立板上，固定在手旁；不要贴在活动手指上。")
    text(15, 109, "码中心为世界原点，纸面向右为 +X，向上为 +Y，垂直纸面向外为 +Z。")
    text(15, 98, "使用现有 align-world 工具实测它与仿真坐标之间的刚体变换。")
    text(15, 87, "移动手、参考板或相机后，重新核验标定；此纸片不提供实测外参。")
    ruler()
    c.showPage()

    header(4, "04 / 相机内参棋盘格", "20 mm 方格 | 10 x 7 格 | 9 x 6 内角点 | A4 横向、100% 打印", wide=True)
    x0, y0 = 48.5, 35
    c.setFillColorRGB(1, 1, 1)
    c.rect(x0*mm, y0*mm, 200*mm, 140*mm, fill=1, stroke=0)
    c.setFillColorRGB(0, 0, 0)
    for row in range(7):
        for col in range(10):
            if (row+col) % 2 == 0:
                c.rect((x0+col*20)*mm, (y0+row*20)*mm, 20*mm, 20*mm, fill=1, stroke=0)
    text(48.5, 23, "贴在刚性平板上。至少拍摄 15 张不同位置、距离和倾角的清晰照片。", 9)
    text(48.5, 17, "标定参数：--cols 9 --rows 6 --square-m 0.02；焦距和分辨率与部署保持一致。", 9)
    c.showPage()

    header(5, "05 / 实体尺寸与打印摆放", "本页为说明图；可打印模板在第 1-4 页。全部实体按 mm / 100% 导入。")
    text(15, 248, "方块 55 x 55 x 55 mm", 13)
    # Diagram is true size; sticker fit can also be checked with a caliper.
    c.setStrokeColorRGB(0, 0, 0)
    c.setLineWidth(.3*mm)
    c.rect(20*mm, 177*mm, 55*mm, 55*mm)
    c.setDash(1*mm, 1*mm)
    c.rect(27.5*mm, 184.5*mm, 40*mm, 40*mm)
    c.setDash()
    text(20, 168, "纸片居中，四边余量 7.5 mm", 9)
    text(15, 150, "笔：210 / Ø16 / 方头宽24 mm", 12)
    # Axial drawing, scale 1:1, with heads and shoulders matching canonical CAD.
    end, head, shaft = [d[k]*1000 for k in ("end_m", "head_start_m", "shaft_end_m")]
    polygon = [(0, -end), (12, -end), (12, -head), (8, -shaft), (8, shaft),
               (12, head), (12, end), (-12, end), (-12, head), (-8, shaft),
               (-8, -shaft), (-12, -head), (-12, -end), (0, -end)]
    path = c.beginPath()
    for i, (x, y) in enumerate(polygon):
        (path.moveTo if i == 0 else path.lineTo)((153+x)*mm, (146+y)*mm)
    c.drawPath(path, fill=0, stroke=1)
    for y, content in ((235, "端头长22"), (209, "肩部10"), (148, "圆杆长146"), (57, "端头长22")):
        text(170, y, content, 8)
    for y, content in ((136, "默认：两半沿接缝平面朝下"), (124, "每盘1半壳+3销；同工程打两次"),
                       (112, "单件高12 mm；无需外部支撑"), (100, "4层壁 / 20% gyroid / 层高0.2"),
                       (88, "顶部5层 / 底部5层 / brim 3 mm"), (76, "先试装，再薄胶粘合并夹紧"),
                       (64, "不要用力压入定位销或留胶凸起")):
        text(15, y, content, 9)
    text(15, 43, "方块：平面朝下，3层壁，20% gyroid，顶/底各5层。不要擅自填满或加配重。", 9)
    text(15, 33, "每片210 x 24 x 12 mm，床面转45°。两片同形，第二盘的3根定位销作为备用。", 9)
    text(15, 23, "胶固化后测量直径、总长、方头与接缝台阶；称带纸成品，并输入 --mass-kg。", 9)
    c.showPage()

    header(6, "06 / 粘贴坐标与装配检查", "贴纸图案不镜像。先选定实物坐标，侧面和端面 ID 不能互换。")
    text(15, 249, "方块：+Z 是顶面，+X 是右面，+Y 是后面。", 12)
    cube_rows = (("0 / 右面 +X", "+Z / 朝顶面"), ("1 / 左面 -X", "+Z / 朝顶面"),
                 ("2 / 后面 +Y", "+Z / 朝顶面"), ("3 / 前面 -Y", "+Z / 朝顶面"),
                 ("4 / 顶面 +Z", "+Y / 朝后面"), ("5 / 底面 -Z", "+Y / 朝后面"))
    for i, (name, top) in enumerate(cube_rows):
        text(20, 235-i*9, f"ID {name}       纸片顶边：{top}", 10)
    text(15, 168, "笔：轴向从负端指向正端为 +Y，选一对平面为 +Z / -Z。", 11)
    text(15, 156, "负端侧码中心 Y=-94 mm；正端侧码中心 Y=+94 mm，基线188 mm。", 10)
    rows = (("+X 面", "10", "11"), ("+Z 面", "12", "13"), ("-X 面", "14", "15"), ("-Z 面", "16", "17"))
    text(20, 142, "同一朝向平面       负端 ID       正端 ID       所有码顶边", 10)
    for i, (name, neg, pos) in enumerate(rows):
        for x, content in ((20, name), (66, neg), (98, pos), (133, "朝 +Y")):
            text(x, 131-i*10, content, 10)
    text(15, 88, "负端外端面 ID18，正端外端面 ID19；两个端面码顶边均朝 +Z。", 10)
    text(15, 77, "接缝为 Z=0 平面：两片同形半壳，3根 Ø4 x 6 mm 销定位，薄胶粘接。", 10)
    text(15, 66, "孔 Ø4.3、单边深3.15 mm。PLA公差须试装；去毛边，不强压，不加外露胶。", 10)
    text(15, 50, "成品记录：总长____mm  杆径____mm  方头____mm  方块边长____mm", 10)
    text(15, 39, "贴码后质量____g  接缝/翘曲____mm  纸尺100mm实测____mm  日期________", 10)
    text(15, 25, "先静态检码、核对方向和位姿，再做 dry-run。此手册不代表实打或真机迁移已验证。", 9)
    c.showPage()
    c.save()
    manifest["objects"] = {kind: dict(dimensions_mm=(np.array(specification(kind)["dimensions_m"])*1000).tolist(),
                                     tags=marker_layout(kind)) for kind in ("cube", "pen")}
    (output / "print-manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("assets/objects/v4"))
    generate(parser.parse_args().output)
