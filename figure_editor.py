import streamlit as st
import cv2
import numpy as np
from PIL import Image
from streamlit_drawable_canvas import st_canvas
import io
import os

# 1. 必须放在第一行
st.set_page_config(layout="wide", page_title="科研作图放大镜")

# ==========================================
# 核心逻辑类
# ==========================================
class FigureProcessor:
    @staticmethod
    def read_image(path):
        try:
            img_array = np.fromfile(path, dtype=np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_UNCHANGED)
            if img is None: return None
            # 统一转为 BGRA (带透明通道，方便处理)
            if len(img.shape) == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGRA)
            elif len(img.shape) == 3:
                if img.shape[2] == 3:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
                elif img.shape[2] == 4:
                    pass # 已经是BGRA
            return img
        except Exception as e:
            print(f"读取错误: {e}")
            return None

    @staticmethod
    def apply_magnifier(img, roi_x, roi_y, roi_w, roi_h, scale, pos_type, custom_x, custom_y, shape_type="rectangle"):
        h, w, c = img.shape
        rx, ry, rw, rh = int(roi_x), int(roi_y), int(roi_w), int(roi_h)
        
        # 边界保护
        rx = max(0, min(rx, w - 1))
        ry = max(0, min(ry, h - 1))
        rw = max(1, min(rw, w - rx))
        rh = max(1, min(rh, h - ry))

        if rw < 5 or rh < 5: return img

        roi = img[ry:ry+rh, rx:rx+rw]
        if roi.size == 0: return img

        inset_w, inset_h = int(rw * scale), int(rh * scale)
        roi_zoomed = cv2.resize(roi, (inset_w, inset_h), interpolation=cv2.INTER_LINEAR)

        # 制作背景
        white_bg = np.full((inset_h, inset_w, 4), 255, dtype=np.uint8)
        
        # 处理 Alpha
        if roi_zoomed.shape[2] == 4:
            alpha = roi_zoomed[:, :, 3].astype(float) / 255.0
        else:
            alpha = np.ones((inset_h, inset_w), dtype=float)

        if shape_type == "circle":
            circle_mask = np.zeros((inset_h, inset_w), dtype=float)
            center, axes = (inset_w // 2, inset_h // 2), (inset_w // 2, inset_h // 2)
            cv2.ellipse(circle_mask, center, axes, 0, 0, 360, 1.0, -1)
            alpha *= circle_mask

        # 混合
        for i in range(3):
            white_bg[:, :, i] = (1.0 - alpha) * white_bg[:, :, i] + alpha * roi_zoomed[:, :, i]
        white_bg[:, :, 3] = 255
        final_inset = white_bg

        # 位置计算
        pad = 20
        pos_map = {
            "右上角": (w - inset_w - pad, pad),
            "右下角": (w - inset_w - pad, h - inset_h - pad),
            "左上角": (pad, pad),
            "左下角": (pad, h - inset_h - pad),
        }
        ix, iy = pos_map.get(pos_type, (int(custom_x), int(custom_y)))
        ix = max(0, min(ix, w - inset_w))
        iy = max(0, min(iy, h - inset_h))

        # 绘图参数
        COLOR_LINE = (50, 50, 50, 255)
        COLOR_ROI = (0, 0, 255, 255)
        COLOR_BORDER = (0, 0, 0, 255)
        THICKNESS = 2

        # 绘制连接线
        if iy < ry:
            pt1_roi, pt2_roi = (rx, ry), (rx+rw, ry)
            pt1_ins, pt2_ins = (ix, iy+inset_h), (ix+inset_w, iy+inset_h)
        else:
            pt1_roi, pt2_roi = (rx, ry+rh), (rx+rw, ry+rh)
            pt1_ins, pt2_ins = (ix, iy), (ix+inset_w, iy)
        
        cv2.line(img, pt1_roi, pt1_ins, COLOR_LINE, THICKNESS, cv2.LINE_AA)
        cv2.line(img, pt2_roi, pt2_ins, COLOR_LINE, THICKNESS, cv2.LINE_AA)

        # 绘制 ROI
        if shape_type == "circle":
            cv2.ellipse(img, (rx + rw // 2, ry + rh // 2), (rw // 2, rh // 2), 0, 0, 360, COLOR_ROI, THICKNESS, cv2.LINE_AA)
        else:
            cv2.rectangle(img, (rx, ry), (rx + rw, ry + rh), COLOR_ROI, THICKNESS)

        # 贴图
        img[iy:iy+inset_h, ix:ix+inset_w] = final_inset
        cv2.rectangle(img, (ix, iy), (ix + inset_w, iy + inset_h), COLOR_BORDER, THICKNESS)

        return img

# ==========================================
# UI 辅助
# ==========================================
def initialize_state():
    defaults = {
        'roi_x': 100, 'roi_y': 100, 'roi_w': 100, 'roi_h': 100,
        'last_click_pos': None,
        'canvas_key': 0 
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

def setup_control_sidebar(img_w, img_h):
    with st.sidebar:
        st.divider()
        st.header("2. 放大镜控制")
        enable_zoom = st.checkbox("✅ 启用放大镜", value=True)
        shape_type = st.radio("ROI 形状", ["rectangle", "circle"], horizontal=True,
                              format_func=lambda x: "⬜ 方" if x == "rectangle" else "⭕ 圆")
        
        c_p1, c_p2 = st.columns(2)
        inset_pos = c_p1.selectbox("放置角", ["右上角", "右下角", "左上角", "左下角", "自定义"])
        inset_scale = c_p2.slider("倍数", 1.5, 10.0, 3.0)
        
        custom_x, custom_y = 0, 0
        if inset_pos == "自定义":
            c_cust1, c_cust2 = st.columns(2)
            custom_x = c_cust1.number_input("Pos X", 0, img_w, 500)
            custom_y = c_cust2.number_input("Pos Y", 0, img_h, 50)
        
        st.divider()
        st.header("3. ROI 精确控制")
        st.info("👉 **操作**: 选 **'⬜ 选区'** 工具，在图上画框，**松手自动更新**。")

        col_xy1, col_xy2 = st.columns(2)
        st.session_state.roi_x = col_xy1.number_input("ROI X", 0, img_w, st.session_state.roi_x)
        st.session_state.roi_y = col_xy2.number_input("ROI Y", 0, img_h, st.session_state.roi_y)
        col_wh1, col_wh2 = st.columns(2)
        st.session_state.roi_w = col_wh1.number_input("ROI 宽", 10, img_w, st.session_state.roi_w)
        st.session_state.roi_h = col_wh2.number_input("ROI 高", 10, img_h, st.session_state.roi_h)

    return enable_zoom, shape_type, inset_pos, inset_scale, custom_x, custom_y

def handle_image_loading():
    uploaded = st.session_state.get('uploaded_file')
    path = st.session_state.get('img_path')
    if uploaded:
        file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_UNCHANGED)
        if img is not None:
            # 确保上传的图也是 BGRA，方便后续合并
            if len(img.shape) == 2: return cv2.cvtColor(img, cv2.COLOR_GRAY2BGRA)
            if img.shape[2] == 3: return cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
            return img
    elif path and os.path.exists(path):
        return FigureProcessor.read_image(path)
    return None

# ==========================================
# 主程序
# ==========================================
def main():
    st.title("🔬 科研作图放大镜 (下载修复版)")
    initialize_state()

    # 1. 侧边栏上传
    with st.sidebar:
        st.header("1. 图片源")
        upload_tab, path_tab = st.tabs(["上传", "路径"])
        with upload_tab:
            st.file_uploader("PNG/JPG", type=["png", "jpg", "jpeg"], key="uploaded_file")
        with path_tab:
            st.text_input("本地路径", key="img_path")

    # 2. 读取图片
    original_bgra = handle_image_loading()
    
    if original_bgra is None:
        st.warning("👈 请先加载图片")
        return

    orig_h, orig_w, _ = original_bgra.shape

    # 3. 参数控制
    (enable_zoom, shape_type, inset_pos, 
     inset_scale, custom_x, custom_y) = setup_control_sidebar(orig_w, orig_h)
    
    # 4. 显示比例计算
    MAX_W = 1000
    display_scale = MAX_W / orig_w if orig_w > MAX_W else 1.0
    disp_w, disp_h = int(orig_w * display_scale), int(orig_h * display_scale)

    # 5. 【后台处理】生成底图 (带放大镜)
    processed_bgra = original_bgra.copy()
    if enable_zoom:
        processed_bgra = FigureProcessor.apply_magnifier(
            processed_bgra, 
            st.session_state.roi_x, st.session_state.roi_y,
            st.session_state.roi_w, st.session_state.roi_h,
            inset_scale, inset_pos, custom_x, custom_y, shape_type
        )
    
    # 转为 PIL 对象，用于显示和后续合成
    base_image_pil = Image.fromarray(cv2.cvtColor(processed_bgra, cv2.COLOR_BGRA2RGBA)).convert("RGBA")

    # 6. 【画布渲染】
    st.markdown("---")
    c1, c2 = st.columns([1, 5])
    with c1:
        st.markdown("##### ✏️ 工具")
        tool = st.radio("模式", ("rect", "transform", "line"), 
                       format_func=lambda x: {"transform":"✋ 移动/取点","line":"↗️ 箭头","rect":"⬜ 选区"}.get(x,x))
        stroke_c = st.color_picker("颜色", "#FF0000")
        
        if st.button("🧹 重置/清空"):
            st.session_state.canvas_key += 1
            st.rerun()

    with c2:
        # 画布只显示，不保存底图数据，只返回手绘数据
        cvs_res = st_canvas(
            fill_color="rgba(0,0,0,0)",
            stroke_width=2,
            stroke_color=stroke_c,
            background_image=base_image_pil, # 这里显示的是处理好的图
            update_streamlit=True,
            height=disp_h, width=disp_w,
            drawing_mode=tool,
            initial_drawing=None,
            key=f"canvas_{st.session_state.canvas_key}" 
        )

    # 7. 【交互逻辑】
    if cvs_res.json_data and cvs_res.json_data["objects"]:
        last_obj = cvs_res.json_data["objects"][-1]
        
        # A. 画框选区 -> 更新ROI
        if tool == 'rect' and last_obj['type'] == 'rect':
            new_x = int(last_obj['left'] / display_scale)
            new_y = int(last_obj['top'] / display_scale)
            new_w = int(last_obj['width'] * last_obj['scaleX'] / display_scale)
            new_h = int(last_obj['height'] * last_obj['scaleY'] / display_scale)
            
            if new_w > 5 and new_h > 5:
                st.session_state.roi_x = new_x
                st.session_state.roi_y = new_y
                st.session_state.roi_w = new_w
                st.session_state.roi_h = new_h
                st.session_state.canvas_key += 1
                st.rerun()

        # B. 点击 -> 取点
        elif tool == 'transform':
             click_x = int(last_obj['left'] / display_scale)
             click_y = int(last_obj['top'] / display_scale)
             st.session_state.last_click_pos = (click_x, click_y)

    # 8. 【修复版下载逻辑】
    st.markdown("### 💾 导出图片")
    
    col_dl_name, col_dl_btn = st.columns([2, 1])
    
    with col_dl_name:
        # 自定义文件名输入框
        fname = st.text_input("文件名 (无需后缀)", value="result_image")
    
    with col_dl_btn:
        st.write("") # 占位对齐
        st.write("") 
        
        # 🟢 核心修复逻辑：图层融合
        # 1. 拿到底图 (base_image_pil) -> 包含原图 + 放大镜
        # 2. 拿到画布手绘层 (cvs_res.image_data) -> 包含你画的箭头等
        # 3. 合并它们
        
        final_pil = base_image_pil # 默认就是底图

        if cvs_res.image_data is not None:
            # 获取手绘层
            annot_pil = Image.fromarray(cvs_res.image_data.astype('uint8'), 'RGBA')
            
            # 尺寸对齐 (防止手绘层和底图尺寸微小差异)
            if annot_pil.size != base_image_pil.size:
                annot_pil = annot_pil.resize(base_image_pil.size, Image.Resampling.LANCZOS)
            
            # 使用 Alpha Composite 完美融合
            final_pil = Image.alpha_composite(base_image_pil, annot_pil)

        # 4. 转为字节流下载
        # PIL RGBA -> PNG Bytes
        buf = io.BytesIO()
        final_pil.save(buf, format="PNG")
        byte_data = buf.getvalue()

        file_fullname = f"{fname}.png"
        
        st.download_button(
            label="⬇️ 下载最终 PNG",
            data=byte_data,
            file_name=file_fullname,
            mime="image/png",
            type="primary"
        )

if __name__ == "__main__":
    main()