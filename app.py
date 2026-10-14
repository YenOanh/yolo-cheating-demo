import glob
import pandas as pd
import streamlit as st
from PIL import Image
from ultralytics import YOLO

st.set_page_config(page_title="Phát hiện gian lận bằng YOLOv10", layout="wide")

ALERT = {"Cheating", "Cheating-Paper", "Phone"}   # tự chỉnh theo ý nghĩa lớp
COLS = ["Lớp", "Độ tin cậy", "x1", "y1", "x2", "y2"]

@st.cache_resource
def load_model():
    return YOLO("best.pt")

model = load_model()
NAMES = model.names

st.title("Phát hiện hành vi gian lận trong phòng thi (YOLOv10)")

with st.sidebar:
    st.header("Thiết lập")
    conf = st.slider("Ngưỡng tin cậy (conf)", 0.1, 0.9, 0.25, 0.05)
    iou = st.slider("Ngưỡng IoU (NMS)", 0.3, 0.9, 0.7, 0.05)
    source = st.radio("Nguồn ảnh", ["Tải ảnh lên", "Ảnh mẫu"])

img = None
if source == "Tải ảnh lên":
    up = st.file_uploader("Chọn ảnh", type=["jpg", "jpeg", "png"])
    if up:
        img = Image.open(up).convert("RGB")
else:
    files = sorted(glob.glob("*.jpg") + glob.glob("examples/*"))
    if files:
        choice = st.selectbox("Chọn ảnh mẫu", files)
        img = Image.open(choice).convert("RGB")
    else:
        st.info("Chưa có ảnh mẫu trong thư mục examples/.")

if img is not None:
    r = model.predict(img, conf=conf, iou=iou, imgsz=640, verbose=False)[0]
    rows = []
    for b in r.boxes:
        x1, y1, x2, y2 = [round(v) for v in b.xyxy[0].tolist()]
        rows.append([NAMES[int(b.cls)], round(float(b.conf), 3), x1, y1, x2, y2])
    df = pd.DataFrame(rows, columns=COLS)

    c1, c2 = st.columns(2)
    c1.subheader("Ảnh đầu vào")
    c1.image(img, use_container_width=True)
    c2.subheader("Kết quả")
    c2.image(r.plot(), channels="BGR", use_container_width=True)

    if df.empty:
        st.info("Không phát hiện đối tượng nào.")
    else:
        counts = df["Lớp"].value_counts()
        flagged = [c for c in counts.index if c in ALERT]
        if flagged:
            st.warning("⚠️ Nghi ngờ gian lận: " + ", ".join(flagged))
        else:
            st.success("✅ Chưa thấy dấu hiệu gian lận")
        st.caption("Số lượng: " + ", ".join(f"{k}: {v}" for k, v in counts.items()))
        st.subheader("Bảng chi tiết các đối tượng phát hiện")
        st.dataframe(df, use_container_width=True)
