"""Single-process Streamlit face-attendance application.

Run with: streamlit run app.py
"""
from __future__ import annotations

import hashlib
import os
import re
from datetime import datetime
from pathlib import Path

import cv2
import face_recognition
import numpy as np
import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "dataset"
STUDENTS_FILE = BASE_DIR / "students.csv"
ATTENDANCE_FILE = BASE_DIR / "attendance.csv"
STUDENT_COLUMNS = ["Name", "RegNo", "Gender", "Age"]
ATTENDANCE_COLUMNS = ["Timestamp", "Name", "Status"]
FACE_CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


def ensure_storage() -> None:
    """Create the two small CSV stores only when they do not exist."""
    DATASET_DIR.mkdir(exist_ok=True)
    for path, columns in ((STUDENTS_FILE, STUDENT_COLUMNS), (ATTENDANCE_FILE, ATTENDANCE_COLUMNS)):
        if not path.exists() or path.stat().st_size == 0:
            pd.DataFrame(columns=columns).to_csv(path, index=False)


def students() -> pd.DataFrame:
    return pd.read_csv(STUDENTS_FILE, dtype={"RegNo": str})


def attendance() -> pd.DataFrame:
    return pd.read_csv(ATTENDANCE_FILE)


def today_logs(logs: pd.DataFrame) -> pd.DataFrame:
    today = datetime.now().strftime("%Y-%m-%d")
    return logs[logs["Timestamp"].astype(str).str.startswith(today)].copy()


def safe_folder_name(name: str, reg_no: str) -> str:
    """Avoid using unchecked user input as a path while keeping folders readable."""
    readable = re.sub(r"[^A-Za-z0-9_-]+", "_", name.strip()).strip("_")
    return f"{readable}_{reg_no.strip()}"


def image_from_upload(uploaded_file) -> np.ndarray:
    data = np.frombuffer(uploaded_file.getvalue(), np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("The camera image could not be decoded.")
    return image


def crop_single_face(image: np.ndarray) -> np.ndarray | None:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = FACE_CASCADE.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(80, 80))
    if len(faces) != 1:
        return None
    x, y, w, h = faces[0]
    padding = int(min(w, h) * 0.15)
    return image[max(0, y-padding):y+h+padding, max(0, x-padding):x+w+padding]


def dataset_signature() -> tuple[tuple[str, int, int], ...]:
    return tuple(
        sorted((str(p.relative_to(DATASET_DIR)), p.stat().st_mtime_ns, p.stat().st_size)
               for p in DATASET_DIR.rglob("*.jpg"))
    )


@st.cache_data(show_spinner="Loading enrolled face profiles…")
def known_faces(_signature: tuple[tuple[str, int, int], ...]):
    encodings, names = [], []
    for relative_path, _, _ in _signature:
        path = DATASET_DIR / relative_path
        try:
            found = face_recognition.face_encodings(face_recognition.load_image_file(path))
            if found:
                encodings.append(found[0])
                names.append(path.parent.name)
        except Exception:
            continue
    return encodings, names


def mark_present(folder_names: list[str]) -> list[str]:
    if not folder_names:
        return []
    logs = attendance()
    day = datetime.now().strftime("%Y-%m-%d")
    existing = set(logs.loc[
        logs["Timestamp"].astype(str).str.startswith(day) & (logs["Status"] == "Present"), "Name"
    ])
    new_names = [name for name in dict.fromkeys(folder_names) if name not in existing]
    if new_names:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        logs = pd.concat([logs, pd.DataFrame({"Timestamp": timestamp, "Name": new_names, "Status": "Present"})], ignore_index=True)
        logs.to_csv(ATTENDANCE_FILE, index=False)
    return new_names


def display_name(folder_name: str, registry: pd.DataFrame) -> str:
    match = registry[registry["Folder"] == folder_name]
    return match.iloc[0]["Name"] if not match.empty else folder_name


def app() -> None:
    ensure_storage()
    st.set_page_config(page_title="FaceAttend AI", page_icon="🟢", layout="wide")
    st.markdown("""<style>
    .stApp { background:#0d0f14; color:#e8eaf0; } [data-testid='stMetric'] {
    background:#1a1c28; border:1px solid #252a38; padding:18px; border-radius:12px; }
    </style>""", unsafe_allow_html=True)
    st.title("🟢 FaceAttend AI System")
    page = st.sidebar.radio("Navigation", ["Dashboard", "Take Attendance", "Enroll Student", "Records"])
    registry = students()
    registry["Folder"] = registry.apply(lambda r: safe_folder_name(str(r["Name"]), str(r["RegNo"])), axis=1) if not registry.empty else []
    logs = attendance()

    if page == "Dashboard":
        today = today_logs(logs)
        present_folders = set(today.loc[today["Status"] == "Present", "Name"])
        st.subheader("Overview")
        a, b, c = st.columns(3)
        a.metric("Enrolled students", len(registry))
        b.metric("Present today", len(present_folders))
        c.metric("Absent today", max(0, len(registry) - len(present_folders)))
        if registry.empty:
            st.info("Enroll a student to start building the recognition dataset.")
        else:
            rate = len(present_folders) / len(registry) * 100
            st.info(f"Today’s attendance is {rate:.0f}% — {len(registry) - len(present_folders)} student(s) have not checked in.")
        left, right = st.columns(2)
        with left:
            st.subheader("Today’s check-ins")
            shown = today.sort_values("Timestamp", ascending=False).copy()
            shown["Name"] = shown["Name"].map(lambda x: display_name(x, registry))
            st.dataframe(shown, hide_index=True, use_container_width=True)
        with right:
            st.subheader("Student index")
            st.dataframe(registry.drop(columns="Folder"), hide_index=True, use_container_width=True)

    elif page == "Enroll Student":
        st.subheader("Enroll a student")
        st.caption("Save at least 5 well-lit, front-facing captures. One face must be visible in each photo.")
        with st.form("enrollment"):
            name = st.text_input("Full name")
            reg_no = st.text_input("Registration number")
            gender = st.selectbox("Gender", ["Male", "Female", "Other"])
            age = st.number_input("Age", min_value=1, max_value=120, value=20)
            submitted = st.form_submit_button("Create profile")
        if submitted:
            if not name.strip() or not reg_no.strip():
                st.error("Name and registration number are required.")
            elif reg_no.strip() in set(registry["RegNo"].astype(str)):
                st.error("That registration number is already enrolled.")
            else:
                new = pd.DataFrame([[name.strip(), reg_no.strip(), gender, int(age)]], columns=STUDENT_COLUMNS)
                pd.concat([registry.drop(columns="Folder", errors="ignore"), new], ignore_index=True).to_csv(STUDENTS_FILE, index=False)
                st.session_state.enrollment_folder = safe_folder_name(name, reg_no)
                (DATASET_DIR / st.session_state.enrollment_folder).mkdir(exist_ok=True)
                st.success("Profile created. Capture training images below.")
        folder = st.session_state.get("enrollment_folder")
        if folder:
            photo = st.camera_input("Take a training photo", key="enrollment_camera")
            if photo and st.button("Save this face capture"):
                crop = crop_single_face(image_from_upload(photo))
                if crop is None:
                    st.error("Please retake the photo with exactly one clear face in frame.")
                else:
                    target = DATASET_DIR / folder
                    number = len(list(target.glob("*.jpg"))) + 1
                    cv2.imwrite(str(target / f"{number:03d}.jpg"), crop)
                    known_faces.clear()
                    st.success(f"Saved capture {number}. Capture several different natural angles.")

    elif page == "Take Attendance":
        st.subheader("Camera recognition")
        st.caption("Take a photo to scan it. A student is marked present only once each day.")
        photo = st.camera_input("Attendance camera", key="attendance_camera")
        if photo and st.button("Recognize faces", type="primary"):
            encodings, names = known_faces(dataset_signature())
            if not encodings:
                st.warning("No usable enrolled face captures found yet.")
            else:
                rgb = cv2.cvtColor(image_from_upload(photo), cv2.COLOR_BGR2RGB)
                found = face_recognition.face_encodings(rgb)
                recognized = []
                for encoding in found:
                    distances = face_recognition.face_distance(encodings, encoding)
                    index = int(np.argmin(distances)) if len(distances) else -1
                    if index >= 0 and distances[index] <= 0.45:
                        recognized.append(names[index])
                added = mark_present(recognized)
                if added:
                    st.success("Marked present: " + ", ".join(display_name(n, registry) for n in added))
                elif recognized:
                    st.info("Recognized student(s) were already marked present today.")
                else:
                    st.warning("No enrolled face was recognized. Improve lighting or add more training captures.")

    else:
        st.subheader("Attendance records")
        view = logs.sort_values("Timestamp", ascending=False).copy()
        view["Name"] = view["Name"].map(lambda x: display_name(x, registry))
        st.dataframe(view, hide_index=True, use_container_width=True)
        st.download_button("Download CSV", view.to_csv(index=False), "attendance_export.csv", "text/csv")


if __name__ == "__main__":
    app()