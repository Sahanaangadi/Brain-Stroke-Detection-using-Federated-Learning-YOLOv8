import streamlit as st
import sqlite3
import bcrypt
from datetime import datetime
from PIL import Image
from ultralytics import YOLO
import tempfile
import pandas as pd
import os
import plotly.express as px
from PIL import Image

import base64

# --- DB Setup ---
conn = sqlite3.connect("users.db", check_same_thread=False)
cursor = conn.cursor()

cursor.execute('''CREATE TABLE IF NOT EXISTS users (
    username TEXT PRIMARY KEY,
    name TEXT,
    password TEXT,
    role TEXT
)''')

cursor.execute('''CREATE TABLE IF NOT EXISTS logs (
    username TEXT,
    name TEXT,
    timestamp TEXT,
    image_name TEXT,
    prediction TEXT
)''')

# Insert dummy users if not exists
def insert_user(username, name, password_plain, role):
    cursor.execute("SELECT * FROM users WHERE username=?", (username,))
    if not cursor.fetchone():
        hashed = bcrypt.hashpw(password_plain.encode(), bcrypt.gensalt()).decode()
        cursor.execute("INSERT INTO users VALUES (?, ?, ?, ?)", (username, name, hashed, role))
        conn.commit()

insert_user("sahana_s_a", "Sahana S Angadi", "sahana123", "Radiologist")
insert_user("admin1", "Admin", "admin123", "Admin")

# --- Auth ---
def check_user_credentials(username, password):
    cursor.execute("SELECT * FROM users WHERE username=?", (username,))
    row = cursor.fetchone()
    if row and bcrypt.checkpw(password.encode(), row[2].encode()):
        return {"username": row[0], "name": row[1], "role": row[3]}
    return None

# --- Session State ---
if "user" not in st.session_state:
    st.session_state.user = None




# Stylish dark theme with orange highlights


# Full custom dark theme with visible input fields

st.markdown(
    """
    <style>
    /* Page background */
    .stApp {
        background: linear-gradient(to bottom right, #0f0f0f, #1c1c1c);
        color: #FFA500;
    }

    /* Headings */
    h1, h2, h3, h4, h5, h6 {
        color: #FFA500;
    }

    /* General text */
    .stMarkdown, .stText, .stDataFrame, .stTable, .stMetric {
        color: #f0f0f0;
    }

    /* Sidebar */
    .css-1d391kg, .css-1v3fvcr, .css-h5rgaw {
        background-color: #141414 !important;
        color: #FFA500 !important;
    }

    /* Input fields (text/password/select) */
    input, textarea {
        background-color: #2a2a2a !important;
        color: #f0f0f0 !important;
        border: 1px solid #FFA500 !important;
        border-radius: 6px !important;
    }

    /* Placeholder text */
    input::placeholder, textarea::placeholder {
        color: #cccccc !important;
    }

    /* Form field labels (this is the missing part you needed) */
    label, .stTextInput label, .stTextArea label, .stSelectbox label {
        color: #FFA500 !important;
        font-weight: bold;
    }

    /* Select dropdown text */
    .stSelectbox>div>div {
        color: #f0f0f0 !important;
        background-color: #2a2a2a !important;
    }

    /* Button style */
    .stButton>button {
        background-color: #FFA500;
        color: black;
        border-radius: 8px;
        padding: 0.5em 1em;
        border: none;
        transition: 0.3s ease;
    }

    .stButton>button:hover {
        background-color: #ff8c00;
        color: white;
    }

    /* File uploader */
    .stFileUploader {
        background-color: #2a2a2a;
        border-radius: 8px;
        color: #f0f0f0;
    }

    /* DataFrame (Log Table) */
    .stDataFrame {
        background-color: #1f1f1f;
        color: #f0f0f0;
        border-radius: 10px;
        padding: 1em;
    }

    /* Download button */
    .stDownloadButton button {
        background-color: #333;
        color: #FFA500;
        border-radius: 6px;
        border: 1px solid #FFA500;
    }

    /* Metrics boxes */
    .stMetric {
        background-color: #2a2a2a;
        border-radius: 10px;
        padding: 0.5em;
    }

    /* Scrollbars */
    ::-webkit-scrollbar {
        width: 8px;
    }
    ::-webkit-scrollbar-track {
        background: #111;
    }
    ::-webkit-scrollbar-thumb {
        background: #FFA500;
        border-radius: 4px;
    }
    </style>
    """,
    unsafe_allow_html=True
)


st.markdown("""
<h1 style='text-align: center; color: #FFA500;'>🏥 BrainCare Hospital</h1>
<hr style='border: 1px solid #FFA500;'>
""", unsafe_allow_html=True)



# Rest of your code follows here
# (Keep your original app code here below this CSS)

# --- Login ---
if not st.session_state.user:
    st.title("🔐 Secure Login")

    user = None  # define before use

    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Login")

    if submitted:
        user = check_user_credentials(username, password)
        if user:
            st.session_state.user = user
            st.success(f"Welcome {user['name']} ({user['role']})!")
            st.rerun()  # Trigger the rerun to show the logged-in UI
        else:
            st.error("Invalid credentials.")

else:
    user = st.session_state.user
    st.sidebar.success(f"Welcome {user['name']}!")

    # --- Logout Button ---
    if st.sidebar.button("🚪 Logout"):
        st.session_state.user = None
        st.rerun()  # Trigger the rerun after logout to go back to login

    # --- Doctor UI ---
    if user["role"] == "Radiologist":
        st.title("🧠 Stroke Detection from CT Scans")
        st.write("Upload a brain CT scan to check for stroke.")

        model = YOLO("C:/Stroke1/final_client_model.pt")

        uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])
        if uploaded_file:
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption="Uploaded Image", width=400)

            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
                image.save(temp_file.name)
                temp_path = temp_file.name

            results = model(temp_path)
            if results:
                probs = results[0].probs
                class_id = int(probs.top1)
                class_name = results[0].names[class_id]
                confidence = float(probs.data[class_id])
                st.success(f"🧾 Prediction: **{class_name.upper()}**")

                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                cursor.execute("INSERT INTO logs VALUES (?, ?, ?, ?, ?)",
                               (user["username"], user["name"], timestamp, uploaded_file.name, class_name))
                conn.commit()

                st.info(f"Logged at {timestamp}")
            else:
                st.error("Model failed to predict.")

    # --- Admin UI ---
    elif user["role"] == "Admin":
        st.title("📊 Admin Dashboard")
        st.markdown("Welcome, Admin. Here's a snapshot of recent scan activity and stats.")

        # Admin - Add New User
        if "show_add_user_form" not in st.session_state:
            st.session_state.show_add_user_form = False

        if st.button("Add New User"):
            st.session_state.show_add_user_form = not st.session_state.show_add_user_form

        if st.session_state.show_add_user_form:
            # Add New User Form
            st.subheader("🔑 Add New User")
            with st.form("add_user_form"):
                new_username = st.text_input("Username")
                new_name = st.text_input("Name")
                new_password = st.text_input("Password", type="password")
                new_role = st.selectbox("Role", ["Doctor", "Admin"])

                add_user_button = st.form_submit_button("Add User")

                if add_user_button:
                    if new_username and new_name and new_password:
                        insert_user(new_username, new_name, new_password, new_role)
                        st.success(f"User {new_username} added successfully!")
                        st.session_state.show_add_user_form = False  # Hide form after submitting
                        st.rerun()  # Trigger the rerun state change
                    else:
                        st.error("Please fill out all fields.")

        # Display the list of users with an option to remove them
        cursor.execute("SELECT username, name, role FROM users WHERE role != 'Admin'")
        users = cursor.fetchall()

        if users:
            st.subheader("💼 User List")
            for user in users:
                col1, col2, col3 = st.columns([2, 1, 1])
                col1.write(f"{user[1]} ({user[0]})")
                col2.write(f"Role: {user[2]}")
                if col3.button("Remove", key=user[0]):
                    # Remove user from the database
                    cursor.execute("DELETE FROM users WHERE username=?", (user[0],))
                    conn.commit()
                    st.success(f"User {user[1]} removed successfully!")
                    st.rerun()  # Trigger the rerun state change

        # Show logs
        cursor.execute("SELECT * FROM logs ORDER BY timestamp DESC")
        logs = cursor.fetchall()

        if logs:
            df = pd.DataFrame(logs, columns=["Username", "Name", "Timestamp", "Image", "Prediction"])
            df["Timestamp"] = pd.to_datetime(df["Timestamp"])

            # --- METRICS ---
            col1, col2, col3 = st.columns(3)
            col1.metric("Total Scans", len(df))
            col2.metric("Stroke Cases", len(df[df['Prediction'].str.lower() == 'stroke']))
            col3.metric("Normal Cases", len(df[df['Prediction'].str.lower() == 'normal']))

            st.markdown("---")

            # --- BAR CHART: Scans per Doctor ---
            st.subheader("🩺 Scans per Doctor")
            scans_by_doc = df["Name"].value_counts().reset_index()
            scans_by_doc.columns = ["Doctor", "Scan Count"]
            st.bar_chart(scans_by_doc.set_index("Doctor"))

            # --- PIE CHART: Stroke vs Normal ---
            st.subheader("🧠 Stroke vs Normal")
            pie_data = df["Prediction"].value_counts().reset_index()
            pie_data.columns = ["Condition", "Count"]
            fig = px.pie(pie_data, names='Condition', values='Count', title='Prediction Distribution')
            st.plotly_chart(fig)

            # --- LINE CHART: Daily Activity ---
            st.subheader("📅 Daily Scan Activity")
            daily_scans = df.groupby(df["Timestamp"].dt.date).size().reset_index(name="Scans")
            fig2 = px.line(daily_scans, x="Timestamp", y="Scans", markers=True, title="Scans Over Time")
            st.plotly_chart(fig2)

            # --- FULL LOG TABLE ---
            st.subheader("📄 Full Prediction Log")
            st.dataframe(df)

            # --- DOWNLOAD --- 
            st.download_button(
                "📥 Download Logs CSV",
                data=df.to_csv(index=False).encode("utf-8"),
                file_name="logs.csv",
                mime="text/csv"
            )

            # --- CLEAR HISTORY BUTTON ---
            if st.button("🧹 Clear History"):
                cursor.execute("DELETE FROM logs")
                #cursor.execute("DELETE FROM users WHERE role != 'Admin'")  # Prevent clearing admin users
                conn.commit()
                st.success("History has been cleared!")

        else:
            st.warning("No logs available yet.")

    if "rerun" in st.session_state and st.session_state.rerun:
        st.session_state.rerun = False  # Reset rerun flag to prevent constant reloading
        st.rerun()  # Use this only for trigger-based rerun
