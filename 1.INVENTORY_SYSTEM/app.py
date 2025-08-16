import streamlit as st
import pandas as pd
import os

CSV_FILE = "shoe_inventory.csv"

# --- Load Inventory ---
def load_inventory():
    if os.path.exists(CSV_FILE):
        return pd.read_csv(CSV_FILE).to_dict(orient="records")
    return []

# --- Save Inventory ---
def save_inventory(inventory):
    pd.DataFrame(inventory).to_csv(CSV_FILE, index=False)

# --- Style ---
st.markdown("""
    <style>
    .main-title {
        font-size: 36px;
        color: #0e4d92;
        text-align: center;
        margin-bottom: 20px;
    }
    .stForm, .stButton button {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0px 4px 10px rgba(0,0,0,0.1);
    }
    .stButton > button {
        background-color: #0e4d92;
        color: white;
        font-weight: bold;
        padding: 10px 25px;
        border-radius: 8px;
        transition: background-color 0.3s ease;
    }
    .stButton > button:hover {
        background-color: #1363c6;
        cursor: pointer;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">👟 Shoe Inventory System</div>', unsafe_allow_html=True)

# --- Session ---
if 'inventory' not in st.session_state:
    st.session_state.inventory = load_inventory()

# --- Add Form ---
all_shoes_df = pd.read_csv(CSV_FILE)
shoe_names = sorted(all_shoes_df["Name"].unique().tolist())

st.markdown("### ➕ Add New Shoe")
with st.form("add_form"):
    name = st.selectbox("Choose Shoe", shoe_names)
    quantity = st.number_input("Quantity", min_value=1, step=1)
    price = st.number_input("Price (NPR)", min_value=0.0, step=100.0)
    submit = st.form_submit_button("Add Shoe")

    if submit:
        new_item = {"Name": name.strip().title(), "Quantity": quantity, "Price": price}
        st.session_state.inventory.append(new_item)
        save_inventory(st.session_state.inventory)
        st.success(f"✅ '{name}' added to inventory.")

# --- Display Inventory ---
if st.session_state.inventory:
    st.markdown("### 📋 Current Inventory")
    df = pd.DataFrame(st.session_state.inventory)
    st.dataframe(df, use_container_width=True)

    # --- Update Quantity ---
    st.markdown("### ✏️ Update Quantity")
    shoes = [item["Name"] for item in st.session_state.inventory]
    selected = st.selectbox("Select shoe", shoes)
    new_qty = st.number_input("New quantity", min_value=0, step=1)
    if st.button("Update Quantity"):
        for item in st.session_state.inventory:
            if item["Name"] == selected:
                item["Quantity"] = new_qty
                save_inventory(st.session_state.inventory)
                st.success(f"✅ Quantity updated for '{selected}'.")

    # --- Delete Shoe ---
    st.markdown("### 🗑️ Delete Shoe")
    delete_target = st.selectbox("Delete which shoe?", shoes, key="delete_select")
    if st.button("Delete Shoe"):
        st.session_state.inventory = [item for item in st.session_state.inventory if item["Name"] != delete_target]
        save_inventory(st.session_state.inventory)
        st.success(f"🗑️ '{delete_target}' removed from inventory.")
else:
    st.info("No shoes in inventory yet.")