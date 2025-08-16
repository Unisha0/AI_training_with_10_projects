import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os

CSV_FILE = "shoe_inventory.csv"

# --- Load and Save Functions ---
def load_inventory():
    if os.path.exists(CSV_FILE):
        return pd.read_csv(CSV_FILE).to_dict(orient="records")
    return []

def save_inventory(inventory):
    pd.DataFrame(inventory).to_csv(CSV_FILE, index=False)

# --- Session Setup ---
if 'inventory' not in st.session_state:
    st.session_state.inventory = load_inventory()

# --- Sidebar ---
st.sidebar.title("👟 Shoe Inventory System")
page = st.sidebar.radio("Navigate", ["Dashboard", "Inventory", "Add Item", "Reports", "Settings"])

# --- Dashboard ---
if page == "Dashboard":
    st.title("📊 Dashboard Overview")
    df = pd.DataFrame(st.session_state.inventory)

    total_items = df["Quantity"].sum()
    total_value = (df["Quantity"] * df["Price"]).sum()
    low_stock = df[df["Quantity"] < 5]

    col1, col2, col3 = st.columns(3)
    col1.metric("👟 Total Pairs", int(total_items))
    col2.metric("💸 Total Value", f"NPR {total_value:,.2f}")
    col3.metric("⚠️ Low Stock Items", len(low_stock))

    st.markdown("### 📦 Inventory Breakdown (Top 10)")
    if not df.empty:
        top_items = df.sort_values(by="Quantity", ascending=False).head(10)
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(top_items["Name"], top_items["Quantity"], color="#4285F4")
        ax.set_xlabel("Shoe Name")
        ax.set_ylabel("Quantity")
        ax.set_title("Top 10 Shoes by Stock")
        plt.xticks(rotation=45, ha='right')
        st.pyplot(fig)
    else:
        st.info("Inventory is empty.")

# --- Inventory Page ---
elif page == "Inventory":
    st.title("📋 View Inventory")
    df = pd.DataFrame(st.session_state.inventory)

    if not df.empty:
        search = st.text_input("🔍 Search Shoe Name")
        sort_by = st.selectbox("Sort by", ["Name", "Quantity", "Price"])

        if search:
            df = df[df["Name"].str.contains(search, case=False)]

        df = df.sort_values(by=sort_by)
        st.dataframe(df, use_container_width=True)

        if not df[df["Quantity"] < 5].empty:
            st.warning("⚠️ Some shoes are low in stock! Reorder soon.")
    else:
        st.info("No inventory found.")

# --- Add Item Page ---
elif page == "Add Item":
    st.title("➕ Add New Shoe")
    all_shoes_df = pd.read_csv(CSV_FILE)
    shoe_names = sorted(all_shoes_df["Name"].unique().tolist())

    with st.form("add_form"):
        name = st.selectbox("Choose Shoe Model", shoe_names)
        qty = st.number_input("Quantity", min_value=1)
        price = st.number_input("Price (NPR)", min_value=0.0, step=100.0)
        add = st.form_submit_button("Add Shoe")

        if add:
            new_item = {
                "Name": name.strip().title(),
                "Quantity": int(qty),
                "Price": float(price)
            }
            st.session_state.inventory.append(new_item)
            save_inventory(st.session_state.inventory)
            st.success(f"✅ '{name}' added to inventory.")

# --- Reports Page ---
elif page == "Reports":
    st.title("📈 Inventory Reports")
    df = pd.DataFrame(st.session_state.inventory)

    if not df.empty:
        df['Total Value'] = df['Quantity'] * df['Price']
        st.markdown("#### 💰 Value per Shoe")
        st.dataframe(df[['Name', 'Quantity', 'Price', 'Total Value']], use_container_width=True)

        st.markdown("#### 📊 Value Distribution")
        fig, ax = plt.subplots()
        ax.pie(df['Total Value'], labels=df['Name'], autopct='%1.1f%%', startangle=90)
        ax.axis('equal')
        st.pyplot(fig)
    else:
        st.warning("No data available for reports.")

# --- Settings Page ---
elif page == "Settings":
    st.title("⚙️ Settings")

    if st.button("🗑️ Clear All Inventory"):
        st.session_state.inventory = []
        save_inventory(st.session_state.inventory)
        st.success("✅ All inventory cleared.")

    st.info("Data is saved to 'shoe_inventory.csv' automatically.")
    st.markdown("THREADSANDTREANDS STORE - YUNISHA")