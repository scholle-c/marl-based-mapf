import streamlit as st


# Define the pages
select_folder_page = st.Page("select_folder_page.py", title="Select data", icon="📂")
overview_page = st.Page("overview_page.py", title="Training Results", icon="📈")
dist_table_page = st.Page("dist_table_page.py", title="Heuristic History", icon="↔️")
comparison_page = st.Page("comparison_page.py", title="Benchmark Comparison", icon="⚖️")
model_explorer_page = st.Page(
    "model_explorer_page.py", title="Model Explorer", icon="🔬"
)


# Set up navigation
pg = st.navigation(
    [
        select_folder_page,
        overview_page,
        dist_table_page,
        comparison_page,
        model_explorer_page,
    ]
)

# Run the selected page
pg.run()
