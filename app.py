import streamlit as st
import cv2
import os
from main import run_enhanced_analysis
from perception import generate_equipment_state

st.set_page_config(page_title="Industrial Safety Twin", layout="wide")
st.title("🏭 Industrial Safety Twin")
st.caption("Powered by NVIDIA Nemotron Ultra 253B via API — no GPU required")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Equipment State")
    img = generate_equipment_state(
        valve_states=[1, 0, 0, 1],
        tools_present=True,
        person_present=True,
        pressure_gauge=55
    )
    st.image(
        cv2.cvtColor(img, cv2.COLOR_BGR2RGB),
        caption="Synthetic Scene: V1 open, V2 closed, V3 closed, V4 open | 55 PSI | Worker + Tools present",
        use_column_width=True
    )

with col2:
    st.subheader("Run Safety Audit")
    if st.button("▶ Analyse with Nemotron Ultra 253B"):
        with st.spinner("Sending scene to Nemotron Ultra 253B via NVIDIA API..."):
            results = run_enhanced_analysis()

        perception = results['perception']
        audit      = results['audit']

        st.subheader("Perception Results")
        st.json({
            'valves_open':   perception['valves_open'],
            'valves_closed': perception['valves_closed'],
            'pressure_psi':  perception['pressure_psi'],
            'worker':        perception['worker_present'],
            'tools':         perception['tools_present']
        })

        st.subheader("Safety Violations")
        if audit['violations']:
            for v in audit['violations']:
                st.error(v)
        else:
            st.success("No violations detected")

        for w in audit.get('warnings', []):
            st.warning(w)

        graph_path = os.path.expanduser("~/safety_twin/knowledge_graph.png")
        if os.path.exists(graph_path):
            st.subheader("Knowledge Graph")
            st.image(graph_path, use_column_width=True)
