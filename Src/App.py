# import streamlit as st
# import pandas as pd
# import numpy as np
# from sklearn.ensemble import IsolationForest
# from datetime import datetime
# import altair as alt
# import time

# # Streamlit config
# st.set_page_config(page_title="Live Anomaly Detection", layout="wide")
# st.title("📉 Real-Time Anomaly Detection")

# # Spec ranges
# specs = {
#     "Top Layer fan": {
#         "range": (-3, 10),
#         "warning": (2, 6),
#         "outlier": (0, 7)
#     },
#     "Main Condensar L": {
#         "range": (40, 90),
#         "warning": (60, 75),
#         "outlier": (50, 80)
#     }
# }

# # Initialize session state
# if "data" not in st.session_state:
#     st.session_state.data = pd.DataFrame(columns=["id", "datetime", "parameter", "value", "rule_based", "model", "source"])
#     st.session_state.model = IsolationForest(contamination=0.1, n_estimators=100, random_state=42)
# if "counter" not in st.session_state:
#     st.session_state.counter = 1

# # Toggle buttons for display options
# st.sidebar.title("Display Options")
# display_option = st.sidebar.radio(
#     "Choose what to display:",
#     ("Both Charts and Tables", "Charts Only", "Tables Only"),
#     index=0
# )

# # Rule-based classification
# def classify_rule(param, val):
#     s = specs[param]
#     if val < s["outlier"][0] or val > s["outlier"][1]:
#         return -1
#     elif val < s["warning"][0] or val > s["warning"][1]:
#         return 1
#     return 0

# # Generate new random record
# def generate_new_record(i):
#     param = np.random.choice(list(specs.keys()))
#     val = round(np.random.uniform(*specs[param]["range"]), 2)
#     return {
#         "id": i,
#         "datetime": datetime.now(),
#         "parameter": param,
#         "value": val
#     }

# # Apply both rule and model detection
# def detect_anomalies(df):
#     df["rule_based"] = df.apply(lambda row: classify_rule(row["parameter"], row["value"]), axis=1)

#     normal_df = df[df["rule_based"] != -1]
#     if len(normal_df) > 30:
#         st.session_state.model.fit(normal_df[["value"]])
#         df["model"] = st.session_state.model.predict(df[["value"]])
#     else:
#         df["model"] = 0

#     def get_source(row):
#         if row["model"] == -1 and row["rule_based"] == -1:
#             return "Both"
#         elif row["model"] == -1:
#             return "Model"
#         elif row["rule_based"] == -1:
#             return "Rule-based"
#         elif row["rule_based"] == 1:
#             return "Warning"
#         else:
#             return "Normal"
#     df["source"] = df.apply(get_source, axis=1)
#     return df

# # Create combined scatter + line chart for each parameter
# def create_combined_chart(data, param_name):
#     param_data = data[data["parameter"] == param_name]
    
#     if param_data.empty:
#         return alt.Chart(pd.DataFrame()).mark_text().encode(
#             text=alt.value(f"No data available for {param_name}")
#         ).properties(width=600, height=400, title=f"{param_name} - No Data")
    
#     base = alt.Chart(param_data).encode(
#         x=alt.X('datetime:T', title='Time'),
#         y=alt.Y('value:Q', title='Value'),
#         tooltip=['id', 'datetime', 'parameter', 'value', 'rule_based', 'model', 'source']
#     )
    
#     # Scatter layer
#     scatter = base.mark_circle(size=100).encode(
#         color=alt.Color('source:N',
#             scale=alt.Scale(
#                 domain=['Both', 'Model', 'Rule-based', 'Warning', 'Normal'],
#                 range=['red', 'red', 'red', 'orange', 'green']
#             )
#         )
#     )
    
#     # Line layer
#     line = base.mark_line(strokeWidth=2).encode(
#         color=alt.value('blue')
#     )
    
#     # Combine scatter and line
#     combined_chart = (line + scatter).properties(
#         width=600, 
#         height=400, 
#         title=f"{param_name} "
#     )
    
#     return combined_chart

# # Create table for each parameter
# def create_parameter_table(data, param_name):
#     param_data = data[data["parameter"] == param_name].tail(15)
#     if param_data.empty:
#         return pd.DataFrame({"Message": [f"No data available for {param_name}"]})
#     return param_data[["id", "datetime", "value", "rule_based", "model", "source"]].reset_index(drop=True)

# # Main placeholder for real-time updates
# placeholder = st.empty()

# # Real-time data loop
# while True:
#     new_row = generate_new_record(st.session_state.counter)
#     new_df = pd.DataFrame([new_row])
#     st.session_state.data = pd.concat([st.session_state.data, new_df], ignore_index=True)
#     st.session_state.counter += 1

#     st.session_state.data = detect_anomalies(st.session_state.data)

#     with placeholder.container():
#         # Latest data snapshot
#         # st.subheader("📊 Latest Data Snapshot")
#         # st.dataframe(st.session_state.data.tail(10).reset_index(drop=True), use_container_width=True)
        
#         # Get parameter names
#         param_names = list(specs.keys())
        
#         # Create layout based on display option
#         if display_option == "Charts Only":
#             # Show only charts
#             col1, col2 = st.columns(2)
            
#             with col1:
#                 st.subheader(f"📈 {param_names[0]}")
#                 chart1 = create_combined_chart(st.session_state.data, param_names[0])
#                 st.altair_chart(chart1, use_container_width=True)
            
#             with col2:
#                 st.subheader(f"📈 {param_names[1]}")
#                 chart2 = create_combined_chart(st.session_state.data, param_names[1])
#                 st.altair_chart(chart2, use_container_width=True)
                
#         elif display_option == "Tables Only":
#             # Show only tables
#             col1, col2 = st.columns(2)
            
#             with col1:
#                 st.subheader(f"📋 {param_names[0]} Data Table")
#                 table1 = create_parameter_table(st.session_state.data, param_names[0])
#                 st.dataframe(table1, use_container_width=True)
            
#             with col2:
#                 st.subheader(f"📋 {param_names[1]} Data Table")
#                 table2 = create_parameter_table(st.session_state.data, param_names[1])
#                 st.dataframe(table2, use_container_width=True)
                
#         else:  # Both Charts and Tables
#             # Show both charts and tables
            
#             # Charts row
#             # st.subheader("📈 Charts")
#             col1, col2 = st.columns(2)
            
#             with col1:
#                 # st.write(f"**{param_names[0]} - Scatter + Line Chart**")
#                 chart1 = create_combined_chart(st.session_state.data, param_names[0])
#                 st.altair_chart(chart1, use_container_width=True)
            
#             with col2:
#                 # st.write(f"**{param_names[1]} - Scatter + Line Chart**")
#                 chart2 = create_combined_chart(st.session_state.data, param_names[1])
#                 st.altair_chart(chart2, use_container_width=True)
            
#             # Tables row
#             # st.subheader("📋 Data Tables")
#             col3, col4 = st.columns(2)
            
#             with col3:
#                 st.write(f"**{param_names[0]} Data Table**")
#                 table1 = create_parameter_table(st.session_state.data, param_names[0])
#                 st.dataframe(table1, use_container_width=True)
            
#             with col4:
#                 st.write(f"**{param_names[1]} Data Table**")
#                 table2 = create_parameter_table(st.session_state.data, param_names[1])
#                 st.dataframe(table2, use_container_width=True)

#     time.sleep(1)






















import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from datetime import datetime
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time

# Streamlit config
st.set_page_config(page_title="Live Anomaly Detection", layout="wide")
# st.title("📉 Real-Time Anomaly Detection (Rule-Based + Isolation Forest)")

# Spec ranges
specs = {
    "Top Layer fan": {
        "range": (-3, 10),
        "warning": (2, 6),
        "outlier": (0, 7),
        "spec_lines": {
            "LRL": 1,
            "LSL": 2, 
            "LWL": 3,
            "TGT": 4,
            "UWL": 5,
            "USL": 6,
            "URL": 7
        }
    },
    "Main Condensar L": {
        "range": (40, 90),
        "warning": (60, 75),
        "outlier": (50, 80),
        "spec_lines": {
            "LRL": 45,
            "LSL": 50,
            "LWL": 55,
            "TGT": 65,
            "UWL": 75,
            "USL": 80,
            "URL": 85
        }
    }
}

# Initialize session state
if "data" not in st.session_state:
    st.session_state.data = pd.DataFrame(columns=["id", "datetime", "parameter", "value", "rule_based", "model", "source"])
    st.session_state.model = IsolationForest(contamination=0.1, n_estimators=100, random_state=42)
if "counter" not in st.session_state:
    st.session_state.counter = 1

# Initialize show_table state for each parameter
if "show_table_Top Layer fan" not in st.session_state:
    st.session_state["show_table_Top Layer fan"] = False
if "show_table_Main Condensar L" not in st.session_state:
    st.session_state["show_table_Main Condensar L"] = False

# # Toggle buttons for display options (COMMENTED OUT)
# st.sidebar.title("Display Options")
# display_option = st.sidebar.radio(
#     "Choose what to display:",
#     ("Both Charts and Tables", "Charts Only", "Tables Only"),
#     index=0
# )

# Rule-based classification
def classify_rule(param, val):
    s = specs[param]
    if val < s["outlier"][0] or val > s["outlier"][1]:
        return -1
    elif val < s["warning"][0] or val > s["warning"][1]:
        return 1
    return 0

# Generate new random record
def generate_new_record(i):
    param = np.random.choice(list(specs.keys()))
    val = round(np.random.uniform(*specs[param]["range"]), 2)
    return {
        "id": i,
        "datetime": datetime.now(),
        "parameter": param,
        "value": val
    }

# Apply both rule and model detection
def detect_anomalies(df):
    df["rule_based"] = df.apply(lambda row: classify_rule(row["parameter"], row["value"]), axis=1)

    normal_df = df[df["rule_based"] != -1]
    if len(normal_df) > 30:
        st.session_state.model.fit(normal_df[["value"]])
        df["model"] = st.session_state.model.predict(df[["value"]])
    else:
        df["model"] = 0

    def get_source(row):
        if row["model"] == -1 and row["rule_based"] == -1:
            return "Both"
        elif row["model"] == -1:
            return "Model"
        elif row["rule_based"] == -1:
            return "Rule-based"
        elif row["rule_based"] == 1:
            return "Warning"
        else:
            return "Normal"
    df["source"] = df.apply(get_source, axis=1)
    return df

# Create Plotly chart for each parameter (Fixed blinking and legend issues)
def create_plotly_chart(data, param_name):
    param_data = data[data["parameter"] == param_name].copy()
    
    if param_data.empty:
        # Create empty chart with message
        fig = go.Figure()
        fig.add_annotation(
            text=f"No data available for {param_name}",
            xref="paper", yref="paper",
            x=0.5, y=0.5, xanchor='center', yanchor='middle',
            showarrow=False, font=dict(size=16)
        )
        fig.update_layout(
            title=f"{param_name} - No Data",
            height=400
        )
        return fig
    
    # Sort by datetime for proper line connection
    param_data = param_data.sort_values('datetime')
    
    # Create figure
    fig = go.Figure()
    
    # Get spec ranges for this parameter
    param_specs = specs[param_name]
    
    # Define colors for each spec line (blue gradient palette)
    spec_colors = {
        "LRL": "rgba(25, 25, 112, 0.6)",      # Midnight Blue
        "LSL": "rgba(65, 105, 225, 0.7)",     # Royal Blue  
        "LWL": "rgba(100, 149, 237, 0.8)",    # Cornflower Blue
        "TGT": "rgba(0, 191, 255, 0.9)",      # Deep Sky Blue (Target - most prominent)
        "UWL": "rgba(135, 206, 250, 0.8)",    # Light Sky Blue
        "USL": "rgba(173, 216, 230, 0.7)",    # Light Blue
        "URL": "rgba(176, 196, 222, 0.6)"     # Light Steel Blue
    }
    
    # Add static horizontal dotted lines for spec values
    spec_lines_for_legend = []
    for spec_name, spec_value in param_specs["spec_lines"].items():
        fig.add_hline(
            y=spec_value,
            line_dash="dot",
            line_color=spec_colors[spec_name],
            line_width=2 if spec_name == "TGT" else 1.5,  # Make target line thicker
            annotation_text="",  # No annotation text
        )
        
        # Add invisible scatter trace for legend
        fig.add_trace(go.Scatter(
            x=[None], y=[None],
            mode='lines',
            line=dict(color=spec_colors[spec_name], dash='dot', width=2),
            name=f'{spec_name}: {spec_value}',
            showlegend=True,
            legendgroup='spec_lines',
            hoverinfo='skip'
        ))
    
    # Add line trace for all data
    fig.add_trace(go.Scatter(
        x=param_data['datetime'],
        y=param_data['value'],
        mode='lines',
        name='Trend Line',
        line=dict(color='lightblue', width=2),
        hoverinfo='skip',
        showlegend=False  # Hide trend line from legend
    ))
    
    # Color mapping for different sources
    color_map = {
        'Both': 'red',
        'Model': 'red', 
        'Rule-based': 'red',
        'Warning': 'orange',
        'Normal': 'green'
    }
    
    # Group similar anomaly types to reduce legend items
    legend_map = {
        'Both': 'Anomaly',
        'Model': 'Anomaly', 
        'Rule-based': 'Anomaly',
        'Warning': 'Warning',
        'Normal': 'Normal'
    }
    
    # Track which legend groups we've added to avoid duplicates
    added_legend_groups = set()
    
    # Add scatter points for each source type
    for source in param_data['source'].unique():
        source_data = param_data[param_data['source'] == source]
        legend_group = legend_map.get(source, source)
        
        fig.add_trace(go.Scatter(
            x=source_data['datetime'],
            y=source_data['value'],
            mode='markers',
            name=legend_group,
            legendgroup=legend_group,  # Group similar items
            showlegend=legend_group not in added_legend_groups,  # Only show legend once per group
            marker=dict(
                color=color_map.get(source, 'blue'),
                size=8,
                line=dict(width=1, color='white')
            ),
            customdata=source_data[['id', 'parameter']],
            hovertemplate='<b>%{customdata[1]}</b><br>' +
                         'Time: %{x}<br>' +
                         'Value: %{y}<br>' +
                         'ID: %{customdata[0]}<br>' +
                         'Source: ' + source +
                         '<extra></extra>'
        ))
        added_legend_groups.add(legend_group)
    
    # Update layout with fixed legend position and reduced blinking
    fig.update_layout(
        xaxis_title="Time",
        yaxis_title="Value",
        height=600,  # Increased height for better visibility
        width=800,   # Increased width
        hovermode='closest',
        # Enhanced styling
        plot_bgcolor='rgba(248, 249, 250, 0.8)',  # Light gray background
        paper_bgcolor='white',
        # Fixed legend position to prevent floating behind selectors
        legend=dict(
            orientation="v",  # Vertical orientation
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.02,  # Position legend to the right of chart
            bgcolor="rgba(255, 255, 255, 0.95)",
            bordercolor="rgba(0, 0, 0, 0.1)",
            borderwidth=1,
            font=dict(size=10),
            itemsizing="constant"
        ),
        xaxis=dict(
            rangeselector=dict(
                buttons=list([
                    dict(count=1, label="1m", step="minute", stepmode="backward"),
                    dict(count=5, label="5m", step="minute", stepmode="backward"),
                    dict(count=15, label="15m", step="minute", stepmode="backward"),
                    dict(count=30, label="30m", step="minute", stepmode="backward"),
                    dict(count=1, label="1h",step="hour", stepmode="backward"),
                    dict(step="all", label="All")
                ]),
                # Position selectors to avoid legend overlap
                y=1.15,
                x=0,
                bgcolor="rgba(255, 255, 255, 0.8)",
                bordercolor="rgba(0, 0, 0, 0.1)",
                borderwidth=1
            ),
            rangeslider=dict(
                visible=True,
                bgcolor="rgba(248, 249, 250, 0.5)"
            ),
            type="date",
            gridcolor="rgba(200, 200, 200, 0.3)",
            showgrid=True,
            # range=[
            #     pd.Timestamp.now() - pd.Timedelta(hours=1),
            #     pd.Timestamp.now()
            # ]
        ),
        yaxis=dict(
            gridcolor="rgba(200, 200, 200, 0.3)",
            showgrid=True,
            zeroline=True,
            zerolinecolor="rgba(200, 200, 200, 0.5)"
        ),
        # Reduce animation to prevent blinking
        transition={'duration': 0},
        # Set margins to accommodate legend and make chart wider
        margin=dict(l=60, r=150, t=80, b=60)
    )
    
    return fig

# Create table for each parameter
def create_parameter_table(data, param_name):
    param_data = data[data["parameter"] == param_name].tail(15)
    if param_data.empty:
        return pd.DataFrame({"Message": [f"No data available for {param_name}"]})
    
    # Format datetime for better display
    param_data_display = param_data.copy()
    param_data_display['datetime'] = param_data_display['datetime'].dt.strftime('%H:%M:%S')
    return param_data_display[["id", "datetime", "value", "source"]].reset_index(drop=True)

# Main placeholder for real-time updates
placeholder = st.empty()

# Real-time data loop
while True:
    new_row = generate_new_record(st.session_state.counter)
    new_df = pd.DataFrame([new_row])
    st.session_state.data = pd.concat([st.session_state.data, new_df], ignore_index=True)
    st.session_state.counter += 1

    st.session_state.data = detect_anomalies(st.session_state.data)

    with placeholder.container():
        # Latest data snapshot
        # st.subheader("📊 Latest Data Snapshot")
        # st.dataframe(st.session_state.data.tail(10).reset_index(drop=True), use_container_width=True)
        
        # Get parameter names
        param_names = list(specs.keys())
        
        # # COMMENTED OUT: Create layout based on display option
        # if display_option == "Charts Only":
        #     # Show only charts
        #     col1, col2 = st.columns(2)
        #     
        #     with col1:
        #         st.subheader(f"📈 {param_names[0]}")
        #         fig1 = create_plotly_chart(st.session_state.data, param_names[0])
        #         st.plotly_chart(fig1, use_container_width=True, key=f"chart_only_{param_names[0]}")
        #     
        #     with col2:
        #         st.subheader(f"📈 {param_names[1]}")
        #         fig2 = create_plotly_chart(st.session_state.data, param_names[1])
        #         st.plotly_chart(fig2, use_container_width=True, key=f"chart_only_{param_names[1]}")
        #         
        # elif display_option == "Tables Only":
        #     # Show only tables
        #     col1, col2 = st.columns(2)
        #     
        #     with col1:
        #         st.subheader(f"📋 {param_names[0]} Data Table")
        #         table1 = create_parameter_table(st.session_state.data, param_names[0])
        #         st.dataframe(table1, use_container_width=True)
        #     
        #     with col2:
        #         st.subheader(f"📋 {param_names[1]} Data Table")
        #         table2 = create_parameter_table(st.session_state.data, param_names[1])
        #         st.dataframe(table2, use_container_width=True)
        #         
        # else:  # Both Charts and Tables
        #     # Show both charts and tables
        #     
        #     # Charts row
        #     st.subheader("📈 Interactive Charts")
        #     col1, col2 = st.columns(2)
        #     
        #     with col1:
        #         st.write(f"**{param_names[0]} - Scatter + Line Chart**")
        #         fig1 = create_plotly_chart(st.session_state.data, param_names[0])
        #         st.plotly_chart(fig1, use_container_width=True, key=f"both_chart_{param_names[0]}")
        #     
        #     with col2:
        #         st.write(f"**{param_names[1]} - Scatter + Line Chart**")
        #         fig2 = create_plotly_chart(st.session_state.data, param_names[1])
        #         st.plotly_chart(fig2, use_container_width=True, key=f"both_chart_{param_names[1]}")
        #     
        #     # Tables row
        #     st.subheader("📋 Data Tables")
        #     col3, col4 = st.columns(2)
        #     
        #     with col3:
        #         st.write(f"**{param_names[0]} Data Table**")
        #         table1 = create_parameter_table(st.session_state.data, param_names[0])
        #         st.dataframe(table1, use_container_width=True)
        #     
        #     with col4:
        #         st.write(f"**{param_names[1]} Data Table**")
        #         table2 = create_parameter_table(st.session_state.data, param_names[1])
        #         st.dataframe(table2, use_container_width=True)

        # NEW FUNCTIONALITY: Always show charts with toggle buttons for tables
        st.subheader("📈 Interactive Charts")
        col1, col2 = st.columns(2)
        
        # First Parameter Chart and Table
        with col1:
            # Chart header with toggle checkbox
            chart_col, button_col = st.columns([3, 1])
            with chart_col:
                st.write(f"**{param_names[0]}**")
            # with button_col:
            #     # Using checkbox instead of button for persistent state
            #     show_table_1 = st.checkbox("📊 Data", 
            #                              key=f"toggle_{param_names[0]}_{st.session_state.counter}", 
            #                              value=st.session_state[f"show_table_{param_names[0]}"],
            #                              help=f"Show/Hide data table for {param_names[0]}")
            #     st.session_state[f"show_table_{param_names[0]}"] = show_table_1

            # Always show chart
            fig1 = create_plotly_chart(st.session_state.data, param_names[0])
            st.plotly_chart(fig1, use_container_width=True, key=f"chart_{param_names[0]}_{st.session_state.counter}")
            
            # Show table if toggled on
            if st.session_state[f"show_table_{param_names[0]}"]:
                st.write(f"**📋 {param_names[0]} Data Table**")
                table1 = create_parameter_table(st.session_state.data, param_names[0])
                st.dataframe(table1, use_container_width=True, key=f"table_{param_names[0]}")
        
        # Second Parameter Chart and Table
        with col2:
            # Chart header with toggle checkbox
            chart_col2, button_col2 = st.columns([3, 1])
            with chart_col2:
                st.write(f"**{param_names[1]}**")
            # with button_col2:
            #     # Using checkbox instead of button for persistent state
            #     show_table_2 = st.checkbox("📊 Data", 
            #                              key=f"toggle_{param_names[1]}_{st.session_state.counter}", 
            #                              value=st.session_state[f"show_table_{param_names[1]}"],
            #                              help=f"Show/Hide data table for {param_names[1]}")
            #     st.session_state[f"show_table_{param_names[1]}"] = show_table_2
            
            # Always show chart
            fig2 = create_plotly_chart(st.session_state.data, param_names[1])
            st.plotly_chart(fig2, use_container_width=True, key=f"chart_{param_names[1]}_{st.session_state.counter}")

            # Show table if toggled on
            if st.session_state[f"show_table_{param_names[1]}"]:
                st.write(f"**📋 {param_names[1]} Data Table**")
                table2 = create_parameter_table(st.session_state.data, param_names[1])
                st.dataframe(table2, use_container_width=True, key=f"table_{param_names[1]}")

    time.sleep(1)