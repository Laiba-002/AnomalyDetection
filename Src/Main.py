# import streamlit as st
# import pandas as pd
# import numpy as np
# from sklearn.ensemble import IsolationForest
# from datetime import datetime
# import plotly.graph_objects as go
# from plotly.subplots import make_subplots
# import time
# import psycopg2

# # Streamlit config
# st.set_page_config(page_title="Live Anomaly Detection", layout="wide")
# # st.title("📉  Anomaly Detection")
# st.markdown(
#     """
#     <h1 style='text-align: center; color: #1f77b4; font-size: 42px;'>📉 Real-Time Anomaly Detection</h1>
#     """,
#     unsafe_allow_html=True
# )

# # --- Database connection using Streamlit secrets ---
# def get_db_connection():
#     conn = psycopg2.connect(
#         host=st.secrets["postgres"]["host"],
#         port=st.secrets["postgres"]["port"],
#         user=st.secrets["postgres"]["user"],
#         password=st.secrets["postgres"]["password"],
#         dbname=st.secrets["postgres"]["dbname"]
#     )
#     return conn

# # --- Fetch machine/unit list for dropdown ---
# @st.cache_data(ttl=600)
# def get_unit_list():
#     conn = get_db_connection()
#     df = pd.read_sql_query("SELECT * FROM unit u", conn)
#     print(df.head)
#     conn.close()
#     return df

# # --- Place dropdown at the top of the app ---
# unit_df = get_unit_list()
# unit_options = unit_df["pu_desc"].tolist()  # Use machine name for dropdown values
# selected_unit = st.selectbox("", unit_options, key="unit_select", placeholder="Select Machine")
# # --- Squeeze/shorten the selection box width ---
# st.markdown("""
#     <style>
#     div[data-baseweb='select'] > div {
#         max-width: 350px !important;
#         min-width: 200px !important;
#     }
#     div[data-testid='stSelectbox'] > div {
#         max-width: 350px !important;
#         min-width: 200px !important;
#     }
#     </style>
#     """, unsafe_allow_html=True)

# # Get pu_id for the selected machine
# selected_pu_id = unit_df[unit_df["pu_desc"] == selected_unit]["pu_id"].values[0]

# # --- Fetch parameters for selected machine ---
# @st.cache_data(ttl=60)
# def get_parameters_for_machine(pu_id):
#     conn = get_db_connection()
#     query = '''
#         SELECT v.*,vg.*,cast(ps.lsl AS float) as  lrl,cast(ps.lcl as float) lsl, cast(ps.lwl AS float), CAST(ps.tgt as float),cast(ps.uwl AS float),cast(ps.ucl AS float) AS usl,cast(ps.usl AS float) AS url FROM variables v
#         JOIN variable_groups vg ON v.vg_id = vg.vg_id
#         JOIN parameterspecifications ps ON v.var_id = ps.parameter_id
#         WHERE vg.pu_id = %s;
#         '''
#     df = pd.read_sql_query(query, conn, params=(int(pu_id),))
#     conn.close()
#     return df

# # Fetch parameter data for selected machine
# param_df = get_parameters_for_machine(selected_pu_id)

# # Show parameter names for debug 
# # st.write("Parameters for selected machine:", param_df[["var_id", "var_desc","lrl","lsl","lwl","tgt","uwl","usl","url"]])


# # --- Generate new random record for a parameter with dynamic specs ---
# def generate_new_record_for_param(i, param_row):
#     # Use the parameter's name and its spec range
#     param_name = param_row['var_desc']
#     lwl = param_row['lwl']
#     uwl = param_row['uwl']
#     # Generate a value in a wider range for more anomalies
#     min_val = param_row['lrl'] if not pd.isnull(param_row['lrl']) else lwl - 10
#     max_val = param_row['url'] if not pd.isnull(param_row['url']) else uwl + 10
#     val = round(np.random.uniform(min_val, max_val), 2)
#     return {
#         'id': i,
#         'datetime': datetime.now(),
#         'parameter': param_name,
#         'value': val
#     }

# # --- Rule-based classification using dynamic specs ---
# def classify_rule_dynamic(row, param_specs):
#     val = row['value']
#     lrl = param_specs['lrl']
#     lsl = param_specs['lsl']
#     lwl = param_specs['lwl']
#     uwl = param_specs['uwl']
#     usl = param_specs['usl']
#     url = param_specs['url']
#     if val < lrl or val > url:
#         return -1  # Outlier
#     elif val < lsl or val > usl:
#         return 1   # Warning
#     elif lwl <= val <= uwl:
#         return 0   # Normal
#     else:
#         return 1   # Warning (between lsl/lwl or uwl/usl)

# # Initialize session state
# if "data" not in st.session_state:
#     st.session_state.data = pd.DataFrame(columns=["id", "datetime", "parameter", "value", "rule_based", "model", "source"])
#     st.session_state.model = IsolationForest(contamination=0.1, n_estimators=100, random_state=42)
# if "counter" not in st.session_state:
#     st.session_state.counter = 1

# # --- Main real-time data loop for dynamic parameters ---
# placeholder = st.empty()

# # Get all parameter names for the selected machine
# param_names = param_df['var_desc'].tolist()

# # Initialize session state for each parameter
# def init_param_states():
#     for pname in param_names:
#         if f"show_table_{pname}" not in st.session_state:
#             st.session_state[f"show_table_{pname}"] = False
# init_param_states()

# # DataFrame to hold all generated data
# if 'data' not in st.session_state:
#     st.session_state.data = pd.DataFrame(columns=['id', 'datetime', 'parameter', 'value', 'rule_based', 'model', 'source'])
# if 'counter' not in st.session_state:
#     st.session_state.counter = 1

# while True:
#     # Generate a new record for each parameter
#     new_rows = []
#     for _, param_row in param_df.iterrows():
#         new_rows.append(generate_new_record_for_param(st.session_state.counter, param_row))
#         st.session_state.counter += 1
#     new_df = pd.DataFrame(new_rows)
#     # Merge specs for rule-based classification
#     merged = pd.merge(new_df, param_df, left_on='parameter', right_on='var_desc', how='left')
#     merged['rule_based'] = merged.apply(lambda row: classify_rule_dynamic(row, row), axis=1)
#     # Model-based detection (Isolation Forest per parameter)
#     merged['model'] = 0  # Default to normal
#     for pname in param_names:
#         pdata = merged[merged['parameter'] == pname]
#         # Only fit if enough normal data
#         normal_vals = pdata[pdata['rule_based'] != -1]['value'].values.reshape(-1, 1)
#         if len(normal_vals) > 30:
#             model = IsolationForest(contamination=0.1, n_estimators=100, random_state=42)
#             model.fit(normal_vals)
#             merged.loc[merged['parameter'] == pname, 'model'] = model.predict(pdata['value'].values.reshape(-1, 1))
#         else:
#             merged.loc[merged['parameter'] == pname, 'model'] = 0
#     def get_source(row):
#         if row['model'] == -1 and row['rule_based'] == -1:
#             return 'Both'
#         elif row['model'] == -1:
#             return 'Model'
#         elif row['rule_based'] == -1:
#             return 'Rule-based'
#         elif row['rule_based'] == 1:
#             return 'Warning'
#         else:
#             return 'Normal'
#     merged['source'] = merged.apply(get_source, axis=1)
#     st.session_state.data = pd.concat([st.session_state.data, merged[['id','datetime','parameter','value','rule_based','model','source']]], ignore_index=True)

#     # --- Spec line colors for consistent charting ---
#     spec_colors = {
#         'lrl': 'red',
#         'lsl': 'orange',
#         'lwl': 'yellow',
#         'tgt': 'blue',
#         'uwl': 'yellow',
#         'usl': 'orange',
#         'url': 'red'
#     }

#     # --- Advanced parameter chart function ---
#     def make_parameter_chart(param_data, param_specs):
#         import plotly.graph_objects as go
#         fig = go.Figure()
#         # Add static horizontal dotted lines for spec values
#         for spec in ['lrl','lsl','lwl','tgt','uwl','usl','url']:
#             spec_value = param_specs[spec]
#             if not pd.isnull(spec_value):
#                 fig.add_hline(
#                     y=spec_value,
#                     line_dash="dot",
#                     line_color=spec_colors[spec],
#                     line_width=2 if spec == "tgt" else 1.5,
#                     annotation_text="",
#                 )
#                 # Add invisible scatter trace for legend
#                 fig.add_trace(go.Scatter(
#                     x=[None], y=[None],
#                     mode='lines',
#                     line=dict(color=spec_colors[spec], dash='dot', width=2 if spec == "tgt" else 1.5),
#                     name=f'{spec.upper()}: {spec_value}',
#                     showlegend=True,
#                     legendgroup='spec_lines',
#                     hoverinfo='skip'
#                 ))
#         # Add line trace for all data
#         fig.add_trace(go.Scatter(
#             x=param_data['datetime'],
#             y=param_data['value'],
#             mode='lines',
#             name='Trend Line',
#             line=dict(color='lightblue', width=2),
#             hoverinfo='skip',
#             showlegend=False
#         ))
#         # Color mapping for different sources
#         color_map = {
#             'Both': 'red',
#             'Model': 'red',
#             'Rule-based': 'red',
#             'Warning': 'orange',
#             'Normal': 'green'
#         }
#         legend_map = {
#             'Both': 'Anomaly',
#             'Model': 'Anomaly',
#             'Rule-based': 'Anomaly',
#             'Warning': 'Warning',
#             'Normal': 'Normal'
#         }
#         marker_symbol_map = {
#             'Anomaly': 'circle',
#             'Warning': 'diamond',
#             'Normal': 'circle-open'
#         }
#         added_legend_groups = set()
#         for source in param_data['source'].unique():
#             source_data = param_data[param_data['source'] == source]
#             legend_group = legend_map.get(source, source)
#             marker_symbol = marker_symbol_map.get(legend_group, 'circle')
#             fig.add_trace(go.Scatter(
#                 x=source_data['datetime'],
#                 y=source_data['value'],
#                 mode='markers',
#                 name=legend_group,
#                 legendgroup=legend_group,
#                 showlegend=legend_group not in added_legend_groups,
#                 marker=dict(
#                     color=color_map.get(source, 'blue'),
#                     size=10 if legend_group == 'Anomaly' else 8,
#                     line=dict(width=2 if legend_group == 'Anomaly' else 1, color='white'),
#                     symbol=marker_symbol
#                 ),
#                 customdata=source_data[['id', 'parameter']],
#                 hovertemplate='<b>%{customdata[1]}</b><br>' +
#                              'Time: %{x}<br>' +
#                              'Value: %{y}<br>' +
#                              'ID: %{customdata[0]}<br>' +
#                              f'Source: {source}' +
#                              '<extra></extra>'
#             ))
#             added_legend_groups.add(legend_group)
#         fig.update_layout(
#             xaxis_title="Time",
#             yaxis_title="Value",
#             height=600,
#             width=800,
#             hovermode='closest',
#             plot_bgcolor='rgba(248, 249, 250, 0.8)',
#             paper_bgcolor='white',
#             legend=dict(
#                 orientation="v",
#                 yanchor="top",
#                 y=1,
#                 xanchor="left",
#                 x=1.02,
#                 bgcolor="rgba(255, 255, 255, 0.95)",
#                 bordercolor="rgba(0, 0, 0, 0.1)",
#                 borderwidth=1,
#                 font=dict(size=10),
#                 itemsizing="constant"
#             ),
#             xaxis=dict(
#                 rangeselector=dict(
#                     buttons=list([
#                         dict(count=1, label="1m", step="minute", stepmode="backward"),
#                         dict(count=5, label="5m", step="minute", stepmode="backward"),
#                         dict(count=15, label="15m", step="minute", stepmode="backward"),
#                         dict(count=30, label="30m", step="minute", stepmode="backward"),
#                         dict(count=1, label="1h",step="hour", stepmode="backward"),
#                         dict(step="all", label="All")
#                     ]),
#                     y=1.15,
#                     x=0,
#                     bgcolor="rgba(255, 255, 255, 0.8)",
#                     bordercolor="rgba(0, 0, 0, 0.1)",
#                     borderwidth=1
#                 ),
#                 rangeslider=dict(
#                     visible=True,
#                     bgcolor="rgba(248, 249, 250, 0.5)"
#                 ),
#                 type="date",
#                 gridcolor="rgba(200, 200, 200, 0.3)",
#                 showgrid=True,
#             ),
#             yaxis=dict(
#                 gridcolor="rgba(200, 200, 200, 0.3)",
#                 showgrid=True,
#                 zeroline=True,
#                 zerolinecolor="rgba(200, 200, 200, 0.5)"
#             ),
#             transition={'duration': 0},
#             margin=dict(l=60, r=150, t=80, b=60)
#         )
#         return fig

#     with placeholder.container():
#         # st.subheader(f"📈 Interactive Charts for {selected_unit}")
#         # st.markdown("<div style='height: 24px;'></div>", unsafe_allow_html=True)  # Add vertical space
#         cols = st.columns(2)
#         for idx, pname in enumerate(param_names):
#             with cols[idx % 2]:
#                 st.write(f"**{pname}**")
#                 param_specs = param_df[param_df['var_desc'] == pname].iloc[0]
#                 pdata = st.session_state.data[st.session_state.data['parameter'] == pname]
#                 fig = make_parameter_chart(pdata, param_specs)
#                 st.plotly_chart(fig, use_container_width=True)
#                 if st.session_state[f"show_table_{pname}"]:
#                     st.write(f"**📋 {pname} Data Table**")
#                     table = pdata.tail(15)[['id','datetime','value','source']]
#                     st.dataframe(table, use_container_width=True)
#     time.sleep(5)





import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from datetime import datetime
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
import psycopg2

# Streamlit config
st.set_page_config(page_title="Live Anomaly Detection", layout="wide")
st.markdown(
    """
    <h1 style='text-align: center; color: #1f77b4; font-size: 42px;'>📉 Real-Time Anomaly Detection</h1>
    """,
    unsafe_allow_html=True
)

# --- Database connection using Streamlit secrets ---
def get_db_connection():
    conn = psycopg2.connect(
        host=st.secrets["postgres"]["host"],
        port=st.secrets["postgres"]["port"],
        user=st.secrets["postgres"]["user"],
        password=st.secrets["postgres"]["password"],
        dbname=st.secrets["postgres"]["dbname"]
    )
    return conn

# --- Fetch machine/unit list for dropdown ---
@st.cache_data(ttl=600)
def get_unit_list():
    conn = get_db_connection()
    df = pd.read_sql_query("SELECT * FROM unit u", conn)
    # print(df.head)
    conn.close()
    return df

# --- Place dropdown at the top of the app ---
unit_df = get_unit_list()
unit_options = unit_df["pu_desc"].tolist()
selected_unit = st.selectbox("", unit_options, key="unit_select", placeholder="Select Machine")

# --- Customize the selection box width ---
st.markdown("""
    <style>
    div[data-baseweb='select'] > div {
        max-width: 350px !important;
        min-width: 200px !important;
    }
    div[data-testid='stSelectbox'] > div {
        max-width: 350px !important;
        min-width: 200px !important;
    }
    </style>
    """, unsafe_allow_html=True)

# Get pu_id for the selected machine
selected_pu_id = unit_df[unit_df["pu_desc"] == selected_unit]["pu_id"].values[0]

# --- Fetch parameters for selected machine ---
@st.cache_data(ttl=60)
def get_parameters_for_machine(pu_id):
    conn = get_db_connection()
    query = '''
        SELECT v.*,vg.*,cast(ps.lsl AS float) as  lrl,cast(ps.lcl as float) lsl, cast(ps.lwl AS float), CAST(ps.tgt as float),cast(ps.uwl AS float),cast(ps.ucl AS float) AS usl,cast(ps.usl AS float) AS url FROM variables v
        JOIN variable_groups vg ON v.vg_id = vg.vg_id
        JOIN parameterspecifications ps ON v.var_id = ps.parameter_id
        WHERE vg.pu_id = %s;
        '''
    df = pd.read_sql_query(query, conn, params=(int(pu_id),))
    conn.close()
    return df

# Fetch parameter data for selected machine
param_df = get_parameters_for_machine(selected_pu_id)

# --- Generate new random record with anomaly generation ---
def generate_new_record_for_param(i, param_row):
    param_name = param_row['var_desc']
    lwl = param_row['lwl']
    uwl = param_row['uwl']
    lrl = param_row['lrl'] if not pd.isnull(param_row['lrl']) else lwl - 10
    url = param_row['url'] if not pd.isnull(param_row['url']) else uwl + 10
    
    # Introduce controlled anomaly generation
    # 80% normal values, 15% warnings, 5% true anomalies
    rand = np.random.random()
    
    if rand < 0.05:  # 5% true anomalies - way outside limits
        if np.random.random() < 0.5:
            val = np.random.uniform(lrl - 20, lrl)  # Below lower limit
        else:
            val = np.random.uniform(url, url + 20)  # Above upper limit
    elif rand < 0.20:  # 15% warnings - between control and spec limits
        if np.random.random() < 0.5:
            val = np.random.uniform(lrl, param_row['lsl'])  # Warning zone low
        else:
            val = np.random.uniform(param_row['usl'], url)  # Warning zone high
    else:  # 80% normal values - within control limits
        val = np.random.normal((lwl + uwl) / 2, (uwl - lwl) / 6)  # Normal distribution
        val = np.clip(val, lwl, uwl)  # Ensure within normal range
    
    return {
        'id': i,
        'datetime': datetime.now(),
        'parameter': param_name,
        'value': round(val, 2)
    }

# --- Rule-based classification with proper thresholds ---
def classify_rule_dynamic(row, param_specs):
    val = row['value']
    lrl = param_specs['lrl']
    lsl = param_specs['lsl'] 
    lwl = param_specs['lwl']
    uwl = param_specs['uwl']
    usl = param_specs['usl']
    url = param_specs['url']
    
    #  Proper anomaly detection logic
    if pd.isnull(lrl) or pd.isnull(url):
        # If limits not defined, use wider range
        if val < (lwl - 15) or val > (uwl + 15):
            return -1  # Outlier/Anomaly
    else:
        if val < lrl or val > url:
            return -1  # Outlier/Anomaly
    
    # Warning zones
    if not pd.isnull(lsl) and not pd.isnull(usl):
        if (lrl <= val < lsl) or (usl < val <= url):
            return 1   # Warning
    
    # Normal range
    if lwl <= val <= uwl:
        return 0   # Normal
    
    return 1   # Default to warning if unclear

# Initialize session state
if "data" not in st.session_state:
    st.session_state.data = pd.DataFrame(columns=["id", "datetime", "parameter", "value", "rule_based", "model", "source"])
if "counter" not in st.session_state:
    st.session_state.counter = 1
if "models" not in st.session_state:
    st.session_state.models = {}  # Store models per parameter

# Get all parameter names for the selected machine
param_names = param_df['var_desc'].tolist()

# Initialize session state for each parameter
def init_param_states():
    for pname in param_names:
        if f"show_table_{pname}" not in st.session_state:
            st.session_state[f"show_table_{pname}"] = False
init_param_states()

# --- Main real-time data loop ---
placeholder = st.empty()

while True:
    # Generate new records for each parameter
    new_rows = []
    for _, param_row in param_df.iterrows():
        new_rows.append(generate_new_record_for_param(st.session_state.counter, param_row))
        st.session_state.counter += 1
    
    new_df = pd.DataFrame(new_rows)
    
    # Merge with parameter specifications for rule-based classification
    merged = pd.merge(new_df, param_df, left_on='parameter', right_on='var_desc', how='left')
    merged['rule_based'] = merged.apply(lambda row: classify_rule_dynamic(row, row), axis=1)
    
    # FIXED: Model-based detection with proper training
    merged['model'] = 0  # Default to normal
    
    for pname in param_names:
        param_data = st.session_state.data[st.session_state.data['parameter'] == pname]
        current_param_data = merged[merged['parameter'] == pname]
        
        if len(param_data) > 50:  
            # Get normal values for training (exclude anomalies)
            normal_data = param_data[param_data['rule_based'] == 0]
            
            if len(normal_data) > 30:
                # Train or retrain the model
                X_train = normal_data['value'].values.reshape(-1, 1)
                
                if pname not in st.session_state.models:
                    st.session_state.models[pname] = IsolationForest(
                        contamination=0.1, 
                        n_estimators=100, 
                        random_state=42
                    )
                
                st.session_state.models[pname].fit(X_train)
                
                # Predict anomalies for current data
                X_current = current_param_data['value'].values.reshape(-1, 1)
                predictions = st.session_state.models[pname].predict(X_current)
                merged.loc[merged['parameter'] == pname, 'model'] = predictions

    # FIXED: Source classification logic
    def get_source(row):
        rule_anomaly = row['rule_based'] == -1
        model_anomaly = row['model'] == -1
        
        if rule_anomaly and model_anomaly:
            return 'Both'
        elif model_anomaly:
            return 'Model'
        elif rule_anomaly:
            return 'Rule-based'
        elif row['rule_based'] == 1:
            return 'Warning'
        else:
            return 'Normal'
    
    merged['source'] = merged.apply(get_source, axis=1)
    
    # Add to session state data
    st.session_state.data = pd.concat([
        st.session_state.data, 
        merged[['id','datetime','parameter','value','rule_based','model','source']]
    ], ignore_index=True)
    
    # Keep only recent data to prevent memory issues
    if len(st.session_state.data) > 1000:
        st.session_state.data = st.session_state.data.tail(800)

    # --- Spec line colors for consistent charting ---
    spec_colors = {
        'lrl': 'red',
        'lsl': 'orange', 
        'lwl': 'yellow',
        'tgt': 'blue',
        'uwl': 'brown',
        'usl': 'black',
        'url': 'purple'
    }

    # --- Enhanced parameter chart function ---
    def make_parameter_chart(param_data, param_specs):
        fig = go.Figure()
        
        # Add specification lines
        for spec in ['lrl','lsl','lwl','tgt','uwl','usl','url']:
            spec_value = param_specs[spec]
            if not pd.isnull(spec_value):
                fig.add_hline(
                    y=spec_value,
                    line_dash="dot",
                    line_color=spec_colors[spec],
                    line_width=2 if spec == "tgt" else 1.5,
                    annotation_text="",
                )
                # Add invisible scatter trace for legend
                fig.add_trace(go.Scatter(
                    x=[None], y=[None],
                    mode='lines',
                    line=dict(color=spec_colors[spec], dash='dot', width=2 if spec == "tgt" else 1.5),
                    name=f'{spec.upper()}: {spec_value}',
                    showlegend=True,
                    legendgroup='spec_lines',
                    hoverinfo='skip'
                ))
        
        # Add trend line
        if len(param_data) > 1:
            fig.add_trace(go.Scatter(
                x=param_data['datetime'],
                y=param_data['value'],
                mode='lines',
                name='Trend Line',
                line=dict(color='lightblue', width=2),
                hoverinfo='skip',
                showlegend=False
            ))
        
        # FIXED: Enhanced color mapping with better visibility
        color_map = {
            'Both': 'darkred',
            'Model': 'red', 
            'Rule-based': 'crimson',
            'Warning': 'orange',
            'Normal': 'green'
        }
        
        legend_map = {
            'Both': 'Anomaly (Both)',
            'Model': 'Anomaly (Model)', 
            'Rule-based': 'Anomaly (Rules)',
            'Warning': 'Warning',
            'Normal': 'Normal'
        }
        
        marker_symbol_map = {
            'Anomaly (Both)': 'x',
            'Anomaly (Model)': 'triangle-up',
            'Anomaly (Rules)': 'diamond',
            'Warning': 'square',
            'Normal': 'circle'
        }
        
        added_legend_groups = set()
        
        for source in param_data['source'].unique():
            source_data = param_data[param_data['source'] == source]
            legend_group = legend_map.get(source, source)
            marker_symbol = marker_symbol_map.get(legend_group, 'circle')
            
            fig.add_trace(go.Scatter(
                x=source_data['datetime'],
                y=source_data['value'],
                mode='markers',
                name=legend_group,
                legendgroup=legend_group,
                showlegend=legend_group not in added_legend_groups,
                marker=dict(
                    color=color_map.get(source, 'blue'),
                    size=12 if 'Anomaly' in legend_group else 8,
                    line=dict(width=2, color='white'),
                    symbol=marker_symbol
                ),
                customdata=source_data[['id', 'parameter']],
                hovertemplate='<b>%{customdata[1]}</b><br>' +
                             'Time: %{x}<br>' +
                             'Value: %{y}<br>' +
                             'ID: %{customdata[0]}<br>' +
                             f'Source: {source}' +
                             '<extra></extra>'
            ))
            added_legend_groups.add(legend_group)
        
        fig.update_layout(
            xaxis_title="Time",
            yaxis_title="Value", 
            height=600,
            width=800,
            hovermode='closest',
            plot_bgcolor='rgba(248, 249, 250, 0.8)',
            paper_bgcolor='white',
            legend=dict(
                orientation="v",
                yanchor="top",
                y=1,
                xanchor="left", 
                x=1.02,
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
            ),
            yaxis=dict(
                gridcolor="rgba(200, 200, 200, 0.3)",
                showgrid=True,
                zeroline=True,
                zerolinecolor="rgba(200, 200, 200, 0.5)"
            ),
            transition={'duration': 0},
            margin=dict(l=60, r=150, t=80, b=60)
        )
        return fig

    # Display charts
    with placeholder.container():
        cols = st.columns(2)
        for idx, pname in enumerate(param_names):
            with cols[idx % 2]:
                st.write(f"**{pname}**")
                st.markdown("<div style='height: 24px;'></div>", unsafe_allow_html=True)
                param_specs = param_df[param_df['var_desc'] == pname].iloc[0]
                pdata = st.session_state.data[st.session_state.data['parameter'] == pname]
                # Only keep the last 100 points for plotting
                pdata = pdata.tail(100)

                if len(pdata) > 0:
                    fig = make_parameter_chart(pdata, param_specs)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Show anomaly statistics
                    if len(pdata) > 0:
                        anomaly_count = len(pdata[pdata['source'].isin(['Both', 'Model', 'Rule-based'])])
                        warning_count = len(pdata[pdata['source'] == 'Warning'])
                        total_count = len(pdata)
                        
                        # st.write(f"📊 **Stats:** {anomaly_count} anomalies, {warning_count} warnings out of {total_count} total points")
                    
                    if st.session_state[f"show_table_{pname}"]:
                        st.write(f"**📋 {pname} Data Table**")
                        table = pdata.tail(15)[['id','datetime','value','source']]
                        st.dataframe(table, use_container_width=True)
    
    time.sleep(5)
