import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Simulation function, updated to handle custom vs. cascaded cache hit/miss rates
def simulate_performance(data_size_mb, l1_rate, l2_rate, l3_rate, ram_capacity_gb, secondary_type, use_custom=False):
    # Fixed latencies in nanoseconds
    l1_time = 1
    l2_time = 3
    l3_time = 10
    ram_time = 100
    secondary_time = 100000 if secondary_type == "SSD" else 10000000

    if use_custom:
        # Use custom percentages where rates sum to less than or equal 100.
        l1_frac = l1_rate / 100.0
        l2_frac = l2_rate / 100.0
        l3_frac = l3_rate / 100.0
        miss_frac = (100 - (l1_rate + l2_rate + l3_rate)) / 100.0
        avg_cache_time = l1_frac * l1_time + l2_frac * l2_time + l3_frac * l3_time + miss_frac * ram_time
    else:
        # Cascaded simulation: sequential dependency
        l1_hit = l1_rate / 100.0
        l2_hit = l2_rate / 100.0
        l3_hit = l3_rate / 100.0
        avg_cache_time = (l1_hit * l1_time +
                         (1 - l1_hit) * (l2_hit * l2_time +
                                         (1 - l2_hit) * (l3_hit * l3_time +
                                                         (1 - l3_hit) * ram_time))
                        )

    # Handle data exceeding RAM capacity
    ram_capacity_mb = ram_capacity_gb * 1024
    if data_size_mb > ram_capacity_mb:
        fraction_secondary = (data_size_mb - ram_capacity_mb) / data_size_mb
    else:
        fraction_secondary = 0

    avg_access_time = (1 - fraction_secondary) * avg_cache_time + fraction_secondary * secondary_time

    # Calculate number of memory accesses (assume 64 bytes per access)
    num_accesses = data_size_mb * 1024 * 1024 / 64
    total_exec_time_sec = num_accesses * avg_access_time / 1e9  # Convert ns to seconds
    data_transfer_rate = data_size_mb / total_exec_time_sec if total_exec_time_sec > 0 else 0

    return avg_access_time, total_exec_time_sec, data_transfer_rate, avg_cache_time, fraction_secondary

# Initialize session state defaults for scenario parameters
if 'data_size_mb' not in st.session_state:
    st.session_state.data_size_mb = 100
if 'access_pattern' not in st.session_state:
    st.session_state.access_pattern = "Sequential"
if 'ram_capacity_gb' not in st.session_state:
    st.session_state.ram_capacity_gb = 8
if 'secondary_type' not in st.session_state:
    st.session_state.secondary_type = "SSD"
if 'l1_hit_rate' not in st.session_state:
    st.session_state.l1_hit_rate = 90
if 'l2_hit_rate' not in st.session_state:
    st.session_state.l2_hit_rate = 80
if 'l3_hit_rate' not in st.session_state:
    st.session_state.l3_hit_rate = 70
if 'l1_hit_custom' not in st.session_state:
    st.session_state.l1_hit_custom = 80
if 'l2_hit_custom' not in st.session_state:
    st.session_state.l2_hit_custom = 15
if 'l3_hit_custom' not in st.session_state:
    st.session_state.l3_hit_custom = 4

st.title("Memory-Storage Hierarchy Simulator")

# Sidebar: Predefined Scenarios
st.sidebar.header("Predefined Scenarios")
col1, col2, col3 = st.sidebar.columns(3)
if col1.button("Video Streaming"):
    st.session_state.data_size_mb = 500
    st.session_state.access_pattern = "Sequential"
    st.session_state.ram_capacity_gb = 16
    st.session_state.secondary_type = "SSD"
    st.session_state.l1_hit_rate = 85
    st.session_state.l2_hit_rate = 10
    st.session_state.l3_hit_rate = 4
    st.session_state.l1_hit_custom = 85
    st.session_state.l2_hit_custom = 10
    st.session_state.l3_hit_custom = 4

if col2.button("Gaming"):
    st.session_state.data_size_mb = 200
    st.session_state.access_pattern = "Random"
    st.session_state.ram_capacity_gb = 8
    st.session_state.secondary_type = "SSD"
    st.session_state.l1_hit_rate = 90
    st.session_state.l2_hit_rate = 5
    st.session_state.l3_hit_rate = 3
    st.session_state.l1_hit_custom = 90
    st.session_state.l2_hit_custom = 5
    st.session_state.l3_hit_custom = 3

if col3.button("Database Query"):
    st.session_state.data_size_mb = 100
    st.session_state.access_pattern = "Random"
    st.session_state.ram_capacity_gb = 32
    st.session_state.secondary_type = "HDD"
    st.session_state.l1_hit_rate = 70
    st.session_state.l2_hit_rate = 20
    st.session_state.l3_hit_rate = 5
    st.session_state.l1_hit_custom = 70
    st.session_state.l2_hit_custom = 20
    st.session_state.l3_hit_custom = 5

# Sidebar: Memory Levels & Data Access Configuration
st.sidebar.header("Memory Levels Configuration")
l1_size_kb = st.sidebar.slider("L1 Cache Size (KB)", 16, 128, 32)
l2_size_kb = st.sidebar.slider("L2 Cache Size (KB)", 128, 2048, 256)
l3_size_kb = st.sidebar.slider("L3 Cache Size (KB)", 2048, 16384, 8192)
ram_capacity_gb = st.sidebar.slider("RAM Capacity (GB)", 2, 64, st.session_state.ram_capacity_gb, key="ram_capacity_gb")
secondary_type = st.sidebar.selectbox("Secondary Storage Type", ["SSD", "HDD"], index=0 if st.session_state.secondary_type=="SSD" else 1, key="secondary_type")
secondary_capacity_gb = st.sidebar.slider("Secondary Storage Capacity (GB)", 64, 2048, 256)

st.sidebar.header("Data Access Simulation")
data_size_mb = st.sidebar.number_input("Data Size (MB)", min_value=1, value=st.session_state.data_size_mb, key="data_size_mb")
access_pattern = st.sidebar.radio("Access Pattern", ["Sequential", "Random"], index=0 if st.session_state.access_pattern=="Sequential" else 1, key="access_pattern")

# Custom cache hit/miss rates option
use_custom_cache = st.sidebar.checkbox("Use Custom Cache Hit/Miss Rates", value=False, key="use_custom_cache")
if use_custom_cache:
    l1_hit_custom = st.sidebar.slider("L1 Hit Rate (%)", 0, 100, st.session_state.l1_hit_custom, key="l1_hit_custom")
    l2_hit_custom = st.sidebar.slider("L2 Hit Rate (%)", 0, 100, st.session_state.l2_hit_custom, key="l2_hit_custom")
    l3_hit_custom = st.sidebar.slider("L3 Hit Rate (%)", 0, 100, st.session_state.l3_hit_custom, key="l3_hit_custom")
    total_custom = l1_hit_custom + l2_hit_custom + l3_hit_custom
    miss_rate_custom = max(0, 100 - total_custom)
    st.sidebar.write(f"Miss Rate: {miss_rate_custom}% (automatically computed)")
    # Use custom rates for simulation
    l1_rate = l1_hit_custom
    l2_rate = l2_hit_custom
    l3_rate = l3_hit_custom
else:
    l1_hit_rate = st.sidebar.slider("L1 Hit Rate (%)", 0, 100, st.session_state.l1_hit_rate, key="l1_hit_rate")
    l2_hit_rate = st.sidebar.slider("L2 Hit Rate (%)", 0, 100, st.session_state.l2_hit_rate, key="l2_hit_rate")
    l3_hit_rate = st.sidebar.slider("L3 Hit Rate (%)", 0, 100, st.session_state.l3_hit_rate, key="l3_hit_rate")
    l1_rate = l1_hit_rate
    l2_rate = l2_hit_rate
    l3_rate = l3_hit_rate

# Run simulation
(avg_access_time, total_exec_time_sec,
 data_transfer_rate, avg_cache_time, fraction_secondary) = simulate_performance(
    data_size_mb, l1_rate, l2_rate, l3_rate, ram_capacity_gb, secondary_type, use_custom_cache
)

st.header("Performance Metrics")
st.write(f"**Average Access Time:** {avg_access_time:.2f} ns")
st.write(f"**Total Execution Time:** {total_exec_time_sec:.4f} sec")
st.write(f"**Data Transfer Rate:** {data_transfer_rate:.2f} MB/s")

# Bar Chart: Memory Level Latencies with labeled values
labels = ['L1', 'L2', 'L3', 'RAM', 'Secondary']
latencies = [1, 3, 10, 100, 100000 if secondary_type == "SSD" else 10000000]

fig, ax = plt.subplots()
bars = ax.bar(labels, latencies, color='skyblue')
ax.set_ylabel("Latency (ns)")
ax.set_title("Memory Level Latencies")
# Label each bar with its latency value
for bar, latency in zip(bars, latencies):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, height, f"{latency}", ha='center', va='bottom')
st.pyplot(fig)

# Pie Chart: Access Time Distribution
if use_custom_cache:
    # For custom simulation, percentages are directly provided
    miss_frac = (100 - (l1_rate + l2_rate + l3_rate)) / 100.0
    contributions = [l1_rate / 100.0, l2_rate / 100.0, l3_rate / 100.0, miss_frac]
    # Time contributions: assign RAM time for cache miss
    time_values = [contributions[0] * 1, contributions[1] * 3, contributions[2] * 10, contributions[3] * 100]
    pie_labels = ['L1', 'L2', 'L3', 'RAM']
else:
    # For cascaded simulation, compute effective contributions
    eff_l1 = l1_rate / 100.0
    eff_l2 = (1 - l1_rate/100.0) * (l2_rate / 100.0)
    eff_l3 = (1 - l1_rate/100.0) * (1 - l2_rate/100.0) * (l3_rate / 100.0)
    eff_ram = (1 - l1_rate/100.0) * (1 - l2_rate/100.0) * (1 - l3_rate/100.0)
    contributions = [eff_l1, eff_l2, eff_l3, eff_ram]
    time_values = [eff_l1 * 1, eff_l2 * 3, eff_l3 * 10, eff_ram * 100]
    pie_labels = ['L1', 'L2', 'L3', 'RAM']

fig2, ax2 = plt.subplots()
ax2.pie(time_values, labels=pie_labels, autopct='%1.1f%%', startangle=90)
ax2.set_title("Access Time Distribution")
st.pyplot(fig2)

# Cost Analysis Section
if st.sidebar.checkbox("Show Cost Analysis"):
    st.header("Cost Analysis")
    # Arbitrary cost rates
    cost_l1 = l1_size_kb * 0.005
    cost_l2 = l2_size_kb * 0.003
    cost_l3 = l3_size_kb * 0.001
    cost_ram = ram_capacity_gb * 1024 * 0.05
    cost_secondary = secondary_capacity_gb * (0.1 if secondary_type == "SSD" else 0.05)
    total_cost = cost_l1 + cost_l2 + cost_l3 + cost_ram + cost_secondary

    st.write(f"**L1 Cache Cost:** ${cost_l1:.2f}")
    st.write(f"**L2 Cache Cost:** ${cost_l2:.2f}")
    st.write(f"**L3 Cache Cost:** ${cost_l3:.2f}")
    st.write(f"**RAM Cost:** ${cost_ram:.2f}")
    st.write(f"**Secondary Storage Cost:** ${cost_secondary:.2f}")
    st.write(f"**Total Cost:** ${total_cost:.2f}")

    cost_labels = ['L1', 'L2', 'L3', 'RAM', 'Secondary']
    cost_values = [cost_l1, cost_l2, cost_l3, cost_ram, cost_secondary]
    fig3, ax3 = plt.subplots()
    ax3.bar(cost_labels, cost_values, color='lightgreen')
    ax3.set_ylabel("Cost ($)")
    ax3.set_title("Cost Breakdown")
    st.pyplot(fig3)

# Prepare CSV export summary
summary_data = {
    "Parameter": [
        "Data Size (MB)",
        "Access Pattern",
        "L1 Cache Size (KB)",
        "L2 Cache Size (KB)",
        "L3 Cache Size (KB)",
        "RAM Capacity (GB)",
        "Secondary Storage Type",
        "Secondary Storage Capacity (GB)",
    ],
    "Value": [
        data_size_mb,
        access_pattern,
        l1_size_kb,
        l2_size_kb,
        l3_size_kb,
        ram_capacity_gb,
        secondary_type,
        secondary_capacity_gb,
    ]
}

if use_custom_cache:
    summary_data["Parameter"].extend(["L1 Hit Rate (%)", "L2 Hit Rate (%)", "L3 Hit Rate (%)", "Miss Rate (%)"])
    summary_data["Value"].extend([
        l1_rate,
        l2_rate,
        l3_rate,
        max(0, 100 - (l1_rate + l2_rate + l3_rate))
    ])
else:
    summary_data["Parameter"].extend(["L1 Hit Rate (%)", "L2 Hit Rate (%)", "L3 Hit Rate (%)", "Cascade Miss (%)"])
    cascade_miss = (1 - l1_rate/100.0) * (1 - l2_rate/100.0) * (1 - l3_rate/100.0) * 100
    summary_data["Value"].extend([l1_rate, l2_rate, l3_rate, f"{cascade_miss:.1f}"])
    
summary_data["Parameter"].extend(["Avg Access Time (ns)", "Total Exec Time (sec)", "Data Transfer Rate (MB/s)"])
summary_data["Value"].extend([f"{avg_access_time:.2f}", f"{total_exec_time_sec:.4f}", f"{data_transfer_rate:.2f}"])

df = pd.DataFrame(summary_data)

csv = df.to_csv(index=False).encode('utf-8')
st.download_button(
    label="Download CSV Summary",
    data=csv,
    file_name='simulation_summary.csv',
    mime='text/csv',
)
