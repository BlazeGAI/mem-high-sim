import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

def calculate_avg_access_time(l1_hit, l2_hit, l3_hit, l1_time, l2_time, l3_time, ram_time, secondary_time):
    # Weighted average for cache hierarchy (L1, then L2, then L3, then RAM)
    avg_cache = (l1_hit * l1_time +
                 (1 - l1_hit) * (l2_hit * l2_time +
                                 (1 - l2_hit) * (l3_hit * l3_time +
                                                 (1 - l3_hit) * ram_time)))
    return avg_cache

def simulate_performance(data_size_mb, l1_hit, l2_hit, l3_hit, ram_capacity_gb, secondary_type):
    # Fixed latencies in nanoseconds
    l1_time = 1
    l2_time = 3
    l3_time = 10
    ram_time = 100
    secondary_time = 100000 if secondary_type == "SSD" else 10000000

    avg_cache_time = calculate_avg_access_time(l1_hit, l2_hit, l3_hit,
                                               l1_time, l2_time, l3_time,
                                               ram_time, secondary_time)
    # Convert RAM capacity to MB
    ram_capacity_mb = ram_capacity_gb * 1024
    # If data exceeds RAM capacity, simulate that the excess goes to secondary storage.
    if data_size_mb > ram_capacity_mb:
        fraction_secondary = (data_size_mb - ram_capacity_mb) / data_size_mb
    else:
        fraction_secondary = 0

    avg_access_time = (1 - fraction_secondary) * avg_cache_time + fraction_secondary * secondary_time

    # Assume a block size of 64 bytes; calculate number of accesses
    num_accesses = data_size_mb * 1024 * 1024 / 64
    total_exec_time_sec = num_accesses * avg_access_time / 1e9  # convert nanoseconds to seconds
    data_transfer_rate = data_size_mb / total_exec_time_sec if total_exec_time_sec > 0 else 0

    return avg_access_time, total_exec_time_sec, data_transfer_rate, avg_cache_time, fraction_secondary

def main():
    st.title("Memory-Storage Hierarchy Simulator")

    # Sidebar: Memory Levels & Data Access Configuration
    st.sidebar.header("Memory Levels Configuration")
    l1_size_kb = st.sidebar.slider("L1 Cache Size (KB)", 16, 128, 32)
    l2_size_kb = st.sidebar.slider("L2 Cache Size (KB)", 128, 2048, 256)
    l3_size_kb = st.sidebar.slider("L3 Cache Size (KB)", 2048, 16384, 8192)
    ram_capacity_gb = st.sidebar.slider("RAM Capacity (GB)", 2, 64, 8)
    secondary_type = st.sidebar.selectbox("Secondary Storage Type", ["SSD", "HDD"])
    secondary_capacity_gb = st.sidebar.slider("Secondary Storage Capacity (GB)", 64, 2048, 256)

    st.sidebar.header("Data Access Simulation")
    data_size_mb = st.sidebar.number_input("Data Size (MB)", min_value=1, value=100)
    access_pattern = st.sidebar.radio("Access Pattern", ["Sequential", "Random"])

    st.sidebar.header("Cache Hit Rates (%)")
    l1_hit_rate = st.sidebar.slider("L1 Hit Rate (%)", 0, 100, 90)
    l2_hit_rate = st.sidebar.slider("L2 Hit Rate (%)", 0, 100, 80)
    l3_hit_rate = st.sidebar.slider("L3 Hit Rate (%)", 0, 100, 70)

    # Convert percentages to fractions
    l1_hit = l1_hit_rate / 100.0
    l2_hit = l2_hit_rate / 100.0
    l3_hit = l3_hit_rate / 100.0

    # Simulate performance
    (avg_access_time, total_exec_time_sec,
     data_transfer_rate, avg_cache_time, fraction_secondary) = simulate_performance(
         data_size_mb, l1_hit, l2_hit, l3_hit, ram_capacity_gb, secondary_type
     )

    st.header("Performance Metrics")
    st.write(f"**Average Access Time:** {avg_access_time:.2f} ns")
    st.write(f"**Total Execution Time:** {total_exec_time_sec:.4f} sec")
    st.write(f"**Data Transfer Rate:** {data_transfer_rate:.2f} MB/s")

    # Bar Chart: Memory Level Latencies
    labels = ['L1', 'L2', 'L3', 'RAM', 'Secondary']
    latencies = [1, 3, 10, 100, 100000 if secondary_type == "SSD" else 10000000]

    fig, ax = plt.subplots()
    ax.bar(labels, latencies, color='skyblue')
    ax.set_ylabel("Latency (ns)")
    ax.set_title("Memory Level Latencies")
    st.pyplot(fig)

    # Pie Chart: Access Time Distribution
    eff_l1 = l1_hit
    eff_l2 = (1 - l1_hit) * l2_hit
    eff_l3 = (1 - l1_hit) * (1 - l2_hit) * l3_hit
    eff_ram = (1 - l1_hit) * (1 - l2_hit) * (1 - l3_hit) * (1 - fraction_secondary)
    eff_secondary = (1 - l1_hit) * (1 - l2_hit) * (1 - l3_hit) * fraction_secondary

    contributions = [eff_l1, eff_l2, eff_l3, eff_ram, eff_secondary]
    time_contributions = [contributions[i] * latencies[i] for i in range(len(contributions))]

    fig2, ax2 = plt.subplots()
    ax2.pie(time_contributions, labels=labels, autopct='%1.1f%%', startangle=90)
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

if __name__ == '__main__':
    main()
