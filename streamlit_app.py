import streamlit as st
import pandas as pd
import altair as alt
import requests
import json
import scipy.stats as stats
from scipy.stats import ttest_ind
from itertools import combinations

def fetch_and_process_data(url):
    response = requests.get(url)
    if response.status_code != 200:
        raise Exception(f"Failed to fetch data from {url}. HTTP Status Code: {response.status_code}")

    raw_data = [json.loads(line) for line in response.text.splitlines() if line.strip()]
    clean_tracker = pd.json_normalize(raw_data)[['timestamp', 'uuid', 'event', 'data.group', 'data.url', 'data.sessionCount', 'data.referrer']]
    clean_tracker.columns = ['timestamp', 'uuid', 'event', 'group', 'url', 'sessionCount', 'referrer']
    clean_tracker['timestamp'] = pd.to_datetime(clean_tracker['timestamp'], errors='coerce', utc=True)
    return clean_tracker

def process_clean_tracker(clean_tracker):
    clean_tracker['test_group'] = clean_tracker['event'].str.extract(r'(test_group_v\d+)').ffill()
    clean_tracker['test_group'].fillna('test_group_v1', inplace=True)
    clean_tracker['treatment'] = clean_tracker.groupby(['uuid', 'test_group'])['group'].transform(lambda g: g.ffill().bfill())
    return clean_tracker

def create_pivot_and_filter(clean_tracker):
    pivot_data = clean_tracker[~clean_tracker['test_group'].isna()]
    test_group_pivot = pivot_data.pivot_table(index='uuid', columns='test_group', values='treatment', aggfunc='first').reset_index()
    clean_tracker = clean_tracker.merge(test_group_pivot, on='uuid', how='left')
    page_view_data = clean_tracker[clean_tracker['event'] == 'page_view']
    return clean_tracker, page_view_data, test_group_pivot

def calculate_statistics(test_group_pivot, clean_tracker, page_view_data):
    summary_data = []

    for test_group in test_group_pivot.columns[1:]:
        filtered_data = clean_tracker[~clean_tracker[test_group].isna()].copy()
        
        if filtered_data.empty:
            print(f"No data for test group {test_group}. Skipping...")
            continue

        group_summary = filtered_data.groupby('treatment').agg(
            num_uuid=('uuid', 'nunique'),
            num_sessions_mean=('sessionCount', lambda x: x.groupby(filtered_data['uuid']).max().mean())
        ).reset_index()

        if group_summary.empty:
            print(f"Group summary is empty for test group {test_group}. Skipping...")
            continue

        group_summary['test_group'] = test_group
        summary_data.append(group_summary)

    if not summary_data:
        print("No summary data created. Returning empty values.")
        return pd.DataFrame(), pd.DataFrame()

    return pd.concat(summary_data, ignore_index=True)


# Function to reorganize p_values_filtered into a wide table
def reorganize_p_values(p_values_filtered):
    wide_table = pd.pivot_table(
        p_values_filtered,
        values='p_value',
        index='p value comparison',
        columns='group_pair',
        aggfunc='first'
    ).reset_index()
    return wide_table


# Function to calculate p-values for number of UUIDs
def calculate_uuid_p_values(test_group_pivot, clean_tracker):
    p_values = []

    for test_group in test_group_pivot.columns[1:]:
        filtered_data = clean_tracker[~clean_tracker[test_group].isna()].copy()
        
        if filtered_data.empty:
            print(f"No data for test group {test_group}. Skipping...")
            continue

        group_summary = filtered_data.groupby('treatment').agg(
            num_uuid=('uuid', 'nunique')
        ).reset_index()

        # Debug: Check group summary
        print(f"Group summary for {test_group}:\n{group_summary}\n")

        treatments = group_summary['treatment'].unique()
        if len(treatments) > 1:
            for g1, g2 in combinations(treatments, 2):
                g1_data = filtered_data[filtered_data['treatment'] == g1]['uuid']
                g2_data = filtered_data[filtered_data['treatment'] == g2]['uuid']

                # Ensure there is data for both groups
                if g1_data.empty or g2_data.empty:
                    print(f"Skipping {test_group} comparison ({g1} vs {g2}) due to empty data.")
                    continue

                # Perform t-test on the raw UUID data
                _, p_val = ttest_ind(g1_data.value_counts(), g2_data.value_counts(), nan_policy='omit')
                p_values.append({
                    'test_group': test_group,
                    'group_pair': f"{g1}-{g2}",
                    'p value comparison': 'num_uuid',
                    'p_value': p_val
                })

    return pd.DataFrame(p_values)


# %%
url = 'https://checkmyads.org/wp-content/themes/checkmyads/tracker-data.txt'
clean_tracker = fetch_and_process_data(url)
clean_tracker = process_clean_tracker(clean_tracker)
clean_tracker, page_view_data, test_group_pivot = create_pivot_and_filter(clean_tracker)

# Compute statistics
summary_stats = calculate_statistics(test_group_pivot, clean_tracker, page_view_data)
# summary_stats = summary_stats[summary_stats['num_uuid'] >= 10]  # Exclude small groups

# Compute p-values for UUID distributions
p_values_df = calculate_uuid_p_values(test_group_pivot, clean_tracker)
p_values_df.dropna(subset=['p_value'], inplace=True)

# Debug output
if not p_values_df.empty:
    print("P-values for num_uuid:")
    print(p_values_df)
else:
    print("No valid p-values for num_uuid.")

# %%
# Streamlit application setup
st.set_page_config(page_title="Balance Check", page_icon="📊", layout="wide")
st.title("📊 Real-time Test Group Balance Check")

# Dropdown for selecting test group
available_test_groups = summary_stats['test_group'].unique()
st.subheader("Select Test Group")
selected_test_group = st.selectbox("Test Group:", options=available_test_groups)

# st.subheader(f"Statistics for {selected_test_group}")
# Filter data based on selection
filtered_data = summary_stats[summary_stats['test_group'] == selected_test_group]
filtered_display = filtered_data.reset_index(drop=True)
p_values_filtered = p_values_df[p_values_df['test_group'] == selected_test_group]

# Reorganize p_values_filtered for display
if not p_values_filtered.empty:
    p_values_filtered_display = reorganize_p_values(p_values_filtered)
    st.write(p_values_filtered_display)

col1, col2 = st.columns(2)
with col1:
    if all(col in filtered_display.columns for col in ['num_uuid', 'num_sessions_mean']):
        st.write(filtered_display[['num_uuid', 'num_sessions_mean']])
    else:
        st.write("Required metrics are not available in the filtered data.")
with col2:
    st.write("""
    - **num_uuid**: Count of unique visitors  
    - **num_sessions_mean**: Mean session count  
    - **homepage_mean**: Fraction of homepage views  
    """)

if not filtered_data.empty:
    bar_chart = (
        alt.Chart(filtered_data)
        .mark_bar()
        .encode(
            x=alt.X('treatment:N', title='Treatment Group'),  # Ensure this matches your column name
            y=alt.Y('num_uuid:Q', title='Number of UUIDs'),
            tooltip=[
                alt.Tooltip('treatment:N', title='Treatment Group'),
                alt.Tooltip('num_uuid:Q', title='Unique Visitors'),
                alt.Tooltip('num_sessions_mean:Q', title='Average Sessions'),
            ],
            color=alt.Color('treatment:N', title="Group")
        )
        .properties(title=f"UUID Distribution by Group ({selected_test_group})", height=400)
    )
    st.altair_chart(bar_chart, use_container_width=True)
else:
    st.write("No data available for visualization.")
    st.write("Debug Info: The filtered dataset is empty. Check if the selected test group has data.")
