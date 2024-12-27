import streamlit as st
import pandas as pd
import altair as alt
import requests
import json
import scipy.stats as stats

# Fetch data from URL
url = 'https://checkmyads.org/wp-content/themes/checkmyads/tracker-data.txt'
response = requests.get(url)

if response.status_code == 200:
    raw_data = response.text.splitlines()
else:
    raise Exception(f"Failed to fetch data from {url}. HTTP Status Code: {response.status_code}")

# Parse data
processed_data = [json.loads(line) for line in raw_data if line.strip()]
raw_tracker = pd.json_normalize(processed_data)

# Extract necessary columns and normalize
clean_tracker = raw_tracker[['timestamp', 'uuid', 'event', 'data.group', 'data.url', 
                             'data.sessionCount', 'data.referrer']]

clean_tracker.columns = ['timestamp', 'uuid', 'event', 'group', 'url', 'sessionCount', 'referrer']
clean_tracker['timestamp'] = pd.to_datetime(clean_tracker['timestamp'], errors='coerce', utc=True)

# Extract `test_group_v*` dynamically from the event field
test_group_cols = clean_tracker['event'].str.extract(r'(test_group_v\d+)')
clean_tracker['test_group'] = test_group_cols
clean_tracker['group'] = clean_tracker['group'].fillna(0)

# Pivot table to dynamically assign group numbers for each `test_group_v*`
test_group_data = clean_tracker[~clean_tracker['test_group'].isna()]
test_group_pivot = test_group_data.pivot_table(
    index='uuid', 
    columns='test_group', 
    values='group', 
    aggfunc='first'
).reset_index()

# Merge the pivoted test group information back into the main tracker data
clean_tracker = clean_tracker.merge(test_group_pivot, on='uuid', how='left')

# Group and summarize data for all test_group_v* dynamically
summary_data = []
p_values = []

for test_group in test_group_pivot.columns[1:]:  # Skip 'uuid'
    filtered_data = clean_tracker[~clean_tracker[test_group].isna()]
    group_summary = filtered_data.groupby(test_group).agg(
        num_uuid=('uuid', 'nunique'),
        num_sessions_mean=('sessionCount', 'mean'),
        homepage_mean=('url', lambda x: x.eq('https://checkmyads.org/').mean())
    ).reset_index()

    group_summary['test_group'] = test_group
    summary_data.append(group_summary)

    # Compute p-values between groups for key metrics
    groups = filtered_data[test_group].unique()
    if len(groups) > 1:
        for i, g1 in enumerate(groups):
            for g2 in groups[i+1:]:
                g1_data = filtered_data[filtered_data[test_group] == g1]['sessionCount'].dropna()
                g2_data = filtered_data[filtered_data[test_group] == g2]['sessionCount'].dropna()
                t_stat, p_val = stats.ttest_ind(g1_data, g2_data, equal_var=False)
                p_values.append({'test_group': test_group, 'group_pair': f"{g1}-{g2}", 'p_value': p_val})

summary_stats = pd.concat(summary_data, ignore_index=True)
p_values_df = pd.DataFrame(p_values)

# Configure Streamlit page
st.set_page_config(page_title="Test Group Analysis", page_icon="📊", layout="wide")
st.title("📊 Real-time Balance Check of CMA Randomization")

# Dropdown menu for selecting test group
available_test_groups = summary_stats['test_group'].unique()
selected_test_group = st.selectbox(
    "Select Test Group to Analyze",
    options=available_test_groups
)

# Filter data for the selected test group
filtered_data = summary_stats[summary_stats['test_group'] == selected_test_group]
p_values_filtered = p_values_df[p_values_df['test_group'] == selected_test_group]
filtered_data_display = filtered_data[[selected_test_group, 'num_uuid', 'num_sessions_mean', 'homepage_mean']]

# Display summary statistics
st.subheader(f"Summary Statistics for {selected_test_group}")

grid = st.columns(2)
with grid[0]:
    st.write(filtered_data[[selected_test_group, 'num_uuid', 'num_sessions_mean', 'homepage_mean']])

with grid[1]:
    st.write(p_values_filtered)
    st.write(
        """
        - **num_uuid**: Number of visitors in each group  
        - **num_sessions_mean**: Average number of sessions initiated by users  
        - **homepage_mean**: Average percentage of page views as visiting homepage  
        - **p_value**: Statistical significance between groups
        """
    )

# Optional: Create bar chart
bar_chart = (
    alt.Chart(filtered_data)
    .mark_bar()
    .encode(
        x=alt.X(f'{selected_test_group}:N', title='Group'),
        y=alt.Y('num_uuid:Q', title='Number of UUIDs'),
        tooltip=[
            alt.Tooltip(f'{selected_test_group}:N', title='Group'),
            alt.Tooltip('num_uuid:Q', title='Number of UUIDs'),
            alt.Tooltip('num_sessions_mean:Q', title='Avg. Sessions'),
            alt.Tooltip('homepage_mean:Q', title='Avg. Homepage View %')
        ],
        color=alt.Color(f'{selected_test_group}:N', title="Group")
    )
    .properties(title=f"Number of UUIDs by Group in {selected_test_group}", height=400)
)

st.altair_chart(bar_chart, use_container_width=True)


