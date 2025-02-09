#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Updated Streamlit App for Real-time Balance Check
Integrated with Newsletter Signup Analysis
"""

import streamlit as st
from utils import *

# STREAMLIT INTERFACE
st.set_page_config(page_title="CMA Balance Check", page_icon="📊", layout="wide")
st.title("📊 Real-time Balance Check")

# Fetch and process data
url = 'https://checkmyads.org/wp-content/themes/checkmyads/tracker-data.txt'
clean_tracker = fetch_and_process_data(url)
clean_tracker = process_clean_tracker(clean_tracker)
clean_tracker = clean_tracker[~clean_tracker['random_group'].isna()]

# Dropdown for selecting test group
available_standard_groups = clean_tracker['standard_group'].unique()
available_standard_groups = available_standard_groups[::-1]

st.subheader("Please select a randomization version we have tested 🔽")
selected_standard_group = st.selectbox("Test Group:", options=available_standard_groups)
selected_clean_tracker = clean_tracker[clean_tracker['standard_group'] == selected_standard_group]
selected_uuid_tracker = process_event_data(selected_clean_tracker)

# Draw visualizations
draw_streamlit_bar(selected_uuid_tracker)
draw_popup_bar_charts(selected_clean_tracker)

# NEWSLETTER SIGNUP ANALYSIS
with st.expander("📧 Newsletter Signup Analysis", expanded=True):
    st.header("Newsletter Signup Analysis")
    newsletter_stats, t_test_results, newsletter_chart = analyze_newsletter_signups(selected_uuid_tracker)

    # Display newsletter results
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Newsletter Signup Statistics")
        st.dataframe(newsletter_stats.style.highlight_max(axis=0))

    with col2:
        st.subheader("Statistical Comparison")
        st.dataframe(t_test_results.style.highlight_min(subset=['p-value']))

    st.altair_chart(newsletter_chart, use_container_width=True)

    # Key Insights
    st.subheader("Key Newsletter Insights")
    total_users = newsletter_stats['Total Users'].sum()
    total_signups = newsletter_stats['Total Signups'].sum()
    conversion_rate = (total_signups / total_users) * 100

    st.write(f"- Total users analyzed: {total_users:,}")
    st.write(f"- Total newsletter signups: {total_signups:,}")
    st.write(f"- Overall conversion rate: {conversion_rate:.2f}%")

# Original balance check tables
group_stats, pairwise_results = gen_output_tables(
    selected_uuid_tracker, 
    datetime_cols=['first_session_start_time', 'average_session_start_time', 'last_session_start_time']
)

# DEMOGRAPHIC ANALYSIS
with st.expander("📊 User Demographics", expanded=True):
    demographic_stats = analyze_demographics(selected_clean_tracker)
    
    st.subheader("User Demographics Analysis")
    
    # Create tabs for different demographic dimensions
    tab_names = ["Platform", "Language", "Vendor", "Timezone"]
    tabs = st.tabs(tab_names)
    
    for tab, (dim_name, stats) in zip(tabs, demographic_stats.items()):
        with tab:
            # Center the table with columns
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                st.write(f"Distribution by {tab_names[list(demographic_stats.keys()).index(dim_name)]}")
                st.dataframe(stats, use_container_width=True)
            
            # Create visualization with proper title string
            chart = alt.Chart(stats).mark_bar().encode(
                x=alt.X(f'{dim_name}:N', title=tab_names[list(demographic_stats.keys()).index(dim_name)]),
                y=alt.Y('percentage:Q', title='Percentage (%)'),
                color=alt.Color('random_group:N', title='Group'),
                tooltip=[
                    alt.Tooltip(dim_name, title=tab_names[list(demographic_stats.keys()).index(dim_name)]),
                    alt.Tooltip('random_group', title='Group'),
                    alt.Tooltip('count', title='Count'),
                    alt.Tooltip('percentage', title='Percentage (%)', format='.1f')
                ]
            ).properties(
                title=f'Distribution by {tab_names[list(demographic_stats.keys()).index(dim_name)]}',
                width=600,
                height=400
            )
            
            # Center the chart with columns
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                st.altair_chart(chart, use_container_width=True)

# SCREEN DIMENSIONS ANALYSIS
with st.expander("📱 Screen Dimensions", expanded=True):
    screen_stats, window_stats = analyze_screen_dimensions(selected_clean_tracker)
    
    st.subheader("Screen and Window Size Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("Screen Size Distribution")
        st.dataframe(screen_stats, use_container_width=True)
        
        screen_chart = alt.Chart(screen_stats).mark_bar().encode(
            x=alt.X('screen_size:N', title='Screen Size'),
            y=alt.Y('count:Q', title='Count'),
            color=alt.Color('random_group:N', title='Group'),
            tooltip=[
                alt.Tooltip('screen_size', title='Screen Size'),
                alt.Tooltip('random_group', title='Group'),
                alt.Tooltip('count', title='Count')
            ]
        ).properties(
            title='Screen Size Distribution',
            width=300,
            height=400
        )
        
        st.altair_chart(screen_chart, use_container_width=True)
    
    with col2:
        st.write("Window Size Distribution")
        st.dataframe(window_stats, use_container_width=True)
        
        window_chart = alt.Chart(window_stats).mark_bar().encode(
            x=alt.X('window_size:N', title='Window Size'),
            y=alt.Y('count:Q', title='Count'),
            color=alt.Color('random_group:N', title='Group'),
            tooltip=[
                alt.Tooltip('window_size', title='Window Size'),
                alt.Tooltip('random_group', title='Group'),
                alt.Tooltip('count', title='Count')
            ]
        ).properties(
            title='Window Size Distribution',
            width=300,
            height=400
        )
        
        st.altair_chart(window_chart, use_container_width=True)

# REFERRAL ANALYSIS
with st.expander("🔗 Referral Source Analysis", expanded=True):
    referral_stats = analyze_referrals(selected_clean_tracker)
    
    if not referral_stats.empty:
        st.subheader("Referral Traffic Analysis")
        
        # Overview metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            total_referrals = referral_stats['total_visits'].sum()
            st.metric("Total Referral Visits", f"{total_referrals:,}")
        with col2:
            total_conversions = referral_stats['total_signups'].sum()
            st.metric("Total Conversions", f"{total_conversions:,}")
        with col3:
            overall_conv_rate = (total_conversions / total_referrals * 100) if total_referrals > 0 else 0
            st.metric("Overall Conversion Rate", f"{overall_conv_rate:.2f}%")
        
        # Detailed statistics
        st.subheader("Referral Source Performance")
        
        # Order the referral_stats by random_group and then by the first column
        referral_stats = referral_stats.sort_values(by=['random_group', 'total_visits'], ascending=[True, False])
        
        # Center the table
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            # Safely apply styling only to columns that exist
            style_columns = ['conversion_rate', 'traffic_share']
            existing_style_columns = [col for col in style_columns if col in referral_stats.columns]
            
            if existing_style_columns:
                styled_df = referral_stats.style.background_gradient(subset=existing_style_columns)
            else:
                styled_df = referral_stats.style
            
            st.dataframe(styled_df, use_container_width=True)
        
        # Visualizations
        st.subheader("Referral Source Visualizations")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Traffic Distribution Chart
            if 'traffic_share' in referral_stats.columns:
                traffic_chart = alt.Chart(referral_stats).mark_bar().encode(
                    x=alt.X('referrer_category:N', title='Referrer', sort='-y'),
                    y=alt.Y('traffic_share:Q', title='Traffic Share (%)'),
                    color=alt.Color('random_group:N', title='Group'),
                    tooltip=[
                        alt.Tooltip('referrer_category', title='Referrer'),
                        alt.Tooltip('random_group', title='Group'),
                        alt.Tooltip('total_visits', title='Total Visits'),
                        alt.Tooltip('traffic_share', title='Traffic Share (%)', format='.1f')
                    ]
                ).properties(
                    title='Traffic Distribution by Referrer',
                    width=400,
                    height=300
                )
                st.altair_chart(traffic_chart, use_container_width=True)
            else:
                st.write("Traffic share data not available")
        
        with col2:
            # Conversion Rate Chart
            if 'conversion_rate' in referral_stats.columns:
                conversion_chart = alt.Chart(referral_stats).mark_bar().encode(
                    x=alt.X('referrer_category:N', title='Referrer', sort='-y'),
                    y=alt.Y('conversion_rate:Q', title='Conversion Rate (%)'),
                    color=alt.Color('random_group:N', title='Group'),
                    tooltip=[
                        alt.Tooltip('referrer_category', title='Referrer'),
                        alt.Tooltip('random_group', title='Group'),
                        alt.Tooltip('total_signups', title='Total Signups'),
                        alt.Tooltip('conversion_rate', title='Conversion Rate (%)', format='.1f')
                    ]
                ).properties(
                    title='Conversion Rates by Referrer',
                    width=400,
                    height=300
                )
                st.altair_chart(conversion_chart, use_container_width=True)
            else:
                st.write("Conversion rate data not available")
        
        # Key Insights
        st.subheader("Key Insights")
        
        # Top referrers by traffic
        if 'traffic_share' in referral_stats.columns:
            top_traffic = referral_stats.nlargest(3, 'traffic_share')
            st.write("Top Traffic Sources:")
            for _, row in top_traffic.iterrows():
                st.write(f"- {row['referrer_category']}: {row['traffic_share']:.1f}% of total traffic")
        
        # Top referrers by conversion
        if 'conversion_rate' in referral_stats.columns:
            top_conversion = referral_stats[referral_stats['total_visits'] >= 5].nlargest(3, 'conversion_rate')
            if not top_conversion.empty:
                st.write("\nBest Converting Sources (min. 5 visits):")
                for _, row in top_conversion.iterrows():
                    st.write(f"- {row['referrer_category']}: {row['conversion_rate']:.1f}% conversion rate")
            else:
                st.write("\nNo sources with sufficient visits for conversion analysis")
    else:
        st.write("No referral data available for analysis.") 