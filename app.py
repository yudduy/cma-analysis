#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Updated Streamlit App for Real-time Balance Check
Integrated with Newsletter Signup Analysis
"""

import streamlit as st
from utils import *
import requests

# STREAMLIT INTERFACE
st.set_page_config(page_title="CMA Experiment", page_icon="📊", layout="wide")
st.title("📊 CMA Experiment Monitor")

st.write(f"Group 1 (Worthiness): Backed by the Ford Foundation: We are creating a healthier online ad ecosystem.")
st.write(f"Group 2 (Numbers): A growing community of 50,000+ members: We are creating a healthier online ad ecosystem.")
st.write(f"Group 3 (Watchdog): The digital advertising watchdog: We are creating a healthier online ad ecosystem.")
st.write(f"Group 4 (Pure Control): We are creating a healthier online ad ecosystem.")

# Fetch and process data
url = 'https://checkmyads.org/wp-content/themes/checkmyads/tracker-data.txt'
clean_tracker = fetch_and_process_data(url)
clean_tracker = process_clean_tracker(clean_tracker)
clean_tracker = clean_tracker[~clean_tracker['random_group'].isna()]

# Process error log data
try:
    # Read local error log file
    with open('error.log', 'r') as f:
        error_log_content = f.read()
    error_log_df = parse_error_log(error_log_content)
    st.success(f"Successfully processed error log with {len(error_log_df)} entries")
except FileNotFoundError:
    st.warning("Could not find error.log file. Proceeding with tracker data only.")
    error_log_df = pd.DataFrame(columns=['timestamp', 'ip_address', 'email', 'event_type'])
except Exception as e:
    st.error(f"Error processing error log: {str(e)}")
    error_log_df = pd.DataFrame(columns=['timestamp', 'ip_address', 'email', 'event_type'])

# Merge IP data
ip_uuid_map = merge_ip_data(clean_tracker, error_log_df)
if not ip_uuid_map.empty:
    st.success(f"Found {len(ip_uuid_map)} unique IP-UUID mappings")

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

# TIME SERIES ANALYSIS
with st.expander("📈 Time Series Analysis", expanded=True):
    st.header("Time Series Analysis")
    
    try:
        # Validate data
        if selected_clean_tracker.empty:
            st.warning("No data available for time series analysis.")
        else:
            # Calculate time series metrics
            daily_counts, dow_patterns = analyze_time_series(selected_clean_tracker)
            
            if daily_counts.empty or dow_patterns.empty:
                st.warning("Unable to generate time series analysis. No signup data available.")
            else:
                # Display rolling average chart
                st.subheader("Signup Trends Over Time")
                rolling_avg_chart, dow_chart, anomaly_chart = create_time_series_charts(daily_counts)
                
                if isinstance(rolling_avg_chart, alt.Chart) and not rolling_avg_chart.data.empty:
                    st.altair_chart(rolling_avg_chart, use_container_width=True)
                    
                    # Day of week patterns - Stacked vertically
                    st.subheader("Day of Week Patterns")
                    if isinstance(dow_chart, alt.Chart) and not dow_chart.data.empty:
                        st.altair_chart(dow_chart, use_container_width=True)
                        
                        # Display day of week statistics in columns
                        st.write("Average Signups by Day:")
                        if not dow_patterns.empty:
                            st.dataframe(
                                dow_patterns.style.highlight_max(subset=['mean'])
                                .format({'mean': '{:.2f}', 'std': '{:.2f}'})
                            )
                        else:
                            st.warning("No day-of-week patterns available.")
                    else:
                        st.warning("Unable to generate day-of-week visualization.")
                
                # Add summary statistics
                st.subheader("Summary Statistics")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    total_signups = daily_counts['signups'].sum()
                    st.metric("Total Signups", f"{total_signups:,}")
                
                with col2:
                    avg_daily_signups = daily_counts.groupby('date')['signups'].sum().mean()
                    st.metric("Average Daily Signups", f"{avg_daily_signups:.1f}")
                
                with col3:
                    total_days = len(daily_counts['date'].unique())
                    st.metric("Days Analyzed", f"{total_days:,}")
    
    except Exception as e:
        st.error(f"Error in time series analysis: {str(e)}")
        st.write("Please try refreshing the page or contact support if the error persists.")
        
# VISITOR LOCATION ANALYSIS
with st.expander("🌍 Visitor Location Analysis", expanded=True):
    st.header("Visitor Location Analysis")
    
    # Process error log data
    try:
        # Read local error log file
        with open('error.log', 'r') as f:
            error_log_content = f.read()
        error_log_df = parse_error_log(error_log_content)
        st.success(f"Successfully processed error log with {len(error_log_df)} entries")
    except FileNotFoundError:
        st.warning("Could not find error.log file. Proceeding with tracker data only.")
        error_log_df = pd.DataFrame(columns=['timestamp', 'ip_address', 'email', 'event_type'])
    except Exception as e:
        st.error(f"Error processing error log: {str(e)}")
        error_log_df = pd.DataFrame(columns=['timestamp', 'ip_address', 'email', 'event_type'])

    # Merge IP data
    ip_uuid_map = merge_ip_data(clean_tracker, error_log_df)
    if not ip_uuid_map.empty:
        st.success(f"Found {len(ip_uuid_map)} unique IP-UUID mappings")
    
    if ip_uuid_map.empty:
        st.warning("No IP-UUID mappings available for location analysis.")
    else:
        # Process location data
        location_analysis, country_stats = analyze_visitor_locations(ip_uuid_map, selected_uuid_tracker)
        
        if location_analysis.empty or country_stats.empty:
            st.warning("No location data available for analysis.")
        else:
            # Display location statistics
            st.subheader("Visitor Distribution by Country")
            
            # Create a pivot table for better visualization
            pivot_stats = country_stats.pivot(
                index='country',
                columns='random_group',
                values=['count', 'percentage']
            ).fillna(0)
            
            # Rename columns for clarity
            pivot_stats.columns = [f'Group {g} Count' if 'count' in c else f'Group {g} %' 
                                 for c, g in pivot_stats.columns]
            
            st.dataframe(pivot_stats.style.highlight_max(axis=1))
            
            # Create and display location visualizations
            map_chart, country_chart = create_location_charts(location_analysis)
            
            st.subheader("Visitor Locations on World Map")
            st.altair_chart(map_chart, use_container_width=True)
            
            st.subheader("Visitor Distribution by Country")
            st.altair_chart(country_chart, use_container_width=True)
            
            # Key Location Insights
            st.subheader("Key Location Insights")
            total_countries = len(country_stats['country'].unique())
            
            # Get stats for each group
            group_insights = []
            for group in range(1, 5):
                group_data = country_stats[country_stats['random_group'] == group]
                total_visitors = group_data['count'].sum()
                countries_reached = len(group_data[group_data['count'] > 0])
                group_insights.append({
                    'Group': f'Group {group}',
                    'Total Visitors': total_visitors,
                    'Countries Reached': countries_reached
                })
            
            group_insights_df = pd.DataFrame(group_insights)
            st.write("Visitor Distribution by Group:")
            st.dataframe(group_insights_df)
            
            # Top countries overall
            st.write("\nTop Countries by Total Visitors:")
            top_countries = country_stats.groupby('country')['count'].sum().sort_values(ascending=False).head(5)
            for country, count in top_countries.items():
                st.write(f"  • {country}: {int(count)} visitors")
            
            # Additional Statistics
            st.subheader("Additional Statistics")
            total_ips = len(ip_uuid_map)
            mapped_users = len(ip_uuid_map['uuid'].unique())
            st.write(f"- Total unique IP addresses: {total_ips}")
            st.write(f"- Total users with IP mapping: {mapped_users}")
            
            # Display any IP resolution errors at the bottom
            if 'error_ips' in location_analysis.columns and not location_analysis['error_ips'].isna().all():
                st.subheader("IP Resolution Issues")
                for ip, error in location_analysis['error_ips'].dropna():
                    st.write(f"Could not get location for IP {ip}: {error}")

# Original balance check tables
group_stats, pairwise_results = gen_output_tables(
    selected_uuid_tracker, 
    datetime_cols=['first_session_start_time', 'average_session_start_time', 'last_session_start_time']
)

# NEW VS RETURNING USER ANALYSIS
with st.expander("👥 User Type Analysis", expanded=True):
    st.header("New vs Returning Users Analysis")
    
    # Process user type data
    user_type_data = analyze_user_types(selected_clean_tracker)
    
    # Display user type statistics
    st.subheader("User Type Distribution")
    
    # Create columns for metrics
    col1, col2 = st.columns(2)
    
    with col1:
        total_users = len(user_type_data)
        new_users = user_type_data['is_new_user'].sum()
        returning_users = total_users - new_users
        
        st.metric("Total Users", f"{total_users:,}")
        st.metric("New Users", f"{new_users:,} ({new_users/total_users*100:.1f}%)")
        st.metric("Returning Users", f"{returning_users:,} ({returning_users/total_users*100:.1f}%)")
    
    with col2:
        # Calculate signup rates by user type
        new_signup_rate = user_type_data[user_type_data['is_new_user']]['has_signup'].mean() * 100
        returning_signup_rate = user_type_data[~user_type_data['is_new_user']]['has_signup'].mean() * 100
        
        st.metric("New User Signup Rate", f"{new_signup_rate:.1f}%")
        st.metric("Returning User Signup Rate", f"{returning_signup_rate:.1f}%")
    
    # Display detailed statistics by group
    st.subheader("Signup Rates by User Type and Group")
    user_type_stats = calculate_user_type_stats(user_type_data)
    st.dataframe(user_type_stats.style.highlight_max(axis=0))
    
    # Create and display visualization
    user_type_chart = create_user_type_charts(user_type_data)
    st.altair_chart(user_type_chart, use_container_width=True)

# A/B TESTING ANALYSIS
with st.expander("🔬 A/B Testing Analysis", expanded=True):
    st.header("A/B Testing Analysis")
    
    # Calculate A/B test statistics using selected_uuid_tracker instead of uuid_tracker
    ab_test_results = calculate_ab_test_stats(selected_uuid_tracker)
    
    # Display test results
    st.subheader("Test Results Summary")
    
    # Create metrics for each test group
    col1, col2, col3 = st.columns(3)
    
    for idx, row in ab_test_results.iterrows():
        with col1 if idx == 0 else col2 if idx == 1 else col3:
            st.metric(
                f"Group {row['test_group']} vs Control",
                f"{row['relative_lift']:.1f}% Lift",
                f"p={row['p_value']:.4f}"
            )
    
    # Display detailed statistics
    st.subheader("Detailed Test Statistics")
    
    # Format the results for display
    display_cols = [
        'test_group', 'control_conv_rate', 'test_conv_rate', 'relative_lift',
        'p_value', 'power', 'control_sample_size', 'test_sample_size'
    ]
    
    formatted_results = ab_test_results[display_cols].copy()
    formatted_results = formatted_results.round(4)
    formatted_results.columns = [
        'Test Group', 'Control Conv. Rate', 'Test Conv. Rate', 'Relative Lift (%)',
        'p-value', 'Statistical Power', 'Control Sample Size', 'Test Sample Size'
    ]
    
    st.dataframe(
        formatted_results.style.highlight_min(subset=['p-value'])
                          .highlight_max(subset=['Relative Lift (%)'])
    )
    
    # Display visualization
    st.subheader("Conversion Rates Comparison")
    conv_rate_chart = create_ab_test_charts(ab_test_results)
    st.altair_chart(conv_rate_chart, use_container_width=True)
    
    # Power Analysis Insights
    st.subheader("Statistical Power Analysis")
    for idx, row in ab_test_results.iterrows():
        power_color = "green" if row['power'] >= 0.8 else "orange" if row['power'] >= 0.5 else "red"
        st.write(f"Group {row['test_group']} vs Control:")
        st.write(f"- Statistical Power: ::{power_color}[{row['power']:.2f}]")
        if row['power'] < 0.8:
            required_n = TTestPower().solve_power(
                effect_size=row['relative_lift']/100,
                power=0.8,
                alpha=0.05
            )
            additional_n = max(0, int(required_n - min(row['control_sample_size'], row['test_sample_size'])))
            if additional_n > 0:
                st.write(f"- Need {additional_n:,} more samples per group for 80% power")