import pandas as pd
import numpy as np
import altair as alt
import requests
import json
import scipy.stats 
import streamlit as st
from itertools import combinations
import re
from datetime import datetime, timedelta
from geoip2.database import Reader
from geoip2.errors import AddressNotFoundError
import maxminddb.errors
from scipy.stats import chi2_contingency, fisher_exact
from statsmodels.stats.power import TTestPower
from statsmodels.stats.proportion import proportion_confint

@st.cache_data(ttl=300)
def fetch_and_process_data(url):
    response = requests.get(url)
    if response.status_code != 200:
        raise Exception(f"Failed to fetch data from {url}. HTTP Status Code: {response.status_code}")

    raw_data = [json.loads(line) for line in response.text.splitlines() if line.strip()]
    
    # Extract browser info and flatten the data
    processed_data = []
    for entry in raw_data:
        data = {
            'timestamp': entry.get('timestamp'),
            'uuid': entry.get('uuid'),
            'event': entry.get('event'),
            'group': entry.get('data', {}).get('group'),
            'url': entry.get('data', {}).get('url'),
            'sessionCount': entry.get('data', {}).get('sessionCount'),
            'referrer': entry.get('data', {}).get('referrer'),
            'popupId': entry.get('data', {}).get('popupId')
        }
        
        # Extract browser info if available
        browser_info = entry.get('data', {}).get('browserInfo', {})
        if browser_info:
            data.update({
                'userAgent': browser_info.get('userAgent'),
                'language': browser_info.get('language'),
                'platform': browser_info.get('platform'),
                'screenWidth': browser_info.get('screenWidth'),
                'screenHeight': browser_info.get('screenHeight'),
                'windowWidth': browser_info.get('windowWidth'),
                'windowHeight': browser_info.get('windowHeight'),
                'timezone': browser_info.get('timezone'),
                'cookiesEnabled': browser_info.get('cookiesEnabled'),
                'vendor': browser_info.get('vendor')
            })
        
        processed_data.append(data)
    
    clean_tracker = pd.DataFrame(processed_data)
    clean_tracker['timestamp'] = pd.to_datetime(clean_tracker['timestamp'], errors='coerce', utc=True)
    return clean_tracker

@st.cache_data
def process_clean_tracker(clean_tracker):
    clean_tracker['standard_group'] = clean_tracker['event'].str.extract(r'(group_v\d+)').ffill()
    clean_tracker['standard_group'] = clean_tracker['standard_group'].fillna('group_v1')
    clean_tracker['random_group'] = clean_tracker.groupby(['uuid', 'standard_group'])['group'].transform(lambda g: g.ffill().bfill())
    return clean_tracker

def process_event_data(clean_tracker):
    def count_event(event_series, event_name):
        return (event_series == event_name).sum()

    def calculate_homepage_pct(url_series, event_series):
        page_view_count = (event_series == 'page_view').sum()
        if page_view_count > 0:
            return (url_series == 'https://checkmyads.org/').sum() / page_view_count
        return np.nan

    def check_url_presence(url_series, keyword):
        return int(any(isinstance(ref, str) and keyword in ref.lower() for ref in url_series))

    uuid_tracker = clean_tracker.groupby('uuid').agg(
        random_group=('random_group', 'first'),
        num_sessions=('event', lambda x: count_event(x, 'session_start')),
        num_page_views=('event', lambda x: count_event(x, 'page_view')),
        num_popup_views=('event', lambda x: count_event(x, 'popup_view')),
        num_referral=('event', lambda x: count_event(x, 'referral')),
        num_newsletter_signup=('event', lambda x: count_event(x, 'newsletter_signup')),
        num_donation=('event', lambda x: count_event(x, 'donation')),
        first_session_start_time=('timestamp', lambda x: x.loc[clean_tracker['event'] == 'session_start'].min()),
        average_session_start_time=('timestamp', lambda x: x.loc[clean_tracker['event'] == 'session_start'].mean()),
        last_session_start_time=('timestamp', lambda x: x.loc[clean_tracker['event'] == 'session_start'].max()),
        homepage_pct=('url', lambda x: calculate_homepage_pct(x, clean_tracker['event'])),
        view_about=('url', lambda x: check_url_presence(x, 'checkmyads.org/about')),
        view_news=('url', lambda x: check_url_presence(x, 'checkmyads.org/news')),
        view_donate=('url', lambda x: check_url_presence(x, 'checkmyads.org/donate')),
        view_google_trial=('url', lambda x: check_url_presence(x, 'checkmyads.org/google')),
        view_shop=('url', lambda x: check_url_presence(x, 'checkmyads.org/shop')),
        referral_google=('referrer', lambda x: check_url_presence(x, 'google')),
        referral_pcgamer=('referrer', lambda x: check_url_presence(x, 'pcgamer')),
        referral_globalprivacycontrol=('referrer', lambda x: check_url_presence(x, 'globalprivacycontrol')),
        referral_duckduckgo=('referrer', lambda x: check_url_presence(x, 'duckduckgo'))
    ).reset_index()

    return uuid_tracker

def analyze_newsletter_signups(uuid_tracker):
    # Calculate newsletter summary statistics
    newsletter_stats = uuid_tracker.groupby('random_group').agg({
        'uuid': 'count',
        'num_newsletter_signup': ['mean', 'std', 'sum']
    }).round(3)
    
    newsletter_stats.columns = ['Total Users', 'Avg Signups', 'Std Dev', 'Total Signups']
    newsletter_stats = newsletter_stats.reset_index()
    
    # Create visualization
    chart = alt.Chart(newsletter_stats).mark_bar().encode(
        x=alt.X('random_group:N', title='Treatment Group'),
        y=alt.Y('Avg Signups:Q', title='Average Newsletter Signups'),
        tooltip=['Total Users', 'Avg Signups', 'Total Signups']
    ).properties(
        title='Newsletter Signup Rates by Treatment Group',
        width=600,
        height=400
    )
    
    # Perform pairwise t-tests
    groups = sorted(uuid_tracker['random_group'].unique())
    t_test_results = []
    
    for g1, g2 in combinations(groups, 2):
        group1_data = uuid_tracker[uuid_tracker['random_group'] == g1]['num_newsletter_signup']
        group2_data = uuid_tracker[uuid_tracker['random_group'] == g2]['num_newsletter_signup']
        
        t_stat, p_val = scipy.stats.ttest_ind(group1_data, group2_data, equal_var=False)  # Change this line
        t_test_results.append({
            'Comparison': f'Group {g1} vs Group {g2}',
            't-statistic': round(t_stat, 3),
            'p-value': round(p_val, 4)
        })
    
    return newsletter_stats, pd.DataFrame(t_test_results), chart
def draw_streamlit_bar(uuid_tracker):
    # Basic visualization of user metrics
    metrics_chart = alt.Chart(uuid_tracker).mark_bar().encode(
        x='random_group:N',
        y='count():Q',
        tooltip=['random_group', 'count()']
    ).properties(title='Users per Group')
    st.altair_chart(metrics_chart, use_container_width=True)

def draw_popup_bar_charts(clean_tracker):
    popup_data = clean_tracker[clean_tracker['event'] == 'popup_view'].groupby('random_group').size().reset_index(name='count')
    popup_chart = alt.Chart(popup_data).mark_bar().encode(
        x='random_group:N',
        y='count:Q',
        tooltip=['random_group', 'count']
    ).properties(title='Popup Views per Group')
    st.altair_chart(popup_chart, use_container_width=True)

def gen_output_tables(uuid_tracker, datetime_cols):
    # Generate basic statistics by group
    group_stats = uuid_tracker.groupby('random_group').agg({
        'num_sessions': ['mean', 'count'],
        'num_page_views': ['mean', 'sum'],
        'num_popup_views': ['mean', 'sum']
    }).round(3)
    
    # Generate pairwise comparisons
    groups = sorted(uuid_tracker['random_group'].unique())
    pairwise_results = []
    for g1, g2 in combinations(groups, 2):
        group1 = uuid_tracker[uuid_tracker['random_group'] == g1]
        group2 = uuid_tracker[uuid_tracker['random_group'] == g2]
        pairwise_results.append({
            'comparison': f'{g1} vs {g2}',
            'sessions_diff': group1['num_sessions'].mean() - group2['num_sessions'].mean(),
            'pageviews_diff': group1['num_page_views'].mean() - group2['num_page_views'].mean()
        })
    
    return group_stats, pd.DataFrame(pairwise_results)

def analyze_demographics(clean_tracker):
    """Simplified demographic analysis focusing on key browser info"""
    
    # Get session_start events which contain browser info
    session_data = clean_tracker[clean_tracker['event'] == 'session_start'].copy()
    
    # Get the first instance of demographic data for each UUID
    demo_data = session_data.groupby('uuid').agg({
        'random_group': 'first',
        'userAgent': 'first',
        'language': 'first',
        'platform': 'first',
        'vendor': 'first',
        'timezone': 'first'
    }).reset_index()
    
    # Calculate statistics for each demographic dimension
    dimensions = ['platform', 'language', 'vendor', 'timezone']
    stats = {}
    
    for dim in dimensions:
        if dim in demo_data.columns:
            dim_stats = demo_data.groupby([dim, 'random_group']).size().reset_index(name='count')
            dim_stats['percentage'] = dim_stats.groupby('random_group')['count'].transform(
                lambda x: x / x.sum() * 100
            ).round(2)
            stats[dim] = dim_stats
    
    return stats

def analyze_screen_dimensions(clean_tracker):
    """Separate analysis for screen and window dimensions"""
    
    session_data = clean_tracker[clean_tracker['event'] == 'session_start'].copy()
    
    # Get the first instance for each UUID
    screen_data = session_data.groupby('uuid').agg({
        'random_group': 'first',
        'screenWidth': 'first',
        'screenHeight': 'first',
        'windowWidth': 'first',
        'windowHeight': 'first'
    }).reset_index()
    
    # Categorize screen sizes
    def categorize_size(width, height):
        if pd.isna(width) or pd.isna(height):
            return 'Unknown'
        area = width * height
        if area < 1000000:  # Less than 1MP
            return 'Small'
        elif area < 2000000:  # Less than 2MP
            return 'Medium'
        else:
            return 'Large'
    
    screen_data['screen_size'] = screen_data.apply(
        lambda x: categorize_size(x['screenWidth'], x['screenHeight']), 
        axis=1
    )
    
    screen_data['window_size'] = screen_data.apply(
        lambda x: categorize_size(x['windowWidth'], x['windowHeight']), 
        axis=1
    )
    
    # Calculate statistics
    screen_stats = screen_data.groupby(['screen_size', 'random_group']).size().reset_index(name='count')
    window_stats = screen_data.groupby(['window_size', 'random_group']).size().reset_index(name='count')
    
    return screen_stats, window_stats

def create_demographic_charts(browser_stats, screen_stats, referrer_stats):
    """Create visualization charts for demographic data"""
    # Browser chart
    browser_chart = alt.Chart(browser_stats).mark_bar().encode(
        x=alt.X('browser:N', title='Browser'),
        y=alt.Y('Avg Signups:Q', title='Average Newsletter Signups'),
        color='random_group:N',
        tooltip=['browser', 'random_group', 'Total Users', 'Avg Signups', 'Total Signups']
    ).properties(
        title='Newsletter Signups by Browser',
        width=600,
        height=400
    )
    
    # Referrer chart
    referrer_chart = alt.Chart(referrer_stats).mark_bar().encode(
        x=alt.X('referrer_category:N', title='Referrer'),
        y=alt.Y('Avg Signups:Q', title='Average Newsletter Signups'),
        color='random_group:N',
        tooltip=['referrer_category', 'random_group', 'Total Users', 'Avg Signups', 'Total Signups']
    ).properties(
        title='Newsletter Signups by Referrer',
        width=600,
        height=400
    )
    
    return browser_chart, None, referrer_chart

# Add statistical testing
def demographic_statistical_test(data, category_col):
    groups = data['random_group'].unique()
    results = []
    
    for cat in data[category_col].unique():
        cat_data = data[data[category_col] == cat]
        for g1, g2 in combinations(groups, 2):
            group1 = cat_data[cat_data['random_group'] == g1]['num_newsletter_signup']
            group2 = cat_data[cat_data['random_group'] == g2]['num_newsletter_signup']
            
            if len(group1) > 1 and len(group2) > 1:
                t_stat, p_val = scipy.stats.ttest_ind(group1, group2, equal_var=False)
                results.append({
                    'Category': cat,
                    'Comparison': f'Group {g1} vs Group {g2}',
                    't-statistic': round(t_stat, 3),
                    'p-value': round(p_val, 4)
                })
    
    return pd.DataFrame(results)

def analyze_referrals(clean_tracker):
    """Analyze referral patterns and their impact on newsletter signups"""
    
    # Get referral data
    referral_data = clean_tracker[clean_tracker['event'] == 'referral'].copy()
    
    # Extract domain from referrer URL
    def extract_domain(url):
        if pd.isna(url):
            return 'direct'
        url = str(url).lower()
        domains = {
            'google': 'Google',
            'duckduckgo': 'DuckDuckGo',
            'bing': 'Bing',
            'yahoo': 'Yahoo',
            'facebook': 'Facebook',
            'twitter': 'Twitter',
            'linkedin': 'LinkedIn',
            'reddit': 'Reddit',
            'github': 'GitHub'
        }
        for key, value in domains.items():
            if key in url:
                return value
        return 'Other'
    
    # Process referral data
    referral_stats = pd.DataFrame()
    if not referral_data.empty:
        # Get newsletter signups per UUID
        newsletter_data = clean_tracker[clean_tracker['event'] == 'newsletter_signup'].groupby('uuid').size()
        
        # Prepare referral analysis
        referral_data['referrer_category'] = referral_data['referrer'].apply(extract_domain)
        referral_analysis = referral_data.groupby(['uuid', 'random_group', 'referrer_category']).first().reset_index()
        
        # Add newsletter signup info
        referral_analysis['has_signup'] = referral_analysis['uuid'].isin(newsletter_data.index)
        
        # Calculate statistics
        referral_stats = referral_analysis.groupby(['referrer_category', 'random_group']).agg({
            'uuid': 'count',
            'has_signup': 'sum'
        }).reset_index()
        
        referral_stats.columns = ['referrer_category', 'random_group', 'total_visits', 'total_signups']
        referral_stats['conversion_rate'] = (referral_stats['total_signups'] / referral_stats['total_visits'] * 100).round(2)
        
        # Calculate percentage of total traffic
        total_visits = referral_stats['total_visits'].sum()
        referral_stats['traffic_share'] = (referral_stats['total_visits'] / total_visits * 100).round(2)
        
        # Sort by total visits
        referral_stats = referral_stats.sort_values('total_visits', ascending=False)
    
    return referral_stats

def parse_error_log(error_log_content):
    """Parse error log to extract timestamps and IP addresses."""
    # Pattern to match both IP logs and email-related logs
    ip_pattern = r'\[(.*?)\] wp_get_client_ip fired\. IP found: ([\da-fA-F:\.]+)'
    email_pattern = r'\[(.*?)\] Email captured: ([^\n]+)'
    
    log_data = []
    
    # Process IP addresses
    ip_matches = re.finditer(ip_pattern, error_log_content)
    for match in ip_matches:
        timestamp_str, ip = match.groups()
        try:
            # Handle UTC timezone explicitly
            timestamp_str = timestamp_str.replace(' UTC', '')
            timestamp = datetime.strptime(timestamp_str, '%d-%b-%Y %H:%M:%S')
            log_data.append({
                'timestamp': timestamp,
                'ip_address': ip.strip(),
                'event_type': 'ip_log'
            })
        except ValueError as e:
            st.warning(f"Failed to parse timestamp: {timestamp_str} - {str(e)}")
            continue
    
    # Process email captures
    email_matches = re.finditer(email_pattern, error_log_content)
    for match in email_matches:
        timestamp_str, email = match.groups()
        try:
            timestamp_str = timestamp_str.replace(' UTC', '')
            timestamp = datetime.strptime(timestamp_str, '%d-%b-%Y %H:%M:%S')
            log_data.append({
                'timestamp': timestamp,
                'email': email.strip(),
                'event_type': 'email_capture'
            })
        except ValueError as e:
            st.warning(f"Failed to parse timestamp: {timestamp_str} - {str(e)}")
            continue
    
    if not log_data:
        st.warning("No valid log entries found in the error log.")
        return pd.DataFrame(columns=['timestamp', 'ip_address', 'email', 'event_type'])
    
    df = pd.DataFrame(log_data)
    # Ensure timestamp is timezone-aware
    df['timestamp'] = pd.to_datetime(df['timestamp']).dt.tz_localize('UTC')
    return df

def get_location_from_ip(ip_address):
    """Get location information from IP address using GeoLite2 database."""
    try:
        if not ip_address or pd.isna(ip_address):
            return {
                'country': 'Unknown',
                'city': 'Unknown',
                'latitude': None,
                'longitude': None,
                'timezone': None
            }
            
        with Reader('GeoLite2-City.mmdb') as reader:
            response = reader.city(ip_address)
            return {
                'country': response.country.name or 'Unknown',
                'city': response.city.name or 'Unknown',
                'latitude': response.location.latitude,
                'longitude': response.location.longitude,
                'timezone': response.location.time_zone
            }
    except (AddressNotFoundError, maxminddb.errors.InvalidDatabaseError, ValueError) as e:
        st.warning(f"Could not get location for IP {ip_address}: {str(e)}")
        return {
            'country': 'Unknown',
            'city': 'Unknown',
            'latitude': None,
            'longitude': None,
            'timezone': None
        }
    except Exception as e:
        st.error(f"Unexpected error getting location for IP {ip_address}: {str(e)}")
        return {
            'country': 'Unknown',
            'city': 'Unknown',
            'latitude': None,
            'longitude': None,
            'timezone': None
        }

def merge_ip_data(clean_tracker, error_log_df):
    """Merge IP data from error log with tracker data based on timestamps."""
    try:
        # Early return if either dataframe is empty
        if clean_tracker.empty or error_log_df.empty:
            st.warning("No data available for IP mapping.")
            return pd.DataFrame(columns=['ip_address', 'uuid'])
        
        # Ensure timestamps are in UTC
        clean_tracker['timestamp'] = pd.to_datetime(clean_tracker['timestamp'], utc=True)
        if 'timestamp' in error_log_df.columns:
            error_log_df['timestamp'] = pd.to_datetime(error_log_df['timestamp'], utc=True)
        else:
            st.warning("No timestamp column found in error log data.")
            return pd.DataFrame(columns=['ip_address', 'uuid'])
        
        # Filter for session_start events and relevant IP logs
        session_starts = clean_tracker[clean_tracker['event'] == 'session_start'].copy()
        ip_logs = error_log_df[error_log_df['event_type'] == 'ip_log'].copy()
        
        if session_starts.empty or ip_logs.empty:
            st.warning("No matching session starts or IP logs found.")
            return pd.DataFrame(columns=['ip_address', 'uuid'])
        
        # Sort both dataframes by timestamp
        session_starts = session_starts.sort_values('timestamp')
        ip_logs = ip_logs.sort_values('timestamp')
        
        # Merge based on closest timestamp within 1 second tolerance
        merged_data = pd.merge_asof(
            session_starts,
            ip_logs[['timestamp', 'ip_address']],
            on='timestamp',
            tolerance=pd.Timedelta('1s'),
            direction='nearest'
        )
        
        # Create IP to UUID mapping
        ip_uuid_map = merged_data[['ip_address', 'uuid', 'random_group']].dropna().drop_duplicates()
        
        return ip_uuid_map
        
    except Exception as e:
        st.error(f"Error merging IP data: {str(e)}")
        return pd.DataFrame(columns=['ip_address', 'uuid', 'random_group'])

def analyze_visitor_locations(ip_uuid_map, uuid_tracker):
    """Analyze visitor locations by experimental group."""
    # Get location data for each IP
    location_data = []
    error_ips = []
    
    for _, row in ip_uuid_map.iterrows():
        try:
            location = get_location_from_ip(row['ip_address'])
            location['uuid'] = row['uuid']
            location['random_group'] = row['random_group']
            location_data.append(location)
        except Exception as e:
            error_ips.append((row['ip_address'], str(e)))
            continue
    
    if not location_data:
        return pd.DataFrame(), pd.DataFrame()
    
    location_df = pd.DataFrame(location_data)
    
    # Ensure all groups are represented
    all_groups = range(1, 5)  # Groups 1-4
    
    # Generate statistics with all groups
    country_stats = []
    for country in location_df['country'].unique():
        for group in all_groups:
            group_data = location_df[
                (location_df['country'] == country) & 
                (location_df['random_group'] == group)
            ]
            count = len(group_data)
            total_in_group = len(location_df[location_df['random_group'] == group])
            percentage = (count / total_in_group * 100) if total_in_group > 0 else 0
            
            country_stats.append({
                'country': country,
                'random_group': group,
                'count': count,
                'percentage': round(percentage, 2)
            })
    
    country_stats_df = pd.DataFrame(country_stats)
    
    # Store error IPs for later display
    if error_ips:
        location_df['error_ips'] = pd.Series(error_ips)
    
    return location_df, country_stats_df

def create_location_charts(location_analysis):
    """Create visualizations for location data."""
    # Add base map layer
    base = alt.Chart(alt.topo_feature('https://cdn.jsdelivr.net/npm/world-atlas@2/countries-110m.json', 'countries')).mark_geoshape(
        fill='lightgray',
        stroke='white'
    ).properties(
        width=800,
        height=400
    )
    
    # Points layer
    points = alt.Chart(location_analysis).mark_circle().encode(
        longitude='longitude:Q',
        latitude='latitude:Q',
        size=alt.value(60),
        color=alt.Color('random_group:N', title='Group'),
        tooltip=['country:N', 'city:N', 'random_group:N']
    )
    
    # Combine layers
    map_chart = (base + points).properties(
        title='Visitor Locations by Experimental Group'
    )
    
    # Country distribution chart
    country_chart = alt.Chart(
        location_analysis.groupby(['country', 'random_group']).size().reset_index(name='count')
    ).mark_bar().encode(
        x=alt.X('country:N', title='Country', sort='-y'),
        y=alt.Y('count:Q', title='Number of Visitors'),
        color=alt.Color('random_group:N', title='Group'),
        tooltip=['country:N', 'random_group:N', 'count:Q']
    ).properties(
        width=600,
        height=400,
        title='Visitor Distribution by Country and Group'
    )
    
    return map_chart, country_chart

def analyze_user_types(clean_tracker):
    """Analyze user behavior by new vs returning users."""
    # Get first event for each user
    first_events = clean_tracker.groupby('uuid').agg({
        'timestamp': 'min',
        'referrer': lambda x: x.iloc[0] if not x.isna().all() else None,
        'random_group': 'first'
    }).reset_index()
    
    # Identify new users (those with referral links)
    first_events['is_new_user'] = first_events['referrer'].notna()
    
    # Get newsletter signups
    newsletter_signups = clean_tracker[clean_tracker['event'] == 'newsletter_signup']['uuid'].unique()
    first_events['has_signup'] = first_events['uuid'].isin(newsletter_signups)
    
    return first_events

def calculate_user_type_stats(user_type_data):
    """Calculate detailed statistics for new vs returning users by group."""
    stats = []
    
    for group in sorted(user_type_data['random_group'].unique()):
        group_data = user_type_data[user_type_data['random_group'] == group]
        
        # Calculate stats for new users
        new_users = group_data[group_data['is_new_user']]
        returning_users = group_data[~group_data['is_new_user']]
        
        stats.append({
            'Group': group,
            'New Users': len(new_users),
            'New User Signup Rate': (new_users['has_signup'].mean() * 100).round(2),
            'Returning Users': len(returning_users),
            'Returning User Signup Rate': (returning_users['has_signup'].mean() * 100).round(2),
            'Total Users': len(group_data),
            'Overall Signup Rate': (group_data['has_signup'].mean() * 100).round(2)
        })
    
    return pd.DataFrame(stats)

def create_user_type_charts(user_type_data):
    """Create visualizations for user type analysis."""
    # Prepare data for visualization
    chart_data = []
    
    for group in sorted(user_type_data['random_group'].unique()):
        group_data = user_type_data[user_type_data['random_group'] == group]
        
        # New users
        new_users = group_data[group_data['is_new_user']]
        new_signup_rate = new_users['has_signup'].mean() * 100
        chart_data.append({
            'Group': f'Group {group}',
            'User Type': 'New',
            'Signup Rate': new_signup_rate
        })
        
        # Returning users
        returning_users = group_data[~group_data['is_new_user']]
        returning_signup_rate = returning_users['has_signup'].mean() * 100
        chart_data.append({
            'Group': f'Group {group}',
            'User Type': 'Returning',
            'Signup Rate': returning_signup_rate
        })
    
    chart_df = pd.DataFrame(chart_data)
    
    # Create visualization
    chart = alt.Chart(chart_df).mark_bar().encode(
        x=alt.X('Group:N', title='Experimental Group'),
        y=alt.Y('Signup Rate:Q', title='Signup Rate (%)'),
        color=alt.Color('User Type:N', scale=alt.Scale(scheme='set2')),
        tooltip=[
            alt.Tooltip('Group:N'),
            alt.Tooltip('User Type:N'),
            alt.Tooltip('Signup Rate:Q', format='.1f')
        ]
    ).properties(
        title='Signup Rates by User Type and Group',
        width=600,
        height=400
    )
    
    return chart

def analyze_time_series(clean_tracker, window_size=7):
    """Analyze time series patterns in signup data with enhanced campaign analysis."""
    try:
        # Convert to daily signup counts
        daily_signups = clean_tracker[clean_tracker['event'] == 'newsletter_signup'].copy()
        daily_signups['date'] = daily_signups['timestamp'].dt.date
        
        # Get user type information
        user_type_info = analyze_user_types(clean_tracker)
        daily_signups = daily_signups.merge(
            user_type_info[['uuid', 'is_new_user']],
            on='uuid',
            how='left'
        )
        
        # Fill missing values
        daily_signups['is_new_user'] = daily_signups['is_new_user'].fillna(False)
        
        # Calculate daily counts by group and user type
        daily_counts = daily_signups.groupby(['date', 'random_group', 'is_new_user']).size().reset_index(name='signups')
        
        # Ensure all combinations exist
        all_dates = pd.date_range(daily_counts['date'].min(), daily_counts['date'].max(), freq='D').date
        all_groups = clean_tracker['random_group'].unique()
        user_types = [True, False]
        
        # Create all possible combinations
        index = pd.MultiIndex.from_product(
            [all_dates, all_groups, user_types],
            names=['date', 'random_group', 'is_new_user']
        )
        daily_counts = daily_counts.set_index(['date', 'random_group', 'is_new_user']).reindex(index, fill_value=0).reset_index()
        
        # Calculate rolling averages
        daily_counts['rolling_avg'] = daily_counts.groupby(['random_group', 'is_new_user'])['signups'].transform(
            lambda x: x.rolling(window=window_size, min_periods=1).mean()
        )
        
        # Add day of week analysis
        daily_counts['day_of_week'] = pd.to_datetime(daily_counts['date']).dt.day_name()
        
        # Calculate day-of-week patterns
        dow_patterns = daily_counts.groupby(['day_of_week', 'random_group', 'is_new_user'])['signups'].agg([
            'mean', 'std', 'count'
        ]).reset_index()
        
        # Detect anomalies (using z-score method)
        z_threshold = 2.5
        daily_counts['zscore'] = daily_counts.groupby(['random_group', 'is_new_user'])['signups'].transform(
            lambda x: (x - x.mean()) / (x.std() if x.std() != 0 else 1)
        )
        daily_counts['is_anomaly'] = abs(daily_counts['zscore']) > z_threshold
        
        return daily_counts, dow_patterns
    
    except Exception as e:
        st.error(f"Error analyzing time series data: {str(e)}")
        return pd.DataFrame(), pd.DataFrame()

def create_time_series_charts(daily_counts):
    """Create enhanced time series visualizations with user type breakdown."""
    try:
        # Rolling average chart
        rolling_avg_chart = alt.Chart(daily_counts).mark_line().encode(
            x=alt.X('date:T', title='Date'),
            y=alt.Y('rolling_avg:Q', title='7-Day Rolling Average Signups'),
            color=alt.Color('random_group:N', title='Group'),
            strokeDash=alt.StrokeDash(
                'is_new_user:N', 
                title='User Type',
                scale=alt.Scale(domain=[True, False], range=[[5,5], [0]]),
                legend=alt.Legend(
                    title='User Type',
                    symbolType='square',
                    labelExpr="datum.value ? 'New' : 'Returning'"
                )
            ),
            tooltip=[
                alt.Tooltip('date:T'),
                alt.Tooltip('random_group:N'),
                alt.Tooltip('is_new_user:N', title='New User'),
                alt.Tooltip('rolling_avg:Q', format='.1f'),
                alt.Tooltip('signups:Q', format='.0f')
            ]
        ).properties(
            title='Newsletter Signup Trends by Group and User Type',
            width=800,
            height=400
        )
        
        # Day of week patterns
        dow_chart = alt.Chart(daily_counts).mark_bar().encode(
            x=alt.X('day_of_week:N', 
                   title='Day of Week',
                   sort=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'],
                   axis=alt.Axis(labelAngle=-45)),
            y=alt.Y('mean(signups):Q', title='Average Signups'),
            color=alt.Color('random_group:N', title='Group'),
            column=alt.Column(
                'is_new_user:N',
                title='User Type',
                header=alt.Header(
                    titleFontSize=14,
                    labelExpr="datum.value ? 'New Users' : 'Returning Users'"
                ),
                spacing=20  # Add spacing between columns
            ),
            tooltip=[
                alt.Tooltip('day_of_week:N'),
                alt.Tooltip('random_group:N'),
                alt.Tooltip('mean(signups):Q', format='.1f')
            ]
        ).properties(
            title='Signup Patterns by Day of Week',
            width=350,  # Increase width for better spacing
            height=300
        ).configure_view(
            stroke=None  # Remove cell borders
        )
        
        # Anomaly visualization
        anomaly_base = alt.Chart(daily_counts).mark_line(opacity=0.2).encode(
            x=alt.X('date:T', title='Date'),
            y=alt.Y('signups:Q', title='Number of Signups'),
            color=alt.Color('random_group:N', title='Group'),
            strokeDash=alt.StrokeDash(
                'is_new_user:N',
                title='User Type',
                legend=alt.Legend(
                    labelExpr="datum.value ? 'New' : 'Returning'"
                )
            )
        )
        
        anomaly_points = alt.Chart(daily_counts[daily_counts['is_anomaly']]).mark_point(
            size=100,
            shape='diamond',
            filled=True
        ).encode(
            x=alt.X('date:T', title='Date'),
            y=alt.Y('signups:Q', title='Number of Signups'),
            color=alt.Color('random_group:N', title='Group'),
            shape=alt.Shape(
                'is_new_user:N',
                title='User Type',
                scale=alt.Scale(domain=[True, False], range=['triangle', 'circle']),
                legend=alt.Legend(
                    labelExpr="datum.value ? 'New' : 'Returning'"
                )
            ),
            tooltip=[
                alt.Tooltip('date:T'),
                alt.Tooltip('random_group:N'),
                alt.Tooltip('is_new_user:N', title='New User'),
                alt.Tooltip('signups:Q', format='.0f'),
                alt.Tooltip('zscore:Q', format='.2f')
            ]
        )
        
        # Combine base line and anomaly points
        anomaly_chart = (anomaly_base + anomaly_points).properties(
            title='Signup Anomalies (Diamonds indicate anomalous points)',
            width=800,
            height=300
        ).configure_view(
            stroke=None
        )
        
        return rolling_avg_chart, dow_chart, anomaly_chart
    
    except Exception as e:
        st.error(f"Error creating time series charts: {str(e)}")
        # Return empty charts as fallback
        empty_chart = alt.Chart().mark_point()
        return empty_chart, empty_chart, empty_chart

def calculate_ab_test_stats(uuid_tracker):
    """Calculate A/B test statistics and confidence intervals."""
    test_results = []
    control_group = 4  # Pure Control group
    
    for test_group in range(1, 4):  # Test groups 1-3
        # Get signup rates for both groups
        control_data = uuid_tracker[uuid_tracker['random_group'] == control_group]
        test_data = uuid_tracker[uuid_tracker['random_group'] == test_group]
        
        # Calculate conversion rates and sample sizes
        control_conv = control_data['num_newsletter_signup'].mean()
        test_conv = test_data['num_newsletter_signup'].mean()
        control_n = len(control_data)
        test_n = len(test_data)
        
        # Calculate confidence intervals
        control_ci = proportion_confint(
            count=(control_data['num_newsletter_signup'] > 0).sum(),
            nobs=control_n,
            alpha=0.05,
            method='wilson'
        )
        test_ci = proportion_confint(
            count=(test_data['num_newsletter_signup'] > 0).sum(),
            nobs=test_n,
            alpha=0.05,
            method='wilson'
        )
        
        # Perform statistical test
        t_stat, p_value = scipy.stats.ttest_ind(
            control_data['num_newsletter_signup'],
            test_data['num_newsletter_signup']
        )
        
        # Calculate effect size (relative lift)
        relative_lift = ((test_conv - control_conv) / control_conv) * 100 if control_conv > 0 else 0
        
        # Calculate power
        effect_size = (test_conv - control_conv) / np.sqrt(
            (control_data['num_newsletter_signup'].var() + test_data['num_newsletter_signup'].var()) / 2
        )
        power_analysis = TTestPower().power(
            effect_size=effect_size,
            nobs=min(control_n, test_n),
            alpha=0.05
        )
        
        test_results.append({
            'test_group': test_group,
            'control_conv_rate': control_conv,
            'test_conv_rate': test_conv,
            'relative_lift': relative_lift,
            'p_value': p_value,
            'power': power_analysis,
            'control_ci_lower': control_ci[0],
            'control_ci_upper': control_ci[1],
            'test_ci_lower': test_ci[0],
            'test_ci_upper': test_ci[1],
            'control_sample_size': control_n,
            'test_sample_size': test_n
        })
    
    return pd.DataFrame(test_results)

def create_ab_test_charts(ab_test_results):
    """Create visualizations for A/B test results."""
    # Conversion rate comparison chart with confidence intervals
    conv_rate_chart = alt.Chart(ab_test_results).mark_bar().encode(
        x=alt.X('test_group:N', title='Test Group'),
        y=alt.Y('test_conv_rate:Q', title='Conversion Rate'),
        color=alt.Color('test_group:N', title='Group'),
        tooltip=['test_group', 'test_conv_rate', 'relative_lift', 'p_value']
    ).properties(
        title='Conversion Rates by Test Group',
        width=600,
        height=400
    )
    
    # Add error bars for confidence intervals
    error_bars = alt.Chart(ab_test_results).mark_errorbar().encode(
        x=alt.X('test_group:N', title='Test Group'),
        y=alt.Y('test_ci_lower:Q', title='Conversion Rate'),
        y2=alt.Y2('test_ci_upper:Q')
    )
    
    # Combine charts
    conv_rate_chart = conv_rate_chart + error_bars
    
    return conv_rate_chart