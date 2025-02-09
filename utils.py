import pandas as pd
import numpy as np
import altair as alt
import requests
import json
import scipy.stats 
import streamlit as st
from itertools import combinations

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