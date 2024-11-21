import streamlit as st
import pandas as pd
import plotly.express as px
import os
import requests
import time



# URL of the raw CSV file
url = "https://raw.githubusercontent.com/klg125/hackathon/refs/heads/main/team_refined.csv"

# Output file path
output_file = "team_refined.csv"

# Download the file
response = requests.get(url)

if response.status_code == 200:
    with open(output_file, "wb") as file:
        file.write(response.content)
else:
    st.error(f"Failed to download file. HTTP Status Code: {response.status_code}")

# Load and preprocess data
df_comparable = pd.read_csv(output_file)
df_comparable["date"] = pd.to_datetime(df_comparable["date"])  # Convert 'date' to datetime
df_comparable = df_comparable.sort_values("date")  # Ensure dates are sorted

# Convert min_date and max_date to datetime.date
min_date, max_date = df_comparable["date"].min().date(), df_comparable["date"].max().date()

# Updated get_northwestern_strategy function
def get_northwestern_strategy(school_name, profiles_folder="school_profiles"):
    # Explicit mapping between teams and file names
    team_file_mapping = {
        ' Rutgers Scarlet Knights': 'Rutgers_Scarlet_Knights__.txt',
        ' Michigan Wolverines': 'Michigan_Wolverines__.txt',
        ' Maryland College Park Terrapins': 'Maryland_Terrapins__.txt',
        ' Penn State Nittany Lion': 'Penn_State_Nittany_Lions__.txt',
        ' Indiana Hoosiers': 'Indiana_Hoosiers__.txt',
        ' Wisconsin Badgers': 'Wisconsin_Badgers__.txt',
        ' Ohio State Buckeyes': 'Ohio_State_Buckeyes__.txt',
        ' Northwestern Wildcats': 'Northwestern_Wildcats__.txt',
        ' Washington Huskies': 'Washington_Huskies__.txt',
        ' Michigan State Spartans': 'Michigan_State_Spartans__.txt',
        ' UCLA Bruins': 'UCLA_Bruins__.txt'
    }

    # Get the corresponding file name for the school
    file_name = team_file_mapping.get(school_name)
    
    if not file_name:
        return "No strategy information available for this team."

    # Construct the file path and read the file
    file_path = os.path.join(profiles_folder, file_name)
    if os.path.exists(file_path):
        with open(file_path, "r") as file:
            content = file.read()
            # Extract relevant sections
            strategy_start = content.find("Northwestern's Strategy:")
            strategy_end = content.find("- **Recommended Columns:**")
            strengths_start = content.find("- **Strengths:**")
            weaknesses_start = content.find("- **Weaknesses:**")
            
            if strategy_end == -1:  # Handle if no "Recommended Columns" section exists
                strategy_end = len(content)

            strategy = content[strategy_start:strategy_end].strip() if strategy_start != -1 else "No strategy available."
            strengths = content[strengths_start:weaknesses_start].strip() if strengths_start != -1 else "No strengths available."
            weaknesses = content[weaknesses_start:strategy_start].strip() if weaknesses_start != -1 else "No weaknesses available."
            recommended_columns = content[strategy_end:].strip() if "- **Recommended Columns:**" in content else "No recommended columns available."
            
            return strategy, strengths, weaknesses, recommended_columns

    return "No strategy information available for this team.", "No strengths available.", "No weaknesses available.", "No recommended columns available."

# Streamlit App
st.title("Competitor Analysis")
st.sidebar.header("Filters")

# Team Selector
teams = df_comparable["team"].unique()
selected_team = st.sidebar.selectbox("Select Team", options=teams)

# Y-axis Selector
y_columns = [
    'goals', 'xg', 'pass_success_rate', 'possession', 'tot_duels_win_rate',
    'sobotr', 'cornwsr', 'fkwsr', 'pen_conversion_rate', 'acc_cross_rate',
    'succ_for_passes_rate', 'succ_back_passes_rate', 'succ_lat_passes_rate',
    'succ_long_passes_rate', 'succ_fin_third_passes_rate', 'succ_prog_passes_rate',
    'succ_smart_passes_rate', 'succ_throw_ins_rate', 'mean_shot_dist',
    'mean_pass_len', 'ppda', 'passes_per_minute', 'goal_conversion_rate',
    'touches_in_box_per_minute', 'recoveries_per_minute', 'shots_on_target_rate'
]
selected_y_column = st.sidebar.selectbox("Select Y-axis", options=y_columns, index=0)

# Recency Filter
st.sidebar.header("Recency Filter")
recency_option = st.sidebar.radio("Select Recency:", options=["Last Match", "Last 5 Matches", "All Time"])

# Adjust strategy folder and date range based on the recency option
if recency_option == "Last Match":
    strategy_folder = "school_profiles_1"
    start_date = pd.Timestamp("2024-11-01")
    end_date = df_comparable["date"].max()  # Use the latest date in the dataset
elif recency_option == "Last 5 Matches":
    strategy_folder = "school_profiles_5"
    start_date = pd.Timestamp("2024-10-01")
    end_date = df_comparable["date"].max()  # Use the latest date in the dataset
else:  # "All Time"
    strategy_folder = "school_profiles"
    start_date = df_comparable["date"].min()
    end_date = df_comparable["date"].max()

# Update the date range slider
selected_date_range = st.sidebar.slider(
    "Select Date Range",
    value=(start_date.date(), end_date.date()),
    min_value=df_comparable["date"].min().date(),
    max_value=df_comparable["date"].max().date(),
    format="MM/DD/YY"
)

# Add a new column to detect year transitions
df_comparable['year'] = df_comparable['date'].dt.year

# Filter data for the selected team
filtered_data = df_comparable[
    (df_comparable["date"] >= pd.Timestamp(selected_date_range[0])) &
    (df_comparable["date"] <= pd.Timestamp(selected_date_range[1])) &
    (df_comparable["team"] == selected_team)
]

# Drop NaN for the selected metric
filtered_data = filtered_data.dropna(subset=[selected_y_column])

# Handle year transitions to avoid connecting lines across years
filtered_data['year'] = filtered_data['date'].dt.year
filtered_data['year_diff'] = filtered_data['year'].diff()
filtered_data.loc[filtered_data['year_diff'] != 0, selected_y_column] = None

# Overlay Northwestern's stats if a team other than Northwestern is selected
if selected_team != "Northwestern Wildcats":
    northwestern_data = df_comparable[
        (df_comparable["date"] >= pd.Timestamp(selected_date_range[0])) &
        (df_comparable["date"] <= pd.Timestamp(selected_date_range[1])) &
        (df_comparable["team"] == "Northwestern Wildcats")
    ]
    
    northwestern_data = northwestern_data.dropna(subset=[selected_y_column])
    northwestern_data['year'] = northwestern_data['date'].dt.year
    northwestern_data['year_diff'] = northwestern_data['year'].diff()
    northwestern_data.loc[northwestern_data['year_diff'] != 0, selected_y_column] = None
    
else:
    northwestern_data = None

# Plot Line Chart
fig = px.line(
    filtered_data,
    x="date",
    y=selected_y_column,
    title=f"{selected_y_column.capitalize()} Over Time for {selected_team}",
    labels={"date": "Date", selected_y_column: selected_y_column.capitalize()},
    line_shape="linear",  # Linear lines without interpolation
)

# Add Northwestern's stats as a secondary trace if applicable
if northwestern_data is not None:
    fig.add_scatter(
        x=northwestern_data["date"],
        y=northwestern_data[selected_y_column],
        mode="lines",
        name="Northwestern Wildcats",
        line=dict(dash="dash", color="orange"),  # Dash
    )

# Render the chart
st.plotly_chart(fig, use_container_width=True)

# Update get_northwestern_strategy to dynamically handle the strategy folder
def get_northwestern_strategy(school_name, profiles_folder):
    # Explicit mapping between teams and file names
    team_file_mapping = {
        ' Rutgers Scarlet Knights': 'Rutgers_Scarlet_Knights__.txt',
        ' Michigan Wolverines': 'Michigan_Wolverines__.txt',
        ' Maryland College Park Terrapins': 'Maryland_Terrapins__.txt',
        ' Penn State Nittany Lion': 'Penn_State_Nittany_Lions__.txt',
        ' Indiana Hoosiers': 'Indiana_Hoosiers__.txt',
        ' Wisconsin Badgers': 'Wisconsin_Badgers__.txt',
        ' Ohio State Buckeyes': 'Ohio_State_Buckeyes__.txt',
        ' Northwestern Wildcats': 'Northwestern_Wildcats__.txt',
        ' Washington Huskies': 'Washington_Huskies__.txt',
        ' Michigan State Spartans': 'Michigan_State_Spartans__.txt',
        ' UCLA Bruins': 'UCLA_Bruins__.txt'
    }

    # Get the corresponding file name for the school
    file_name = team_file_mapping.get(school_name)
    
    if not file_name:
        return "No strategy information available for this team."

    # Construct the file path and read the file
    file_path = os.path.join(profiles_folder, file_name)
    if os.path.exists(file_path):
        with open(file_path, "r") as file:
            content = file.read()
            # Extract relevant sections
            strategy_start = content.find("Northwestern's Strategy:")
            strategy_end = content.find("- **Recommended Columns:**")
            strengths_start = content.find("- **Strengths:**")
            weaknesses_start = content.find("- **Weaknesses:**")
            
            if strategy_end == -1:  # Handle if no "Recommended Columns" section exists
                strategy_end = len(content)

            strategy = content[strategy_start:strategy_end].strip() if strategy_start != -1 else "No strategy available."
            strengths = content[strengths_start:weaknesses_start].strip() if strengths_start != -1 else "No strengths available."
            weaknesses = content[weaknesses_start:strategy_start].strip() if weaknesses_start != -1 else "No weaknesses available."
            recommended_columns = content[strategy_end:].strip() if "- **Recommended Columns:**" in content else "No recommended columns available."
            
            return strategy, strengths, weaknesses, recommended_columns

    return "No strategy information available for this team.", "No strengths available.", "No weaknesses available.", "No recommended columns available."

# Display the strategy in a formatted way
st.subheader("Northwestern's Strategy Against Selected Team")
with st.spinner("Loading strategy..."):
    time.sleep(1)  # Simulate loading time
    strategy, strengths, weaknesses, recommended_columns = get_northwestern_strategy(selected_team, strategy_folder)
    formatted_strategy = strategy.replace("**:", ":").replace("- ", "\n- **").replace("**", "")
    st.markdown(formatted_strategy)

# Expandable Sections
with st.expander("Strengths"):
    st.write(strengths)

with st.expander("Weaknesses"):
    st.write(weaknesses)

with st.expander("Recommended Columns"):
    st.write(recommended_columns)