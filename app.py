"""
South Carolina Location Analysis Tool
A Streamlit application for optimal facility location selection using Maximum Coverage Location-Allocation

METHODOLOGY DOCUMENTATION:
========================

1. LOCATION-ALLOCATION MODEL TYPE:
   - Maximum Coverage Location Problem (MCLP)
   - Objective: Maximize the total covered demand (uninsured population) subject to selecting 
     exactly P facilities from a set of candidate locations
   
2. NETWORK ANALYSIS DATA SOURCES:
   When "Use Road Network Analysis" is enabled:
   - Network Data: OpenStreetMap (OSM) via OSMnx library
   - Road Network Type: 
     * 'drive' mode: Drivable roads with speed limits based on road classification
     * 'walk' mode: Walkable paths (includes sidewalks, pedestrian paths)
   
3. TRAVEL TIME CALCULATION:
   
   A. Network-Based (when enabled):
      - Method: Dijkstra's shortest path algorithm on actual road network
      - Distance Metric: Network distance (meters along roads)
      - Speed Assumptions:
        DRIVING (road-type based speed profiles):
        * motorway/trunk: 65 mph (105 km/h)
        * primary: 50 mph (80 km/h)
        * secondary: 40 mph (64 km/h)
        * tertiary: 30 mph (48 km/h)
        * residential: 25 mph (40 km/h)
        * service/unclassified: 20 mph (32 km/h)
        
        WALKING:
        * All paths: 5 km/h (3.1 mph) - standard pedestrian speed
      
      - Time Calculation: Network distance / speed (by road segment)
      - Service Area: Actual isochrone (all reachable points within time threshold)
   
   B. Manhattan Distance (default):
      - Method: Rectilinear distance approximation
      - Formula: |Δlat| × 69 + |Δlon| × 69 × cos(lat)
      - Speed: Average 30 mph for driving, 5 km/h for walking
   
   NOTE: OpenStreetMap does not include historical traffic data or posted speed limits
   in most areas. For true historical speeds like ArcGIS Pro, you would need:
   - Esri StreetMap Premium / HERE data (commercial)
   - Google Maps API / Mapbox API (requires API key)
   - OSRM with custom speed profiles (requires server setup)
   
4. COVERAGE DETERMINATION:
   - A facility "covers" a demand point if travel time ≤ threshold
   - Binary coverage matrix: C[i,j] = 1 if facility i covers demand point j, 0 otherwise
   - Coverage is weighted by uninsured population at each demand point
   
5. OPTIMIZATION SOLVER:
   - Solver: PuLP with CBC (COIN-OR Branch and Cut) solver
   - Problem Type: Integer Linear Programming (ILP)
"""

import streamlit as st
import pandas as pd
import geopandas as gpd
import folium
import json
from pathlib import Path
import numpy as np
from pulp import *
import osmnx as ox
import networkx as nx
from shapely.geometry import Point, Polygon, MultiPolygon
from shapely.ops import unary_union
from streamlit_folium import st_folium
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="SC Location Analysis Tool",
    page_icon="🗺️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    /* Style for multiselect pills */
    .stMultiSelect [data-baseweb="tag"] {
        background-color: #ff4b4b !important;
        border-radius: 20px !important;
        padding: 5px 10px !important;
        margin: 2px !important;
    }
    
    /* Make stats cards more compact */
    [data-testid="stMetricValue"] {
        font-size: 24px !important;
    }
    
    /* Improve sidebar spacing */
    .css-1d391kg {
        padding-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Default configuration (can be overridden by config.py)
JSON_PATH = Path("sc_app_data.json")
DEFAULT_TRAVEL_MODE = 'drive'
DEFAULT_TIME_THRESHOLD = 10
DEFAULT_NUM_FACILITIES = 3
DEFAULT_USE_NETWORK = False
WALKING_SPEED_KMH = 5.0  # Changed from 3 mph to 5 km/h

# Road-type based speed profiles (mph)
SPEED_PROFILES = {
    'motorway': 65,
    'motorway_link': 55,
    'trunk': 65,
    'trunk_link': 55,
    'primary': 50,
    'primary_link': 45,
    'secondary': 40,
    'secondary_link': 35,
    'tertiary': 30,
    'tertiary_link': 25,
    'residential': 25,
    'living_street': 15,
    'service': 20,
    'unclassified': 20,
    'road': 25
}
DEFAULT_DRIVING_SPEED = 30  # Fallback speed for Manhattan distance

FACILITY_COLORS = {
    'Church': 'purple',
    'Primary Care': 'red',
    'Grocery': 'green',
    'Other': 'gray'
}

# Try to import custom configuration (optional)
try:
    from config import *
except ImportError:
    pass  # Use defaults defined above

# Initialize session state for preserving analysis results
if 'analysis_complete' not in st.session_state:
    st.session_state.analysis_complete = False
if 'selected_facilities' not in st.session_state:
    st.session_state.selected_facilities = None
if 'coverage_matrix' not in st.session_state:
    st.session_state.coverage_matrix = None
if 'demand_reset' not in st.session_state:
    st.session_state.demand_reset = None
if 'candidates_reset' not in st.session_state:
    st.session_state.candidates_reset = None
if 'covered_pop' not in st.session_state:
    st.session_state.covered_pop = 0
if 'service_areas' not in st.session_state:
    st.session_state.service_areas = None
if 'last_params' not in st.session_state:
    st.session_state.last_params = {}


@st.cache_data
def load_data(json_path):
    """
    Load all data from the JSON file
    Returns: zip_gdf, candidates_df, demand_df
    """
    with open(json_path, 'r') as f:
        data = json.load(f)

    # Load ZIP code boundaries
    from shapely.geometry import Polygon

    zip_data = data.get('zip_boundaries', data.get('zips', {}))

    if not zip_data:
        raise ValueError("No ZIP boundaries found")

    geometries = []
    properties = []

    # Handle dictionary format where keys are ZIP codes
    if isinstance(zip_data, dict):
        for zip_code, zip_info in zip_data.items():
            try:
                # Your format: coords is a list of [lat, lon] pairs
                if 'coords' in zip_info:
                    coords = zip_info['coords']
                    # Convert from [lat, lon] to [lon, lat] for Shapely
                    coords_lonlat = [[pt[1], pt[0]] for pt in coords]

                    # Create polygon geometry
                    geom = Polygon(coords_lonlat)
                    geometries.append(geom)

                    # Build properties
                    props = {
                        'ZIP_CODE': zip_code,
                        'po_name': zip_info.get('po_name', zip_code)
                    }
                    properties.append(props)

            except Exception as e:
                st.warning(f"Error parsing ZIP {zip_code}: {e}")
                continue
    else:
        raise ValueError("ZIP data should be a dictionary with ZIP codes as keys")

    if not geometries:
        raise ValueError(f"Could not parse any ZIP boundary geometries from {len(zip_data)} ZIP codes")

    # Create GeoDataFrame
    zip_gdf = gpd.GeoDataFrame(properties, geometry=geometries, crs='EPSG:4326')

    # Load facilities
    candidates_data = data.get('candidate_facilities', data.get('facilities', []))
    if not candidates_data:
        raise ValueError("No facilities found")
    candidates_df = pd.DataFrame(candidates_data)

    # Load demand points
    demand_data = data.get('demand_points', data.get('demand', []))
    if not demand_data:
        raise ValueError("No demand points found")
    demand_df = pd.DataFrame(demand_data)

    return zip_gdf, candidates_df, demand_df


def add_speed_to_network(G, network_type='drive'):
    """
    Add speed attributes to network edges based on road type
    
    For driving: Uses road classification-based speed profiles
    For walking: Uses constant 5 km/h
    
    Returns: Graph with 'speed_kph' attribute added to edges
    """
    if network_type == 'walk':
        # Walking: constant 5 km/h
        for u, v, data in G.edges(data=True):
            data['speed_kph'] = WALKING_SPEED_KMH
    else:
        # Driving: road-type based speeds
        for u, v, data in G.edges(data=True):
            road_type = data.get('highway', 'unclassified')
            
            # Handle lists (sometimes highway is a list of types)
            if isinstance(road_type, list):
                road_type = road_type[0]
            
            # Get speed from profile, convert mph to km/h
            speed_mph = SPEED_PROFILES.get(road_type, DEFAULT_DRIVING_SPEED)
            data['speed_kph'] = speed_mph * 1.60934  # Convert mph to km/h
    
    return G


def calculate_travel_time_with_speed(G):
    """
    Calculate travel time for each edge based on length and speed
    Adds 'travel_time' attribute (in minutes) to edges
    """
    for u, v, data in G.edges(data=True):
        length_km = data['length'] / 1000  # Convert meters to km
        speed_kph = data.get('speed_kph', 5.0)  # Default to walking speed
        travel_time_hours = length_km / speed_kph
        data['travel_time'] = travel_time_hours * 60  # Convert to minutes
    
    return G


@st.cache_data
def get_network_graph(place_name, network_type='drive'):
    """
    Download and cache the road network for the area
    network_type: 'drive' or 'walk'
    """
    try:
        G = ox.graph_from_place(place_name, network_type=network_type)
        return G
    except Exception as e:
        st.warning(f"Could not download network for {place_name}: {e}")
        return None


def generate_service_area(G, origin_point, max_time_minutes, network_type='drive'):
    """
    Generate actual service area (isochrone) polygon for a facility
    
    Args:
        G: Network graph with speed attributes
        origin_point: (lat, lon) tuple
        max_time_minutes: Maximum travel time
        network_type: 'drive' or 'walk'
    
    Returns: Shapely Polygon representing the service area
    """
    if G is None:
        return None
    
    try:
        # Add speed and travel time to network
        G = add_speed_to_network(G, network_type)
        G = calculate_travel_time_with_speed(G)
        
        # Find nearest node to origin
        origin_node = ox.nearest_nodes(G, origin_point[1], origin_point[0])
        
        # Get all nodes reachable within time threshold
        try:
            # Use travel_time as weight
            subgraph = nx.ego_graph(G, origin_node, radius=max_time_minutes, 
                                   distance='travel_time')
            reachable_nodes = list(subgraph.nodes())
        except:
            return None
        
        if len(reachable_nodes) < 3:
            return None
        
        # Get coordinates of reachable nodes
        node_points = []
        for node in reachable_nodes:
            node_data = G.nodes[node]
            node_points.append(Point(node_data['x'], node_data['y']))
        
        # Create ONE continuous service area polygon per facility
        if len(node_points) > 0:
            gdf_points = gpd.GeoDataFrame(geometry=node_points, crs='EPSG:4326')
            
            # Buffer each point and create union
            buffered = gdf_points.buffer(0.003)  # ~300m buffer
            service_area = buffered.union_all()
            
            # If result is MultiPolygon (disconnected), take convex hull to make one polygon
            if service_area.geom_type == 'MultiPolygon':
                service_area = service_area.convex_hull
            
            # Simplify to reduce complexity
            service_area = service_area.simplify(0.001, preserve_topology=True)
            return service_area
        
        return None
        
    except Exception as e:
        st.warning(f"Service area generation error: {e}")
        return None


def calculate_network_travel_time(G, origin, destinations_df, max_time_minutes, network_type='drive'):
    """
    Calculate travel times from origin to all destinations using network analysis
    with road-type based speed profiles
    
    Returns: array of travel times (same length as destinations_df, NaN if unreachable)
    """
    if G is None:
        return np.full(len(destinations_df), np.nan)
    
    try:
        # Add speed and travel time attributes
        G = add_speed_to_network(G, network_type)
        G = calculate_travel_time_with_speed(G)
        
        # Find nearest network node to origin
        origin_node = ox.nearest_nodes(G, origin[1], origin[0])
        
        # Get all shortest path lengths from origin using travel_time as weight
        try:
            lengths = nx.single_source_dijkstra_path_length(
                G, origin_node, cutoff=max_time_minutes, weight='travel_time'
            )
        except:
            return np.full(len(destinations_df), np.nan)
        
        # For each destination, find nearest node and get travel time
        travel_times = []
        for idx, dest in destinations_df.iterrows():
            try:
                dest_node = ox.nearest_nodes(G, dest['longitude'], dest['latitude'])
                if dest_node in lengths:
                    time_minutes = lengths[dest_node]
                    travel_times.append(time_minutes)
                else:
                    travel_times.append(np.nan)
            except:
                travel_times.append(np.nan)
        
        return np.array(travel_times)
    
    except Exception as e:
        st.warning(f"Network analysis error: {e}")
        return np.full(len(destinations_df), np.nan)




def calculate_network_travel_time_preprocessed(G, origin, destinations_df, max_time_minutes):
    """
    Calculate travel times using ALREADY PREPROCESSED graph (speeds already added)
    This avoids redundant edge processing - MUCH FASTER
    """
    if G is None:
        return np.full(len(destinations_df), np.nan)
    
    try:
        # Find nearest network node to origin
        origin_node = ox.nearest_nodes(G, origin[1], origin[0])
        
        # Get all shortest path lengths using travel_time weight (already calculated)
        try:
            lengths = nx.single_source_dijkstra_path_length(
                G, origin_node, cutoff=max_time_minutes, weight='travel_time'
            )
        except:
            return np.full(len(destinations_df), np.nan)
        
        # For each destination, find nearest node and get travel time
        travel_times = []
        for idx, dest in destinations_df.iterrows():
            try:
                dest_node = ox.nearest_nodes(G, dest['longitude'], dest['latitude'])
                if dest_node in lengths:
                    travel_times.append(lengths[dest_node])
                else:
                    travel_times.append(np.nan)
            except:
                travel_times.append(np.nan)
        
        return np.array(travel_times)
    
    except Exception as e:
        return np.full(len(destinations_df), np.nan)


def generate_service_area_preprocessed(G, origin_point, max_time_minutes):
    """
    Generate service area using ALREADY PREPROCESSED graph - MUCH FASTER
    """
    if G is None:
        return None
    
    try:
        origin_node = ox.nearest_nodes(G, origin_point[1], origin_point[0])
        
        subgraph = nx.ego_graph(G, origin_node, radius=max_time_minutes, 
                               distance='travel_time')
        
        reachable_nodes = list(subgraph.nodes())
        if len(reachable_nodes) < 3:
            return None
        
        node_coords = [(G.nodes[node]['y'], G.nodes[node]['x']) for node in reachable_nodes]
        
        from shapely.geometry import Point, MultiPoint
        points = MultiPoint([Point(lon, lat) for lat, lon in node_coords])
        buffered = points.buffer(0.002)
        service_area = buffered.union_all()
        
        if service_area.geom_type == 'MultiPolygon':
            service_area = service_area.convex_hull
        
        service_area = service_area.simplify(0.001, preserve_topology=True)
        return service_area
        
    except:
        return None


def calculate_manhattan_distance_time(origin_lat, origin_lon, dest_lat, dest_lon, mode='drive'):
    """
    Calculate Manhattan distance-based travel time
    
    Args:
        mode: 'drive' (30 mph avg) or 'walk' (5 km/h)
    
    Returns: travel time in minutes
    """
    # Convert lat/lon differences to approximate miles
    lat_diff = abs(dest_lat - origin_lat) * 69  # 1 degree lat ≈ 69 miles
    lon_diff = abs(dest_lon - origin_lon) * 69 * np.cos(np.radians(origin_lat))
    
    manhattan_miles = lat_diff + lon_diff
    
    # Calculate time based on mode
    if mode == 'drive':
        speed_mph = DEFAULT_DRIVING_SPEED
        travel_time_minutes = (manhattan_miles / speed_mph) * 60
    else:  # walk
        speed_kmh = WALKING_SPEED_KMH
        manhattan_km = manhattan_miles * 1.60934
        travel_time_minutes = (manhattan_km / speed_kmh) * 60
    
    return travel_time_minutes


def build_coverage_matrix(candidates_subset, demand_subset, max_time, network_type='drive', 
                          use_network=False, G=None):
    """
    Build binary coverage matrix with optional service area generation
    OPTIMIZED: Pre-processes network ONCE before loop (10-50x faster!)
    """
    n_facilities = len(candidates_subset)
    n_demand = len(demand_subset)
    
    coverage = np.zeros((n_facilities, n_demand), dtype=int)
    service_areas = []
    
    candidates_reset = candidates_subset.reset_index(drop=True)
    demand_reset = demand_subset.reset_index(drop=True)
    
    # *** KEY OPTIMIZATION: Pre-process network ONCE before loop ***
    # This avoids redundantly adding speeds to every edge N times (once per facility)
    if use_network and G is not None:
        G = add_speed_to_network(G, network_type)
        G = calculate_travel_time_with_speed(G)
    
    for i, facility in candidates_reset.iterrows():
        facility_point = (facility['latitude'], facility['longitude'])
        
        if use_network and G is not None:
            # Use preprocessed versions (no redundant edge processing)
            service_area = generate_service_area_preprocessed(G, facility_point, max_time)
            service_areas.append({
                'facility_idx': i,
                'geometry': service_area,
                'name': facility['name']
            })
            
            travel_times = calculate_network_travel_time_preprocessed(
                G, facility_point, demand_reset, max_time
            )
            
            for j in range(len(travel_times)):
                if not np.isnan(travel_times[j]) and travel_times[j] <= max_time:
                    coverage[i][j] = 1
        else:
            service_areas.append(None)
            for j, demand_pt in demand_reset.iterrows():
                travel_time = calculate_manhattan_distance_time(
                    facility['latitude'], facility['longitude'],
                    demand_pt['latitude'], demand_pt['longitude'],
                    mode=network_type
                )
                if travel_time <= max_time:
                    coverage[i][j] = 1
    
    return coverage, candidates_reset, demand_reset, service_areas


def solve_maxcover(coverage_matrix, demand_weights, num_facilities):
    """
    Solve Maximum Coverage Location Problem using Integer Linear Programming
    """
    n_facilities, n_demand = coverage_matrix.shape
    
    # Create the model
    model = LpProblem("Max_Coverage", LpMaximize)
    
    # Decision variables
    x = LpVariable.dicts("facility", range(n_facilities), cat='Binary')
    y = LpVariable.dicts("covered", range(n_demand), cat='Binary')
    
    # Objective: maximize covered demand
    model += lpSum([demand_weights[j] * y[j] for j in range(n_demand)])
    
    # Constraint: select exactly num_facilities
    model += lpSum([x[i] for i in range(n_facilities)]) == num_facilities
    
    # Constraint: demand covered only if covering facility selected
    for j in range(n_demand):
        covering_facilities = [i for i in range(n_facilities) if coverage_matrix[i][j] == 1]
        if covering_facilities:
            model += y[j] <= lpSum([x[i] for i in covering_facilities])
    
    # Solve
    model.solve(PULP_CBC_CMD(msg=0))
    
    # Extract results
    selected = [i for i in range(n_facilities) if x[i].varValue and x[i].varValue == 1]
    
    covered_demand = 0
    for j in range(n_demand):
        if y[j].varValue is not None and y[j].varValue > 0:
            covered_demand += demand_weights[j] * y[j].varValue
    
    return selected, covered_demand


def create_map(zip_gdf, selected_zip, candidates_df, demand_df, selected_facilities=None, 
               coverage_matrix=None, demand_reset=None, candidates_reset=None, service_areas=None):
    """
    Create interactive map with optional service area polygons
    """
    # Get the selected ZIP boundary
    zip_boundary = zip_gdf[zip_gdf['ZIP_CODE'] == selected_zip].iloc[0]

    bounds = zip_boundary.geometry.bounds
    center_lat = float((bounds[1] + bounds[3]) / 2)
    center_lon = float((bounds[0] + bounds[2]) / 2)

    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=10,
        tiles='OpenStreetMap',
        prefer_canvas=True
    )

    # Add ZIP boundary
    style_dict = {
        'fillColor': '#E6F2FF',  # Very light blue fill
        'color': '#1E90FF',      # DodgerBlue border (lighter than dark blue)
        'weight': 4,             # Thick border
        'fillOpacity': 0.1,      # Transparent fill
        'opacity': 0.9           # Slightly transparent border
    }
    
    folium.GeoJson(
        zip_boundary.geometry.__geo_interface__,
        style_function=lambda x: style_dict
    ).add_to(m)

    # Color scheme for facility types
    type_colors = {
        'Churches': 'purple',
        'Community Health Clinics': 'red', 
        'Food Banks': 'green',
        'Homeless Services': 'blue',
        'Hospitals': 'darkred',
        'Rural Primary Care': 'pink'
    }

    # Determine covered demand points
    covered_demand_indices = set()
    if selected_facilities is not None and coverage_matrix is not None and demand_reset is not None:
        selected_facility_indices = list(selected_facilities.index)
        
        for j in range(len(demand_reset)):
            for i in selected_facility_indices:
                if coverage_matrix[i, j] == 1:
                    covered_demand_indices.add(j)
                    break

    candidates_in_zip = candidates_df[candidates_df['zip_code'] == selected_zip]
    analysis_complete = selected_facilities is not None and len(selected_facilities) > 0

    # Add service area polygons if available - all same green color
    if analysis_complete and service_areas is not None and candidates_reset is not None:
        # Map selected facility original indices to their positions in candidates_reset
        selected_positions = []
        for sel_idx in selected_facilities.index:
            for i, cand_idx in enumerate(candidates_reset.index):
                if sel_idx == cand_idx:
                    selected_positions.append(i)
                    break
        
        for service_area_data in service_areas:
            if service_area_data is not None:
                facility_idx = service_area_data.get('facility_idx')
                if facility_idx is not None and facility_idx in selected_positions:
                    geom = service_area_data.get('geometry')
                    if geom is not None:
                        try:
                            folium.GeoJson(
                                geom.__geo_interface__,
                                style_function=lambda x: {
                                    'fillColor': '#90EE90',    # Light green
                                    'color': '#228B22',        # Forest green border
                                    'weight': 2,
                                    'fillOpacity': 0.3,
                                    'opacity': 0.8
                                },
                                tooltip=f"Service Area: {service_area_data.get('name', 'Unknown')}"
                            ).add_to(m)
                        except Exception as e:
                            pass  # Skip if geometry is invalid
        
        # Debug: Show service area info
        if len(selected_positions) > 0:
            valid_service_areas = sum(1 for sa in service_areas if sa is not None and sa.get('geometry') is not None)
            st.sidebar.info(f"Service areas: {valid_service_areas} generated, {len(selected_positions)} facilities selected")

    # Add facilities
    if analysis_complete:
        for idx, facility in candidates_in_zip.iterrows():
            is_selected = False
            for sel_idx in selected_facilities.index:
                if (abs(facility['latitude'] - selected_facilities.loc[sel_idx, 'latitude']) < 0.0001 and
                    abs(facility['longitude'] - selected_facilities.loc[sel_idx, 'longitude']) < 0.0001):
                    is_selected = True
                    break
            
            lat = float(facility['latitude'])
            lon = float(facility['longitude'])
            name = str(facility['name'])
            ftype = str(facility['type'])
            
            if is_selected:
                folium.Marker(
                    location=[lat, lon],
                    popup=f"<b>Proposed Site</b><br><b>{name}</b><br>{ftype}",
                    icon=folium.Icon(color='green', icon='star', prefix='fa')
                ).add_to(m)
            else:
                folium.CircleMarker(
                    location=[lat, lon],
                    radius=7,
                    popup=f"<b>{name}</b><br>{ftype}",
                    color='#4169E1',
                    fill=True,
                    fillColor='#6495ED',
                    fillOpacity=0.6,
                    weight=2
                ).add_to(m)
    else:
        for idx, facility in candidates_in_zip.iterrows():
            color = type_colors.get(str(facility['type']), 'gray')
            folium.CircleMarker(
                location=[float(facility['latitude']), float(facility['longitude'])],
                radius=5,
                popup=f"<b>{facility['name']}</b><br>{facility['type']}",
                color=color,
                fill=True,
                fillColor=color,
                fillOpacity=0.6,
                weight=2
            ).add_to(m)

    # Add demand points
    demand_in_zip = demand_df[demand_df['zip_code'] == selected_zip]

    for idx, demand in demand_in_zip.iterrows():
        lat = float(demand['latitude'])
        lon = float(demand['longitude'])
        uninsured = int(demand['uninsured_pop'])
        
        if analysis_complete and demand_reset is not None:
            demand_reset_index = None
            for reset_idx in range(len(demand_reset)):
                if (abs(demand_reset.iloc[reset_idx]['latitude'] - demand['latitude']) < 0.0001 and
                    abs(demand_reset.iloc[reset_idx]['longitude'] - demand['longitude']) < 0.0001):
                    demand_reset_index = reset_idx
                    break

            is_covered = demand_reset_index in covered_demand_indices

            if is_covered:
                radius = 3
                color = '#90EE90'
                fill_color = '#98FB98'
                popup_text = f"<b>Covered Demand Point</b><br>Uninsured: {uninsured}"
            else:
                radius = 3
                color = 'darkorange'
                fill_color = 'orange'
                popup_text = f"<b>Uncovered Demand Point</b><br>Uninsured: {uninsured}"
        else:
            radius = 3
            color = 'darkorange'
            fill_color = 'orange'
            popup_text = f"<b>Block Group Centroid</b><br>Uninsured: {uninsured}"

        folium.CircleMarker(
            location=[lat, lon],
            radius=radius,
            popup=popup_text,
            color=color,
            fill=True,
            fillColor=fill_color,
            fillOpacity=0.7,
            weight=1
        ).add_to(m)

    # Dynamic legend
    if analysis_complete:
        legend_html = '''
        <div style="position: fixed; 
                    bottom: 50px; right: 50px; width: 250px; height: auto; 
                    background-color: white; z-index:9999; font-size:14px;
                    border:2px solid grey; border-radius: 5px; padding: 10px">
        <p style="margin:0; font-weight:bold; text-align:center;">Legend</p>
        <hr style="margin:5px 0;">
        <p style="margin:3px 0; font-weight:bold;">Facilities:</p>
        <p style="margin:3px 0; margin-left:10px;"><i class="fa fa-star" style="color:green"></i> Proposed Site/s</p>
        <p style="margin:3px 0; margin-left:10px;"><span style="color:#4169E1; font-size:18px;">●</span> Other Sites</p>
        <hr style="margin:5px 0;">
        <p style="margin:3px 0; font-weight:bold;">Service Areas:</p>
        <p style="margin:3px 0; margin-left:10px;"><span style="color:#90EE90; font-size:18px;">▬</span> Reachable Area</p>
        <hr style="margin:5px 0;">
        <p style="margin:3px 0; font-weight:bold;">Demand Points:</p>
        <p style="margin:3px 0; margin-left:10px;"><span style="color:#90EE90; font-size:14px;">●</span> Covered</p>
        <p style="margin:3px 0; margin-left:10px;"><span style="color:darkorange; font-size:14px;">●</span> Uncovered</p>
        </div>
        '''
    else:
        legend_html = '''
        <div style="position: fixed; 
                    bottom: 50px; right: 50px; width: 240px; height: auto; 
                    background-color: white; z-index:9999; font-size:14px;
                    border:2px solid grey; border-radius: 5px; padding: 10px">
        <p style="margin:0; font-weight:bold; text-align:center;">Legend</p>
        <hr style="margin:5px 0;">
        <p style="margin:3px 0; font-weight:bold;">Facility Types:</p>
        <p style="margin:3px 0; margin-left:10px;"><span style="color:purple; font-size:20px;">●</span> Churches</p>
        <p style="margin:3px 0; margin-left:10px;"><span style="color:red; font-size:20px;">●</span> Community Health</p>
        <p style="margin:3px 0; margin-left:10px;"><span style="color:green; font-size:20px;">●</span> Food Banks</p>
        <p style="margin:3px 0; margin-left:10px;"><span style="color:blue; font-size:20px;">●</span> Homeless Services</p>
        <p style="margin:3px 0; margin-left:10px;"><span style="color:darkred; font-size:20px;">●</span> Hospitals</p>
        <p style="margin:3px 0; margin-left:10px;"><span style="color:pink; font-size:20px;">●</span> Rural Primary Care</p>
        <hr style="margin:5px 0;">
        <p style="margin:3px 0; font-weight:bold;">Demand Points:</p>
        <p style="margin:3px 0; margin-left:10px;"><span style="color:darkorange; font-size:14px;">●</span> Block Group Centroid</p>
        </div>
        '''
    
    m.get_root().html.add_child(folium.Element(legend_html))

    padding = 0.03
    m.fit_bounds([
        [float(bounds[1]) - padding, float(bounds[0]) - padding], 
        [float(bounds[3]) + padding, float(bounds[2]) + padding]
    ])

    return m


def main():
    st.title("🗺️ South Carolina Facility Location Analysis Tool")
    st.markdown("""
    This tool helps identify optimal facility locations to maximize coverage of uninsured populations 
    using Maximum Coverage Location-Allocation analysis.
    """)
    
    # Add methodology expander
    with st.expander("📖 Methodology Documentation"):
        st.markdown("""
        ### Location-Allocation Model
        
        **Model Type:** Maximum Coverage Location Problem (MCLP)
        
        ### Travel Time Calculation
        
        #### Network Analysis (When Enabled)
        - **Data Source:** OpenStreetMap via OSMnx
        - **Algorithm:** Dijkstra's shortest path
        - **Speed Profiles:**
          - **Driving:** Road-type based (highways 65mph, residential 25mph, etc.)
          - **Walking:** 5 km/h (3.1 mph)
        - **Service Areas:** Actual isochrone polygons (all points reachable within time threshold)
        
        #### Manhattan Distance (Default)
        - **Method:** Rectilinear distance approximation
        - **Speed:** 30 mph driving average, 5 km/h walking
        
        **Note on Historical Traffic Data:**
        OpenStreetMap does not include historical traffic speeds or real-time data. For ArcGIS-style 
        historical speeds, you would need commercial data (Esri StreetMap Premium, HERE, Google Maps API).
        This tool uses road classification-based speed estimates as the best available free alternative.
        """)
    
    # Load data
    try:
        with st.spinner("Loading data..."):
            zip_gdf, candidates_df, demand_df = load_data(JSON_PATH)
        st.success("✅ Data loaded successfully!")
    except Exception as e:
        st.error(f"❌ Error loading data: {e}")
        st.info(f"Please ensure your data file exists at: {JSON_PATH}")
        st.stop()
    
    # Sidebar controls
    st.sidebar.header("Analysis Parameters")
    
    # Get current parameters
    zip_options = sorted(zip_gdf['ZIP_CODE'].unique())
    zip_labels = [f"{zip_code} ({zip_gdf[zip_gdf['ZIP_CODE']==zip_code].iloc[0]['po_name']})" 
                  for zip_code in zip_options]
    zip_dict = dict(zip(zip_labels, zip_options))
    
    selected_zip_label = st.sidebar.selectbox(
        "Select ZIP Code",
        options=zip_labels,
        index=0
    )
    selected_zip = zip_dict[selected_zip_label]
    
    facility_types = sorted(candidates_df['type'].unique())
    selected_types = st.sidebar.multiselect(
        "Select Facility Types",
        options=facility_types,
        default=facility_types
    )
    
    travel_mode = st.sidebar.radio(
        "Travel Mode",
        options=['drive', 'walk'],
        format_func=lambda x: 'Driving' if x == 'drive' else 'Walking'
    )
    
    time_threshold = st.sidebar.selectbox(
        "Maximum Travel Time (minutes)",
        options=[5, 10, 15, 20, 30],
        index=1
    )
    
    candidates_in_zip = candidates_df[
        (candidates_df['zip_code'] == selected_zip) & 
        (candidates_df['type'].isin(selected_types))
    ]
    max_facilities = len(candidates_in_zip)
    
    num_facilities = st.sidebar.slider(
        "Number of Facilities to Select",
        min_value=1,
        max_value=max(1, max_facilities),
        value=min(3, max_facilities),
        disabled=(max_facilities == 0)
    )
    
    use_network = st.sidebar.checkbox(
        "Use Road Network Analysis (slower but more accurate)",
        value=False,
        help="Generates actual service area polygons and uses road-based routing"
    )
    
    # Check if parameters changed
    current_params = {
        'zip': selected_zip,
        'types': tuple(selected_types),
        'mode': travel_mode,
        'time': time_threshold,
        'num': num_facilities,
        'network': use_network
    }
    
    params_changed = (st.session_state.last_params != current_params)
    
    # Run Analysis Button
    run_analysis = st.sidebar.button("🔍 Run Analysis", type="primary", use_container_width=True)
    
    # Reset analysis if parameters changed
    if params_changed and not run_analysis:
        st.session_state.analysis_complete = False
        st.session_state.selected_facilities = None
    
    # Main content area
    col1, col2 = st.columns([3, 1])
    
    with col2:
        st.subheader("📊 Summary Statistics")
        
        demand_in_zip = demand_df[demand_df['zip_code'] == selected_zip]
        total_uninsured = demand_in_zip['uninsured_pop'].sum()
        
        st.metric("Total Uninsured Population", f"{int(total_uninsured):,}")
        st.metric("Available Candidate Sites", len(candidates_in_zip))
        st.metric("Demand Points", len(demand_in_zip))
    
    with col1:
        st.subheader("🗺️ Interactive Map")
        
        if max_facilities == 0:
            st.warning("No candidate facilities available for the selected ZIP code and facility types.")
            m = create_map(zip_gdf, selected_zip, candidates_df, demand_df)
            try:
                st_folium(m, width=900, height=750, returned_objects=[])
            except Exception as e:
                st.components.v1.html(m._repr_html_(), height=750, scrolling=True)
                
        elif run_analysis or st.session_state.analysis_complete:
            # Run analysis if button clicked OR display cached results
            if run_analysis:
                with st.spinner("Running optimization analysis..."):
                    
                    G = None
                    method_used = "Manhattan Distance"
                    
                    if use_network:
                        zip_geom = zip_gdf[zip_gdf['ZIP_CODE'] == selected_zip].iloc[0].geometry
                        zip_center = zip_geom.centroid
                        
                        try:
                            with st.spinner("Downloading road network (1-2 minutes)..."):
                                network_type = 'drive' if travel_mode == 'drive' else 'walk'
                                G = ox.graph_from_point(
                                    (zip_center.y, zip_center.x),
                                    dist=15000,  # 15km radius for larger service areas
                                    network_type=network_type
                                )
                                method_used = "Road Network Analysis (OSM with road-type speeds)"
                                st.success(f"✓ Using {method_used}")
                        except Exception as e:
                            st.warning(f"Network download failed: {e}. Using Manhattan distance.")
                            G = None
                    else:
                        speed_info = "30 mph avg" if travel_mode=='drive' else "5 km/h"
                        st.info(f"ℹ️ Using {method_used} approximation ({speed_info})")
                    
                    # Build coverage matrix with service areas
                    coverage_matrix, candidates_reset, demand_reset, service_areas = build_coverage_matrix(
                        candidates_in_zip,
                        demand_in_zip,
                        time_threshold,
                        network_type=travel_mode,
                        use_network=use_network,
                        G=G
                    )
                    
                    total_possible_coverage = np.sum(coverage_matrix)
                    st.info(f"Coverage matrix: {coverage_matrix.shape[0]} facilities × {coverage_matrix.shape[1]} demand points. Coverage links: {total_possible_coverage}")
                    
                    demand_weights = demand_reset['uninsured_pop'].values
                    
                    selected_indices, covered_pop = solve_maxcover(
                        coverage_matrix,
                        demand_weights,
                        num_facilities
                    )
                    
                    selected_facilities = candidates_reset.iloc[selected_indices]
                    
                    # Store in session state
                    st.session_state.analysis_complete = True
                    st.session_state.selected_facilities = selected_facilities
                    st.session_state.coverage_matrix = coverage_matrix
                    st.session_state.demand_reset = demand_reset
                    st.session_state.candidates_reset = candidates_reset
                    st.session_state.covered_pop = covered_pop
                    st.session_state.service_areas = service_areas
                    st.session_state.last_params = current_params
            
            # Display results (from session state)
            selected_facilities = st.session_state.selected_facilities
            coverage_matrix = st.session_state.coverage_matrix
            demand_reset = st.session_state.demand_reset
            candidates_reset = st.session_state.candidates_reset
            covered_pop = st.session_state.covered_pop
            service_areas = st.session_state.service_areas
            
            coverage_pct = (covered_pop / total_uninsured * 100) if total_uninsured > 0 else 0
            
            covered_demand_count = 0
            selected_indices = list(selected_facilities.index)
            for j in range(coverage_matrix.shape[1]):
                for i in selected_indices:
                    if coverage_matrix[i, j] == 1:
                        covered_demand_count += 1
                        break
            
            with col2:
                st.metric("Covered Uninsured Population", f"{int(covered_pop):,}")
                st.metric("Coverage Percentage", f"{coverage_pct:.1f}%")
                st.metric("Covered Demand Points", f"{covered_demand_count} / {len(demand_reset)}")
                
                # Calculate individual coverage for ranking (for table below map)
                facility_coverage = []
                for i, (idx, facility) in enumerate(selected_facilities.iterrows()):
                    # Find position in candidates_reset
                    position = None
                    for j, cand_idx in enumerate(candidates_reset.index):
                        if idx == cand_idx:
                            position = j
                            break
                    
                    # Count how many demand points this facility covers
                    if position is not None and position < coverage_matrix.shape[0]:
                        covered_pop = 0
                        for k in range(coverage_matrix.shape[1]):
                            if coverage_matrix[position, k] == 1:
                                covered_pop += demand_reset.iloc[k]['uninsured_pop']
                        facility_coverage.append(covered_pop)
                    else:
                        facility_coverage.append(0)
                
                # Create ranking (1 = highest coverage)
                sorted_indices = sorted(range(len(facility_coverage)), key=lambda i: facility_coverage[i], reverse=True)
                ranks = [0] * len(facility_coverage)
                for rank, idx in enumerate(sorted_indices, 1):
                    ranks[idx] = rank
                
                # Create table data (will be displayed below map only)
                table_data = []
                for i, (idx, facility) in enumerate(selected_facilities.iterrows()):
                    table_data.append({
                        'Rank': ranks[i],
                        'Name': facility['name'],
                        'Type': facility['type'],
                        'Address': facility.get('address', 'N/A'),
                        'Individual Coverage': int(facility_coverage[i])
                    })
                
                # Create DataFrame sorted by rank (for display below map)
                df_display = pd.DataFrame(table_data)
                df_display = df_display.sort_values('Rank').reset_index(drop=True)
            
            # Create map with service areas
            m = create_map(zip_gdf, selected_zip, candidates_df, demand_df, 
                         selected_facilities, coverage_matrix, demand_reset, 
                         candidates_reset, service_areas)
            
            with col1:
                try:
                    st_folium(m, width=900, height=750, returned_objects=[])
                except Exception as e:
                    st.components.v1.html(m._repr_html_(), height=750, scrolling=True)
                
                # Display table below map
                st.markdown("---")
                st.subheader("📍 Proposed Sites")
                st.dataframe(df_display, use_container_width=True, hide_index=True)
            
            # Export options
            st.subheader("📥 Export Results")
            col_exp1, col_exp2 = st.columns(2)
            
            with col_exp1:
                export_cols = ['facility_id', 'name', 'type', 'address', 'latitude', 'longitude']
                available_cols = [col for col in export_cols if col in selected_facilities.columns]
                csv_data = selected_facilities[available_cols].to_csv(index=False)
                st.download_button(
                    label="Download Proposed Sites (CSV)",
                    data=csv_data,
                    file_name=f"proposed_sites_{selected_zip}.csv",
                    mime="text/csv",
                    key="csv_download"  # Added key to prevent reset
                )
            
            with col_exp2:
                gdf_selected = gpd.GeoDataFrame(
                    selected_facilities,
                    geometry=gpd.points_from_xy(selected_facilities['longitude'], selected_facilities['latitude']),
                    crs='EPSG:4326'
                )
                geojson_data = gdf_selected.to_json()
                st.download_button(
                    label="Download Proposed Sites (GeoJSON)",
                    data=geojson_data,
                    file_name=f"proposed_sites_{selected_zip}.geojson",
                    mime="application/geo+json",
                    key="geojson_download"  # Added key to prevent reset
                )
        
        else:
            # Show initial map
            m = create_map(zip_gdf, selected_zip, candidates_df, demand_df)
            try:
                st_folium(m, width=900, height=750, returned_objects=[])
            except Exception as e:
                st.components.v1.html(m._repr_html_(), height=750, scrolling=True)
            st.info("👈 Configure parameters in the sidebar and click 'Run Analysis' to optimize facility locations.")

if __name__ == "__main__":
    main()