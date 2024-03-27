import mesa
import mesa_geo as mg
from src.model.model import AgentsAndNetworks
from src.visualization.server import (
    agent_draw,
    clock_element,
    status_chart,
    location_chart,
)
import os

script_dir = os.path.dirname(os.path.abspath(__file__))

if __name__ == "__main__":
    region_params = {
        "data_crs": "epsg:4326", "commuter_speed": 5.0
    }

    model_params = {
        "data_crs": region_params["data_crs"],
        "start_date": '2023-05-01',
        "bounding_box":(4.3338,51.9853,4.3658,52.0204),
        "bounding_box_trip":(4.2929,52.0597,4.3157,52.0871),
        "commuter_speed_drive": 5.0,
        "num_commuters": mesa.visualization.Slider(
            "Number of Commuters", value=2, min_value=1, max_value=30, step=1
        ),
        "commuter_speed_walk": mesa.visualization.Slider(
            "Commuter Walking Speed (m/s)",
            value=region_params["commuter_speed"],
            min_value=0.1,
            max_value=30,
            step=0.1,
        ),
        "step_duration": mesa.visualization.NumberInput(
            "Step Duration (seconds)",
            value=60,
        ),
        "allow_trips": mesa.visualization.Checkbox(
            "Allow Trips",
            value=False,
        ),
        "only_same_day_trips": mesa.visualization.Checkbox(
            "Trips Occur on the Same Day",
            value=False,
        ),
        "alpha": mesa.visualization.NumberInput(
            "Exponent jump size distribution (truncated power law)",
            value=0.55,
        ),
        "tau_jump_min": mesa.visualization.NumberInput(
            "Min jump (km) jump size distribution (truncated power law)",
            value=1.0,
        ),
        "tau_jump": mesa.visualization.NumberInput(
            "Max jump (km) jump size distribution (truncated power law)",
            value=100.0,
        ),
        "beta": mesa.visualization.NumberInput(
            "Exponent waiting time distribution (truncated power law)",
            value=0.8,
        ),
        "tau_time_min": mesa.visualization.NumberInput(
            "Min time (hour) waiting time distribution (truncated power law)",
            value=20/60,
        ),
        "tau_time": mesa.visualization.NumberInput(
            "Max time (hour) waiting time distribution (truncated power law)",
            value=17,
        ),
        "rho": mesa.visualization.NumberInput(
            "Constant in probability of exploration",
            value=1,
        ),
        "gamma": mesa.visualization.NumberInput(
            "Exponent in probability of exploration",
            value=2,
        ),
        "buildings_file": os.path.join(script_dir, '..', 'data', 'zuid-holland', 'gis_osm_buildings_a_free_1.zip'),
        "buildings_file_trip": os.path.join(script_dir, '..', 'data', 'noord-holland', 'gis_osm_buildings_a_free_1.zip'),
        "walkway_file": os.path.join(script_dir, '..', 'data', 'zuid-holland', 'gis_osm_roads_free_1.zip'),
        "walkway_file_trip": os.path.join(script_dir, '..', 'data', 'noord-holland', 'gis_osm_roads_free_1.zip'),
    }

    map_element = mg.visualization.MapModule(agent_draw, map_height=600, map_width=600)
    server = mesa.visualization.ModularServer(
        AgentsAndNetworks,
        # use following if you want map functionality
        [map_element, clock_element],
        #[clock_element],
        "Agents and Networks",
        model_params,
    )
    server.launch()

