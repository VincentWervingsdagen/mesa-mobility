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
    model_params = {
        "data_crs": "epsg:4326",
        "start_date": '2023-05-01',
        "bounding_box":(4.3338,51.9853,4.3658,52.0204), #Delft
        #"bounding_box": (4.1874, 51.8280, 4.593, 52.0890), #Zuid holland
        "num_commuters": mesa.visualization.Slider(
            "Number of Commuters", value=1, min_value=1, max_value=10, step=1
        ),
        "walking_allowed": mesa.visualization.Choice(
            "Can paths or sidewalks be used?",
            value=True,
            choices=[False,True]
        ),
        "step_duration": mesa.visualization.NumberInput(
            "Step Duration (seconds)",
            value=10,
        ),
        "alpha": mesa.visualization.NumberInput(
            "Exponent jump size distribution (truncated power law)",
            value=0.55,
        ),
        "tau_jump_min": mesa.visualization.NumberInput(
            "Min jump (km) jump size distribution (truncated power law)",
            value=0.1,
        ),
        "tau_jump": mesa.visualization.NumberInput(
            "Max jump (km) jump size distribution (truncated power law)",
            value=0.2,
        ),
        "beta": mesa.visualization.NumberInput(
            "Exponent waiting time distribution (truncated power law)",
            value=0.8,
        ),
        "tau_time_min": mesa.visualization.NumberInput(
            "Min time (hour) waiting time distribution (truncated power law)",
            value=0.01,
        ),
        "tau_time": mesa.visualization.NumberInput(
            "Max time (hour) waiting time distribution (truncated power law)",
            value=0.02,
        ),
        "rho": mesa.visualization.NumberInput(
            "Constant in probability of exploration",
            value=2,
        ),
        "gamma": mesa.visualization.NumberInput(
            "Exponent in probability of exploration",
            value=1,
        ),
        "buildings_file": os.path.join(script_dir, '..', 'data', 'zuid-holland', 'gis_osm_buildings_a_free_1.zip'),
        "walkway_file": os.path.join(script_dir, '..', 'data', 'zuid-holland', 'gis_osm_roads_free_1.zip'),
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

