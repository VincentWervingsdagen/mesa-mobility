import pandas as pd
from pyproj import Transformer
import os
import construct_markov_chains as MC
import numpy as np
import pgeocode

script_dir = os.path.dirname(os.path.abspath(__file__))

cell_file=os.path.join(script_dir, '..','..', 'data', '20191202131001.csv')
bounding_box = (4.1874, 51.8280, 4.593, 52.0890)
antenna_type='LTE'
level = 'postal2'


def state_space_Omega(cell_file,bounding_box,antenna_type,level='postal3'):
    df_cell = pd.read_csv(cell_file)
    # drop useless columns
    df_cell = df_cell.drop(
        ['Samenvatting', 'Vermogen', 'Frequentie', 'Veilige afstand', 'id', 'objectid', 'WOONPLAATSNAAM',
         'DATUM_PLAATSING',
         'DATUM_INGEBRUIKNAME', 'GEMNAAM', 'Hoogte', 'Hoofdstraalrichting', 'sat_code'], axis=1)
    # drop types except the antenna_type
    df_cell = df_cell.loc[df_cell['HOOFDSOORT'] == antenna_type]
    # Transform to wgs84
    df_cell['lat'], df_cell['lon'] = Transformer.from_crs("EPSG:28992", "EPSG:4979").transform(df_cell['X'],
                                                                                               df_cell['Y'])
    # Only keep cell towers in bounding box
    df_cell = df_cell.loc[
        (df_cell['lon'] >= bounding_box[0]) & (df_cell['lon'] <= bounding_box[2])
        & (df_cell['lat'] >= bounding_box[1]) & (df_cell['lat'] <= bounding_box[3])]

    if level == 'antenna':
        return np.sort(np.unique(df_cell['POSTCODE'].dropna()))
    elif level == 'postal':
        df_cell = df_cell['POSTCODE'].dropna()
        array_zip_codes = [element[0:4] for element in df_cell]
        return np.sort(np.unique(array_zip_codes))
    elif level == 'postal3':
        df_cell = df_cell['POSTCODE'].dropna()
        array_zip_codes = [element[0:3] for element in df_cell]
        return np.sort(np.unique(array_zip_codes))
    elif level == 'postal2':
        df_cell = df_cell['POSTCODE'].dropna()
        array_zip_codes = [element[0:2] for element in df_cell]
        return np.sort(np.unique(array_zip_codes))
    else:
        raise ValueError(
            'The specified state space level is not implemented. Please choose either antenna,postal,postal3')

def calculate_distance_prior(number_of_states,states,level):
    # Initialize the pgeocode geodistance object for the Netherlands
    geoDistance = pgeocode.GeoDistance('nl')

    # Initialize distance matrix
    if level=='antenna':
        states_copy = states.copy()
        states = [element[0:4] for element in states]
    elif level=='postal':
        states_copy = states.copy()
    elif level=='postal3':
        states_copy = states.copy()
        states = np.char.add(np.array(states,dtype=str), np.array(number_of_states*['1'],dtype=str))
    elif level == 'postal2':
        states_copy = states.copy()
        states = np.char.add(np.array(states, dtype=str), np.array(number_of_states * ['11'], dtype=str))

    else:
        raise ValueError('State space level is not implemented.')

    distance_matrix = pd.DataFrame(0.0,index=states,columns=states)
    # Calculate the distance between each pair of postal codes
    for i in states:
        for j in states:
            if i != j:
                distance_matrix.loc[i, j] += geoDistance.query_postal_code(x=i,y=j)
            else:
                distance_matrix.loc[i, j] += 0  # Distance to itself is 0
    distance_matrix = 1/np.sqrt(np.ones(np.shape(distance_matrix))+distance_matrix) # Should find a better function based on literature. Maybe the power law?
    distance_matrix[np.isnan(distance_matrix)] = 1/number_of_states
    normalisation_matrix = np.transpose(np.tile(1/(distance_matrix.apply(func='sum',axis=0)),number_of_states).reshape(number_of_states,number_of_states))
    distance_matrix = np.multiply(distance_matrix,normalisation_matrix) # Normalises matrix. Not sure if this is needed.
    distance_matrix = distance_matrix.set_axis(states_copy,axis=0)
    distance_matrix = distance_matrix.set_axis(states_copy, axis=1)
    return distance_matrix

states = state_space_Omega(cell_file, bounding_box, antenna_type,level)
distance_matrix = calculate_distance_prior(len(states),states,level=level)
print(distance_matrix)
# distance_matrix.to_csv('./{0}/{1}.csv'.format(antenna_type,level))
distance_matrix.to_csv('./{0}_dist2/{1}.csv'.format(antenna_type,level))