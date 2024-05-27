import numpy as np
import pandas as pd
from pyproj import Transformer
import warnings
import pgeocode
import random
from deap import base, creator, tools, algorithms


def transform_data(observation_file,level):
    df_observations = pd.read_csv(observation_file)
    if level == 'antenna':
        pass
    elif level == 'postal':
        df_observations['cellinfo.postal_code'] = [element[0:4] for element in df_observations['cellinfo.postal_code']]
    elif level == 'postal3':
        df_observations['cellinfo.postal_code'] = [element[0:3] for element in df_observations['cellinfo.postal_code']]
    else:
        raise ValueError('The specified state space level is not implemented. Please choose either antenna,postal,postal3')
    df_normal = df_observations[df_observations['device'] == '0_1']
    df_burner = df_observations[df_observations['device'] == '0_2']
    df_normal = df_normal[['timestamp', 'cellinfo.postal_code']]
    df_burner = df_burner[['timestamp', 'cellinfo.postal_code']]
    return df_normal,df_burner

def state_space_Omega(cell_file,bounding_box,antenna_type,level='postal3'):
    if bounding_box == (4.1874, 51.8280, 4.593, 52.0890): # You can calculate the state space omega for differnt bounding boxes with
        # Calculate_distance_omega.py as well and then read them in here, to make the program a lot quicker for antenna/postal state space.
        return pd.read_csv('./LTE/{0}.csv'.format(level))
    else:
        pass
    df_cell = pd.read_csv(cell_file)
    #drop useless columns
    df_cell = df_cell.drop(['Samenvatting','Vermogen', 'Frequentie','Veilige afstand','id','objectid','WOONPLAATSNAAM','DATUM_PLAATSING',
           'DATUM_INGEBRUIKNAME','GEMNAAM','Hoogte','Hoofdstraalrichting','sat_code'],axis=1)
    #drop types except the antenna_type
    df_cell = df_cell.loc[df_cell['HOOFDSOORT'] == antenna_type]
         #Transform to wgs84
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
    else:
        raise ValueError('The specified state space level is not implemented. Please choose either antenna,postal,postal3')


def state_space_observations(df_normal,df_burner):
    states = np.sort(np.unique(pd.concat([df_normal['cellinfo.postal_code'],df_burner['cellinfo.postal_code']])))
    return states

def uniform_prior(number_of_states,states):
    return pd.DataFrame(1/number_of_states,index=states,columns=states)

def distance_prior(number_of_states,states,level):
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
    distance_matrix = 1/np.sqrt(np.exp(distance_matrix)) # Should find a better function based on literature. Maybe the power law?
    distance_matrix[np.isnan(distance_matrix)] = np.finfo(float).eps
    normalisation_matrix = np.transpose(np.tile(1/(distance_matrix.apply(func='sum',axis=0)),number_of_states).reshape(number_of_states,number_of_states))
    distance_matrix = distance_matrix*normalisation_matrix # Normalises matrix. Not sure if this is needed.
    distance_matrix = distance_matrix.set_axis(states_copy,axis=0)
    distance_matrix = distance_matrix.set_axis(states_copy, axis=1)
    return distance_matrix

def population_prior():
    raise NotImplementedError

def discrete_markov_chain(df,prior,states,loops_allowed): # Calculates posterior mean markov chain
    # First construct count matrix
    matrix_normal = pd.DataFrame(0.0,index=states,columns=states)
    for i in df.index[:-1]:
        matrix_normal.loc[df['cellinfo.postal_code'][i],df['cellinfo.postal_code'][i+1]] += 1
    # Add prior.
    matrix_normal = matrix_normal + prior

    if loops_allowed == True:
        pass
    elif loops_allowed == False:
        matrix_normal.values[[np.arange(df.shape[0])] * 2] = 0
    else:
        raise ValueError('Loops allowed must be a bool variable.')
    #Normalise
    matrix_normal = matrix_normal/matrix_normal.apply(func='sum',axis=0)
    matrix_normal = np.transpose(matrix_normal)
    return matrix_normal


def continuous_markov_chain():
    raise NotImplementedError


def cut_weight(matrix,S,T):
    if (len(S)==0) or (len(T)==0):
        return 0
    else:
        return matrix.loc[S,T].to_numpy().sum()

def cut_distance(individual,matrix_normal,matrix_burner,states):
    S = states[np.array(individual) == 1]
    T = states[np.array(individual) == 0]
    distance = np.abs(cut_weight(matrix_normal,S,T)-cut_weight(matrix_burner,S,T))
    return distance,

def genetic_cut_distance(matrix_normal,matrix_burner,states,number_of_states):
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()
    toolbox.register("attr_bool", random.randint, 0, 1)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, n=number_of_states)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    toolbox.register("evaluate", cut_distance, matrix_normal=matrix_normal, matrix_burner=matrix_burner, states=states)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
    toolbox.register("select", tools.selTournament,tournsize=3)

    # Genetic Algorithm parameters
    population = toolbox.population(n=100)
    ngen = 10
    cxpb = 0.5
    mutpb = 0.2

    # Run the Genetic Algorithm
    result = algorithms.eaSimple(population, toolbox, cxpb, mutpb, ngen, verbose=False)

    # Extract the best individual
    best_ind = tools.selBest(population, 1)[0]
    S = [states[i] for i in range(len(best_ind)) if best_ind[i] == 1]
    T = [states[i] for i in range(len(best_ind)) if best_ind[i] == 0]

    print("Best partition:")
    print("S:", S)
    print("T:", T)
    print("ind:", best_ind)
    print("Cut distance:", (1/number_of_states)*cut_distance(best_ind,matrix_normal, matrix_burner,states)[0])

    return (1/number_of_states)*cut_distance(best_ind,matrix_normal, matrix_burner,states)[0]

def frequent_transition_distance():
    raise NotImplementedError