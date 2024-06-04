import itertools
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pyproj import Transformer
import pgeocode
import random
from deap import base, creator, tools, algorithms
from scipy.linalg import eig


def transform_data(observation_file,level,phone_normal,phone_burner):
    df_observations = pd.read_csv(observation_file)
    # Create a mask for each tuple
    mask1 = (df_observations['device'] == phone_normal)
    mask2 = (df_observations['device'] == phone_burner)

    # Combine the masks using logical OR
    combined_mask = mask1 | mask2
    df_observations = df_observations[combined_mask]
    df_observations = df_observations.dropna()  # Drop nan values
    df_observations = df_observations.reset_index() # Reset the index
    pattern = r'^\d{4}[A-Z]{2}$' # Check whether every item is a valid postal code.
    valid_postal_codes = df_observations['cellinfo.postal_code'].str.match(pattern)
    df_observations = df_observations[valid_postal_codes]
    if level == 'antenna':
        pass
    elif level == 'postal':
        df_observations['cellinfo.postal_code'] = [element[0:4] for element in df_observations['cellinfo.postal_code']]
    elif level == 'postal3':
        df_observations['cellinfo.postal_code'] = [element[0:3] for element in df_observations['cellinfo.postal_code']]
    elif level == 'postal2':
        df_observations['cellinfo.postal_code'] = [element[0:2] for element in df_observations['cellinfo.postal_code']]
    else:
        raise ValueError('The specified state space level is not implemented. Please choose either antenna,postal,postal3')
    df_normal = df_observations[df_observations['device'] == phone_normal]
    df_burner = df_observations[df_observations['device'] == phone_burner]
    df_normal = df_normal[['timestamp', 'cellinfo.postal_code']]
    df_burner = df_burner[['timestamp', 'cellinfo.postal_code']]
    return df_normal,df_burner

def state_space_Omega(cell_file,bounding_box,antenna_type,level='postal3'):
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
    elif level == 'postal2':
        df_cell = df_cell['POSTCODE'].dropna()
        array_zip_codes = [element[0:2] for element in df_cell]
        return np.sort(np.unique(array_zip_codes))
    else:
        raise ValueError('The specified state space level is not implemented. Please choose either antenna,postal,postal3')


def state_space_observations(df_normal,df_burner):
    states = np.sort(np.unique(pd.concat([df_normal['cellinfo.postal_code'],df_burner['cellinfo.postal_code']])))
    return states

def uniform_prior(number_of_states,states):
    return pd.DataFrame(1/number_of_states,index=states,columns=states)

def zero_prior(states):
    return pd.DataFrame(1000*np.finfo(float).eps,index=states,columns=states)

def distance_prior(distance_matrix_file,states,bounding_box):
    if bounding_box == (4.1874, 51.8280, 4.593, 52.0890):
        distance_matrix = pd.read_csv(distance_matrix_file,
                                      dtype={'Unnamed: 0': str})
        distance_matrix_index = distance_matrix['Unnamed: 0']
        distance_matrix = distance_matrix.drop('Unnamed: 0', axis=1)
        distance_matrix = distance_matrix.set_index(distance_matrix_index)
        distance_matrix.columns = distance_matrix_index
        distance_matrix = distance_matrix.loc[states,states]
        return distance_matrix
    else:
        raise NotImplementedError('The distance matrix for state space level {0}, was not calculated for this bounding box. Please see Calculate_distance_omega.'.format(level))


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
        np.fill_diagonal(matrix_normal.values,0)
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
    try:
        del creator.FitnessMax
        del creator.Individual
    except Exception as e:
        pass

    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()
    toolbox.register("attr_bool", np.random.binomial, 1, np.random.uniform(0.1,0.9))
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, n=number_of_states)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    toolbox.register("evaluate", cut_distance, matrix_normal=matrix_normal, matrix_burner=matrix_burner, states=states)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
    toolbox.register("select", tools.selTournament,tournsize=3)

    # Genetic Algorithm parameters
    population = toolbox.population(n=1000)
    ngen = 20
    cxpb = 0.5
    mutpb = 0.2

    # Run the Genetic Algorithm
    result = algorithms.eaSimple(population, toolbox, cxpb, mutpb, ngen, verbose=False)

    # Extract the best individual
    best_ind = tools.selBest(population, 1)[0]
    S = [states[i] for i in range(len(best_ind)) if best_ind[i] == 1]
    T = [states[i] for i in range(len(best_ind)) if best_ind[i] == 0]

    # print("Best partition:")
    # print("S:", S)
    # print("T:", T)
    # print("ind:", best_ind)
    # print("Cut distance:", (1/number_of_states)*cut_distance(best_ind,matrix_normal, matrix_burner,states)[0])

    return (1/number_of_states)*cut_distance(best_ind,matrix_normal, matrix_burner,states)[0]


def important_states_cut_distance(matrix_normal,matrix_burner,states):
    stationary_distribution = calculate_stationary_distribution(matrix_normal)
    index_important_states = np.argsort(stationary_distribution)[-5:]
    distances = []
    for element in itertools.product(range(2),repeat=5):
        individual = np.zeros(len(states))
        individual[index_important_states] = element
        distances.append(cut_distance(individual,matrix_normal,matrix_burner,states)[0])
    return max(distances)


def calculate_stationary_distribution(transition_matrix):
    # Ensure the transition matrix is a square matrix
    assert transition_matrix.shape[0] == transition_matrix.shape[1], "Transition matrix must be square"

    # Compute the eigenvalues and right eigenvectors
    eigenvalues, eigenvectors = eig(transition_matrix.T)

    # Find the index of the eigenvalue 1 (stationary distribution)
    stationary_index = np.argmin(np.abs(eigenvalues - 1))

    # Extract the corresponding eigenvector
    stationary_vector = np.real(eigenvectors[:, stationary_index])

    # Normalize the stationary vector to sum to 1
    stationary_distribution = stationary_vector / stationary_vector.sum()

    return stationary_distribution

def calculate_conductance(individual,matrix,states,stationary_distribution,number_of_states):
    if (np.array(individual).sum() == 0) | (np.array(individual).sum() == number_of_states):
        return 1, # Not a possible individual
    S = states[np.array(individual) == 1]
    T = states[np.array(individual) == 0]

    conductance = (stationary_distribution[np.array(individual) == 1]*matrix.loc[S, T].to_numpy().sum(axis=1)).sum()
    return conductance,

def calculate_freq_distance(matrix_normal,matrix_burner,stationary_distribution,conductance,number_of_states):
    frequent_transition_matrix_normal = np.multiply(np.transpose(number_of_states*[stationary_distribution]),(matrix_normal.to_numpy()))
    # frequent_transition_matrix_burner = np.multiply(np.transpose(number_of_states*[stationary_distribution]),(matrix_burner.to_numpy()))
    frequent_transition_matrix_normal = frequent_transition_matrix_normal>conductance
    # frequent_transition_matrix_burner = frequent_transition_matrix_burner>conductance
    # frequent_transition_matrix = (frequent_transition_matrix_normal+frequent_transition_matrix_burner)>0
    frequent_transition_matrix = frequent_transition_matrix_normal
    result = np.abs(np.ones(frequent_transition_matrix.sum())-matrix_burner.to_numpy()[frequent_transition_matrix]/matrix_normal.to_numpy()[frequent_transition_matrix])
    if len(result)==0:
        return 100000.,100000.

    # print('number of frequent transitions: ',len(result))
    # true_indices = np.argwhere(frequent_transition_matrix)
    # max_distance_indices = true_indices[result.argmax()] # Useful for printing the rows of the maximum distance etc
    # print(max_distance_indices)
    # print(pd.Series(matrix_burner.to_numpy()[max_distance_indices[0]]))
    # print(pd.Series(matrix_normal.to_numpy()[max_distance_indices[0]]))
    return max(result)

def frequent_transition_distance(matrix_normal,matrix_burner,states,number_of_states):
    stationary_distribution = calculate_stationary_distribution(matrix_normal)

    try:
        del creator.FitnessMax
        del creator.Individual
    except Exception as e:
        pass

    creator.create("FitnessMax", base.Fitness, weights=(-1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()
    toolbox.register("attr_bool", np.random.binomial, 1, 0.1)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, n=number_of_states)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    toolbox.register("evaluate", calculate_conductance, matrix=matrix_normal, states=states,stationary_distribution=stationary_distribution,number_of_states=number_of_states)
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
    conductance = calculate_conductance(best_ind,matrix_normal,states=states,stationary_distribution=stationary_distribution,number_of_states=number_of_states)[0]

    # print("Best partition:")
    # print("S:", S)
    # print("T:", T)
    # print("ind:", best_ind)
    # print("Conductance:", conductance)

    distance = calculate_freq_distance(matrix_normal=matrix_normal,matrix_burner=matrix_burner,stationary_distribution=stationary_distribution,conductance=conductance,number_of_states=number_of_states)

    # print('distance:',distance)

    return distance

def frobenius_norm(matrix_normal,matrix_burner):
    return np.linalg.norm(x=(matrix_normal-matrix_burner),ord='fro')

def trace_norm(matrix_normal,matrix_burner):
    return np.linalg.norm(x=(matrix_normal-matrix_burner),ord='nuc')