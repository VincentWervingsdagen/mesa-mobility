import pandas as pd

from telcell.models import Model
import construct_markov_chains as MC


class MarkovChain():
    bounding_box: list
    state_space: list
    markov_chain_normal_phone: list[list]
    markov_chain_burner_phone: list[list]
    evidence: float
    number_of_states: int
    prior_chain: list[list]
    state_space_level: str
    observations_normal: pd.DataFrame
    observations_burner: pd.DataFrame

    def __init__(
        self,
        cell_file,
        observation_file,
        bounding_box,
        state_space='Omega',
        state_space_level='postal3',
        antenna_type='LTE',
        prior_type = 'distance',
        markov_type = 'discrete',
        distance = 'freq-distance',
        loops_allowed = True
    ) -> None:
        self.bounding_box = bounding_box
        self.state_space_level = state_space_level
        # Read in the observations.
        self.observations_normal,self.observations_burner=MC.transform_data(observation_file,self.state_space_level)

        # Construct the state space
        if state_space == 'Omega': #Select all the antennas/postal/postal3 codes in the bounding box
            self.state_space = MC.state_space_Omega(cell_file, self.bounding_box, antenna_type,
                                                                         self.state_space_level)
        elif state_space == 'observations': #Select the antennas/postal/postal3 codes that were observed by either of the phones.
            self.state_space = MC.state_space_observations(self.observations_normal,self.observations_burner)
        else:
            raise ValueError('The specified state space is not implemented. Please use Omega or observations.')
        self.number_of_states = len(self.state_space)

        # Construct the prior
        if prior_type == 'uniform': # Returns a nxn matrix with each value 1/n
            self.prior_chain = MC.uniform_prior(number_of_states=self.number_of_states,states=self.state_space)
        elif prior_type == 'distance': # Returns a nxn matrix based on the distance between the antennas/postal/postal3 codes.
            self.prior_chain = MC.distance_prior(number_of_states=self.number_of_states,states=self.state_space,level=self.state_space_level)
        elif prior_type == 'population': # Not implemented.
            self.prior_chain = MC.population_prior()
        else:
            raise ValueError('The specified prior movement distribution is not implemented. Please use uniform, distance or population.')

        # Construct the markov chains
        if markov_type == 'discrete':
            self.markov_chain_normal_phone = MC.discrete_markov_chain(df=self.observations_normal,prior=self.prior_chain,states=self.state_space,loops_allowed=loops_allowed)
            self.markov_chain_burner_phone = MC.discrete_markov_chain(df=self.observations_burner,prior=self.prior_chain,states=self.state_space,loops_allowed=loops_allowed)
        elif markov_type == 'continuous':
            MC.continuous_markov_chain()
        else:
            raise ValueError('The specified Markov chain type is not implemented. Please use discrete or continuous.')

        # Calculate the distance
        if distance == 'cut-distance':
            score = MC.genetic_cut_distance(matrix_normal=self.markov_chain_normal_phone,matrix_burner=self.markov_chain_burner_phone,states=self.state_space,number_of_states=self.number_of_states)
        elif distance == 'freq-distance':
            MC.frequent_transition_distance()
        else:
            raise ValueError('The specified Markov chain type is not implemented. Please use discrete or continuous.')
        print('reached the end!')