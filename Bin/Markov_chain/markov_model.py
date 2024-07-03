from typing import Tuple, Optional, Mapping

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from collections import defaultdict
from tqdm import tqdm

import lir

import construct_markov_chains as MC


class MarkovChain():
    bounding_box: list
    prior_chain: pd.DataFrame
    series_Markov_chains: pd.Series(pd.DataFrame)
    data: pd.DataFrame
    markov_tye: str
    loops_allowed: bool
    distance: str
    scores_H_p_train: list[float]
    scores_H_d_train: list[float]
    scores_H_p_test: list[float]
    scores_H_d_test: list[float]
    kde_calibrator: lir.KDECalibrator()


    def __init__(
            self,
            observation_file,
            cell_file,
            distance_matrix_file,
            bounding_box,
            state_space='observations',
            state_space_level='postal2',
            antenna_type='LTE',
            prior_type='distance',
            markov_type='discrete',
            distance='frobenius',
            loops_allowed=True
    ) -> None:
        # Set global parameters
        self.cell_file = cell_file
        self.distance_matrix_file = distance_matrix_file
        self.bounding_box = bounding_box
        self.loops_allowed = loops_allowed
        self.markov_tye = markov_type
        self.distance = distance

        # Transform the data to a dataframe with owner, device, timestamp, postal_code
        self.data = MC.transform_data(observation_file,state_space_level)
        # Construct the state space.
        self.construct_state_space(state_space,state_space_level,antenna_type)
        # Construct the prior matrix, so that we can use it for estimating our movement matrices.
        self.construct_prior(prior_type)

        # Construct the Markov chains for each device.
        list_Markov_chains = []
        list_devices = []
        grouped = self.data.groupby(['device'])

        for device, track in grouped:
            list_Markov_chains.append(self.construct_markov_chain(track,markov_type,loops_allowed))
            list_devices.append(device[0])

        self.series_Markov_chains = pd.Series(list_Markov_chains,index=list_devices)

        df_train,df_test,df_validation = self.make_test_train_set(list_devices,distance)

        self.kde_calibrator = lir.KDECalibrator(bandwidth='silverman')
        self.kde_calibrator.fit(np.array(df_train['score']), np.array(df_train['hypothesis']))

        with lir.plotting.show() as ax:
            ax.calibrator_fit(self.kde_calibrator, score_range=[0, df_train['score'].max()+1]);
            ax.score_distribution(scores=df_train['score'], y=df_train['hypothesis'],
                                  bins=np.linspace(0, df_train['score'].max()+1, 10), weighted=True)
            ax.xlabel('distance')
            H1_legend = mpatches.Patch(color='tab:blue', alpha=.3, label='H_p-true')
            H2_legend = mpatches.Patch(color='tab:orange', alpha=.3, label='H_d-true')
            ax.legend(handles=[H1_legend, H2_legend])

        plt.show()

        self.kde_calibrator = lir.ELUBbounder(self.kde_calibrator)
        lrs_train_kde = self.kde_calibrator.fit_transform(np.array(df_train['score']), np.array(df_train['hypothesis']))

        plt.scatter(np.array(df_train['score']), np.log10(lrs_train_kde), color='tab:purple', marker=".",
                    label='KDE')
        plt.xlim([0, df_train['score'].max()+1])

        plt.legend()

        plt.xlabel('distance')
        plt.ylabel('log(LR)')
        plt.show()

        with lir.plotting.show() as ax:
            ax.pav(lrs_train_kde, np.array(df_train['hypothesis']))

        plt.show()

        cllr = lir.metrics.cllr(lrs_train_kde, np.array(df_train['hypothesis']))
        print(f'\n The log likelihood ratio cost is {cllr} (lower is better)\n')




        # self.kde_calibrator = lir.ELUBbounder(lir.KDECalibrator(bandwidth='silverman'))
        # self.kde_calibrator.fit(np.array(df_validation['score']), np.array(df_validation['hypothesis']))
        #
        # with lir.plotting.show() as ax:
        #     ax.calibrator_fit(self.kde_calibrator, score_range=[0, df_validation['score'].max()+1]);
        #     ax.score_distribution(scores=df_validation['score'], y=df_validation['hypothesis'],
        #                           bins=np.linspace(0, df_validation['score'].max()+1, 10), weighted=True)
        #     ax.xlabel('distance')
        #     H1_legend = mpatches.Patch(color='tab:blue', alpha=.3, label='H_p-true')
        #     H2_legend = mpatches.Patch(color='tab:orange', alpha=.3, label='H_d-true')
        #     ax.legend(handles=[H1_legend, H2_legend])
        #
        # plt.show()
        #
        # lr = self.kde_calibrator.transform(np.array(df_test['score']))
        # plt.scatter(np.array(df_test['score']), np.log10(lr), color='tab:purple', marker=".",
        #             label='KDE')
        # plt.xlim([0, df_train['score'].max()+1])
        #
        # plt.legend()
        #
        # plt.xlabel('distance')
        # plt.ylabel('log(LR)')
        # plt.show()
        #
        # with lir.plotting.show() as ax:
        #     ax.pav(lr, np.array(df_test['hypothesis']))
        #
        # plt.show()
        #
        # with lir.plotting.show() as ax:
        #     ax.ece(lr, np.array(df_test['hypothesis']),log_prior_odds_range=[-3,3])
        #
        # plt.show()
        #
        # cllr = lir.metrics.cllr(lr, np.array(df_test['hypothesis']))
        # print(f'\n The log likelihood ratio cost is {cllr} (lower is better)\n')


    def construct_state_space(self,state_space,state_space_level,antenna_type):
        # Construct the state space
        if state_space == 'Omega':  # Select all the antennas/postal/postal3 codes in the bounding box
            self.state_space = MC.state_space_Omega(self.cell_file, self.bounding_box,antenna_type,
                                                    state_space_level)
        elif state_space == 'observations':  # Select the antennas/postal/postal3 codes that were observed by either of the phones.
            self.state_space = MC.state_space_observations(self.data)
        else:
            raise ValueError('The specified state space is not implemented. Please use Omega or observations.')
        self.number_of_states = len(self.state_space)

    def construct_prior(self,prior_type):
        # Construct the prior
        if prior_type == 'uniform':  # Returns a nxn matrix with each value 1/n
            self.prior_chain = MC.uniform_prior(number_of_states=self.number_of_states, states=self.state_space)
        elif prior_type == 'zero':  # Returns a nxn matrix based on the distance between the antennas/postal/postal3 codes.
            self.prior_chain = MC.zero_prior(states=self.state_space)
        elif prior_type == 'distance':  # Returns a nxn matrix based on the distance between the antennas/postal/postal3 codes.
            self.prior_chain = MC.distance_prior(states=self.state_space, distance_matrix_file=self.distance_matrix_file,
                                                 bounding_box=self.bounding_box)
        elif prior_type == 'population':  # Not implemented.
            self.prior_chain = MC.population_prior()
        else:
            raise ValueError(
                'The specified prior movement distribution is not implemented. Please use uniform, distance or population.')

    def construct_markov_chain(self,track,markov_type,loops_allowed):
        # Construct the markov chains
        if markov_type == 'discrete':
            return MC.discrete_markov_chain(track=track,prior=self.prior_chain, states=self.state_space,
                                                                      loops_allowed=loops_allowed)
        elif markov_type == 'continuous':
            MC.continuous_markov_chain()
        else:
            raise ValueError('The specified Markov chain type is not implemented. Please use discrete or continuous.')

    def calculate_score(self,distance,phone1,phone2):
        matrix1 = self.series_Markov_chains.loc[phone1]
        matrix2 = self.series_Markov_chains.loc[phone2]
        # Calculate the distance
        if distance == 'cut_distance':
            return MC.genetic_cut_distance(matrix_normal=matrix1,matrix_burner=matrix2)
        elif distance == 'freq_distance':
            return MC.frequent_transition_distance(matrix_normal=matrix1, matrix_burner=matrix2)
        elif distance == 'frobenius':
            return MC.frobenius_norm(matrix_normal=matrix1,matrix_burner=matrix2)
        elif distance == 'trace':
            return MC.trace_norm(matrix_normal=matrix1,matrix_burner=matrix2)
        elif distance == 'important_cut_distance_5':
            return MC.important_states_cut_distance_5(matrix_normal=matrix1,matrix_burner=matrix2)
        elif distance == 'important_cut_distance':
            return MC.important_states_cut_distance(matrix_normal=matrix1,matrix_burner=matrix2,data_normal=self.data[self.data['device']==phone1])
        else:
            raise ValueError(
                'The specified distance function is not implemented. Please use cut-distance, freq-distance, frobenius, trace or important_cut_distance.')

    def make_test_train_set(self,list_devices,distance):
        # Group the phones together per owner, this allows for owners having multiple phones.
        owner_groups = defaultdict(list)
        for device in list_devices:
            owner, dev = device.split('_')
            owner_groups[owner].append(device)

        grouped_devices = list(owner_groups.values())
        pairs_with_labels_H_p, pairs_with_labels_H_d = MC.create_pairs(np.concatenate(grouped_devices))
        list_phones = np.vstack([*pairs_with_labels_H_p,*pairs_with_labels_H_d])
        df_validation = pd.DataFrame({'phone1':[item[0] for item in list_phones],'phone2':[item[1] for item in list_phones],
                                'hypothesis':[*len(pairs_with_labels_H_p)*[1],*len(pairs_with_labels_H_d)*[0]]}, columns=['phone1','phone2','hypothesis'])

        tqdm.pandas()
        df_validation['score'] = df_validation.progress_apply(lambda row: self.calculate_score(distance,row['phone1'],row['phone2']),axis=1)

        list_owners_train,list_owners_test = train_test_split(grouped_devices,test_size=0.3,random_state=1)
        list_owners_train = [device for devices in list_owners_train for device in devices]
        list_owners_test = [device for devices in list_owners_test for device in devices]

        df_train = df_validation[(df_validation['phone1'].isin(list_owners_train))&(df_validation['phone2'].isin(list_owners_train))]
        df_test = df_validation[(df_validation['phone1'].isin(list_owners_test))&(df_validation['phone2'].isin(list_owners_test))]

        return df_train, df_test, df_validation
