#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from typing import Tuple, Callable, List

import toh_mdp as tm


def value_iteration(
        mdp: tm.TohMdp, v_table: tm.VTable
) -> Tuple[tm.VTable, tm.QTable, float]:
    """Computes one step of value iteration.

    Hint 1: Since the terminal state will always have value 0 since
    initialization, you only need to update values for nonterminal states.

    Hint 2: It might be easier to first populate the Q-value table.

    Args:
        mdp: the MDP definition.
        v_table: Value table from the previous iteration.

    Returns:
        new_v_table: tm.VTable
            New value table after one step of value iteration.
        q_table: tm.QTable
            New Q-value table after one step of value iteration.
        max_delta: float
            Maximum absolute value difference for all value updates, i.e.,
            max_s |V_k(s) - V_k+1(s)|.
    """
    new_v_table: tm.VTable = v_table.copy()
    q_table: tm.QTable = {}
    # noinspection PyUnusedLocal
    max_delta = 0.0
    for state in mdp.nonterminal_states:
        #initializes the v value to a low number in order to find accurate maximum value
        new_v_table[state] = -1000000
        for action in mdp.actions:
            q_value = 0.0
            for next_state in mdp.all_states:
                q_value += mdp.transition(state,action,next_state)*(mdp.reward(state,action,next_state)+(mdp.config.gamma*v_table[next_state]))
            q_table[(state,action)] = q_value
            #v value is maximum of q values for state and the possible actions
            new_v_table[state] = max(q_table[(state,action)], new_v_table[state])
        max_delta = max(abs(v_table[state] - new_v_table[state]), max_delta)
    return new_v_table, q_table, max_delta


def extract_policy(
        mdp: tm.TohMdp, q_table: tm.QTable
) -> tm.Policy:
    """Extract policy mapping from Q-value table.

    Remember that no action is available from the terminal state, so the
    extracted policy only needs to have all the nonterminal states (can be
    accessed by mdp.nonterminal_states) as keys.

    Args:
        mdp: the MDP definition.
        q_table: Q-Value table to extract policy from.

    Returns:
        policy: tm.Policy
            A Policy maps nonterminal states to actions.
    """
    p_table: tm.Policy = {}
    for state in mdp.nonterminal_states:
        #initializes the q value to a low number in order to find accurate maximum value
        max_q = -1000000
        best_action = ''
        for state_action in q_table:
        #optimal policy for a state is the action that will result in the maximum q value for that state
            if state == state_action[0] and q_table[state_action] > max_q:
                max_q = q_table[state_action]
                best_action = state_action[1]
        p_table[state] = best_action
    return p_table


def q_update(
        mdp: tm.TohMdp, q_table: tm.QTable,
        transition: Tuple[tm.TohState, tm.TohAction, float, tm.TohState],
        alpha: float) -> None:
    """Perform a Q-update based on a (S, A, R, S') transition.

    Update the relevant entries in the given q_update based on the given
    (S, A, R, S') transition and alpha value.

    Args:
        mdp: the MDP definition.
        q_table: the Q-Value table to be updated.
        transition: A (S, A, R, S') tuple representing the agent transition.
        alpha: alpha value (i.e., learning rate) for the Q-Value update.
    """
    state, action, reward, next_state = transition
    #terminal state has value of 0
    if next_state == mdp.terminal:
        max_q = 0.0
    else:
        #find the maximum q value for the next state
        max_q = -1000000
        for next_action in mdp.actions:
            max_q = max(q_table[(next_state, next_action)], max_q)
    q_table[(state,action)] = ((1-alpha)*q_table[(state,action)]) + (alpha*(reward + (mdp.config.gamma * max_q)))

    

def extract_v_table(mdp: tm.TohMdp, q_table: tm.QTable) -> tm.VTable:
    """Extract the value table from the Q-Value table.

    Args:
        mdp: the MDP definition.
        q_table: the Q-Value table to extract values from.

    Returns:
        v_table: tm.VTable
            The extracted value table.
    """
    v_table: tm.VTable = {}
    #terminal state has value of 0
    v_table[mdp.terminal] = 0.0
    #v_table contains the maximum q values for a state
    for state in mdp.nonterminal_states:
        v_table[state] = -1000000
        for action in mdp.actions:
            v_table[state] = max(q_table[(state,action)], v_table[state])
    return v_table


def choose_next_action(
        mdp: tm.TohMdp, state: tm.TohState, epsilon: float, q_table: tm.QTable,
        epsilon_greedy: Callable[[List[tm.TohAction], float], tm.TohAction]
) -> tm.TohAction:
    """Use the epsilon greedy function to pick the next action.

    You can assume that the passed in state is neither the terminal state nor
    any goal state.

    You can think of the epsilon greedy function passed in having the following
    definition:

    def epsilon_greedy(best_actions, epsilon):
        # selects one of the best actions with probability 1-epsilon,
        # selects a random action with probability epsilon
        ...

    See the concrete definition in QLearningSolver.epsilon_greedy.

    Args:
        mdp: the MDP definition.
        state: the current MDP state.
        epsilon: epsilon value in epsilon greedy.
        q_table: the current Q-value table.
        epsilon_greedy: a function that performs the epsilon

    Returns:
        action: tm.TohAction
            The chosen action.
    """
    max_q = -1000000
    actions = []
    #find the maximum q value for the state
    for state_action in q_table:
        if state == state_action[0] and q_table[state_action] > max_q:
            max_q = q_table[state_action]
    #find all actions from the state that result in the maximum q value
    for state_action2 in q_table:
        if state == state_action2[0] and q_table[state_action2] == max_q:
            actions.append(state_action2[1])
    return epsilon_greedy(actions, epsilon)
    
 

def custom_epsilon(n_step: int) -> float:
    """Calculates the epsilon value for the nth Q learning step.

    Define a function for epsilon based on `n_step`.

    Args:
        n_step: the nth step for which the epsilon value will be used.

    Returns:
        epsilon: float
            epsilon value when choosing the nth step.
    """
    #since exploration decreases over time, as the number of steps increases, epsilon should decrease and approach 0
    #to reduce unecessary state exploration. A function 1/n will approach 0 as n approaches infinity. Additionally,
    #the next epsilon greedy policy is an improvement to the current policy, so the function should take into account
    #the next step
    return 1/n_step + 1


def custom_alpha(n_step: int) -> float:
    """Calculates the alpha value for the nth Q learning step.

    Define a function for alpha based on `n_step`.

    Args:
        n_step: the nth update for which the alpha value will be used.

    Returns:
        alpha: float
            alpha value when performing the nth Q update.
    """
    # *** BEGIN OF YOUR CODE ***
