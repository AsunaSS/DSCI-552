'''
Chaoyu Li
Date: 4/18/2022
'''

# five elements for HMM
states = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10)

obs = (8, 6, 4, 6, 5, 4, 5, 5, 7, 9)

prior_prob = {1: 0.1, 2: 0.1, 3: 0.1, 4: 0.1, 5: 0.1, 6: 0.1, 7: 0.1, 8: 0.1, 9: 0.1, 10: 0.1}

trans_prob = {
    1: {2: 1.0, 1: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0, 10: 0},
    2: {1: 0.5, 3: 0.5, 2: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0, 10: 0},
    3: {2: 0.5, 4: 0.5, 1: 0, 3: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0, 10: 0},
    4: {3: 0.5, 5: 0.5, 1: 0, 2: 0, 4: 0, 6: 0, 7: 0, 8: 0, 9: 0, 10: 0},
    5: {4: 0.5, 6: 0.5, 1: 0, 2: 0, 3: 0, 5: 0, 7: 0, 8: 0, 9: 0, 10: 0},
    6: {5: 0.5, 7: 0.5, 1: 0, 2: 0, 3: 0, 4: 0, 6: 0, 8: 0, 9: 0, 10: 0},
    7: {6: 0.5, 8: 0.5, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 7: 0, 9: 0, 10: 0},
    8: {7: 0.5, 9: 0.5, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 8: 0, 10: 0},
    9: {8: 0.5, 10: 0.5, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 9: 0},
    10: {9: 1.0, 1: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 2: 0, 10: 0}
}

emission_prob = {
    1: {1: 0.5, 2: 0.5, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0, 10: 0},
    2: {1: 1/3, 2: 1/3, 3: 1/3, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0, 10: 0},
    3: {2: 1/3, 3: 1/3, 4: 1/3, 1: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0, 10: 0},
    4: {3: 1/3, 4: 1/3, 5: 1/3, 1: 0, 2: 0, 6: 0, 7: 0, 8: 0, 9: 0, 10: 0},
    5: {4: 1/3, 5: 1/3, 6: 1/3, 1: 0, 2: 0, 3: 0, 7: 0, 8: 0, 9: 0, 10: 0},
    6: {5: 1/3, 6: 1/3, 7: 1/3, 1: 0, 2: 0, 3: 0, 4: 0, 8: 0, 9: 0, 10: 0},
    7: {6: 1/3, 7: 1/3, 8: 1/3, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 9: 0, 10: 0},
    8: {7: 1/3, 8: 1/3, 9: 1/3, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 10: 0},
    9: {8: 1/3, 9: 1/3, 10: 1/3, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0},
    10: {9: 0.5, 10: 0.5, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 1: 0, 2: 0}
}

def viterbi(obs, states, prior_prob, trans_prob, emission_prob):
    # The trellis consists of nodes for each possible state at each step in the hidden sequence.
    trellis = [{}]
    # The current path through the trellis.
    path = {}

    # Add the probabilities of beginning the sequence with each possible state
    for state in states:
        trellis[0][state] = prior_prob[state] * emission_prob[state][obs[0]]
        path[state] = [state]

    # Add probabilities for each subsequent state transitioning to each state.
    for obs_index in range(1, len(obs)):
        # Add a new path for the added step in the sequence.
        trellis.append({})
        new_path = {}
        # For each possible state,
        for state in states:
            # Find the most probable state of:
            # The previous most probable state's probability *
            # The probability of the previous most probable state transitioning to the predicted state *
            # The probability that the current observation corresponds to the predicted state
            (prob, lar_state) = max([(trellis[obs_index-1][y0] * trans_prob[y0][state] * emission_prob[state][obs[obs_index]], y0) for y0 in states])

            # Add the probability of the state occuring at this step of the sequence to the trellis.
            trellis[obs_index][state] = prob
            # Add the state to the current path
            new_path[state] = path[lar_state] + [state]

        print(trellis)
        print(new_path)

        path = new_path

    # Make a list of the paths that traverse the entire observation sequence and their probabilities, and select the most probable.
    (prob, state) = max([(trellis[len(obs) - 1][state], state) for state in states])
    # Return the most probable path and its probability.
    return prob, path[state]

if __name__ == '__main__':
    prob, final_path = viterbi(obs, states, prior_prob, trans_prob, emission_prob)
    print('The most likely sequence is: ', final_path)
    print('The possible of the sequence above is: ', prob)
