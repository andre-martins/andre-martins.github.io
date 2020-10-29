import numpy as np
from hw2_decoder import viterbi, forward_backward


def main():
    weather = ["Sunny", "Windy", "Rainy"]
    activities = ["Surf", "Beach", "Videogame", "Study"]
    str2state = {w: i for i, w in enumerate(weather)}
    str2emission = {e: i for i, e in enumerate(activities)}

    # emissions probabilities from the handout: emission_probabilities[i, j]
    # is the probability of the emission i given the state j
    emission_probabilities = np.array(
        [[0.4, 0.5, 0.1],
         [0.4, 0.1, 0.1],
         [0.1, 0.2, 0.3],
         [0.1, 0.2, 0.5]]
    )

    # transition probabilities from the handout: transition_probabilities[i, j]
    # is the probability of transitioning to state i from state j
    transition_probabilities = np.array(
        [[0.6, 0.3, 0.2],
         [0.3, 0.5, 0.3],
         [0.1, 0.2, 0.5]]
    )

    observations = ["Videogame",
                    "Study",
                    "Study",
                    "Surf",
                    "Beach",
                    "Videogame",
                    "Beach"]

    initial_weather = "Rainy"
    final_weather = "Sunny"

    x = [str2emission[observation] for observation in observations]
    emission_scores = np.log(emission_probabilities[x])
    transition_scores = np.log(np.array(
        [transition_probabilities for observation in observations]
    ))

    initial_state = str2state[initial_weather]
    final_state = str2state[final_weather]

    initial_scores = np.log(transition_probabilities[:, initial_state])
    final_scores = np.log(transition_probabilities[final_state])

    viterbi_path = viterbi(
        initial_scores, transition_scores, final_scores, emission_scores)
    viterbi_weather = [weather[i] for i in viterbi_path]

    posteriors, _, _ = forward_backward(
        initial_scores, transition_scores, final_scores, emission_scores)
    posterior_path = posteriors.argmax(1)
    posterior_weather = [weather[i] for i in posterior_path]

    print("Sequence given by Viterbi: %s" % " -> ".join(viterbi_weather))
    print("Sequence given by posterior: %s" % " -> ".join(posterior_weather))


if __name__ == "__main__":
    main()
