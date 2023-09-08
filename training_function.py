import random
import traceback

import lugo4py
import lugo4py.mapper as mapper
import lugo4py.rl as rl
import threading

# Training settings
train_iterations = 50
steps_per_iteration = 600

def my_training_function(training_ctrl: rl.TrainingController, stop_event: threading.Event):
    print("Let's train")

    possible_actions = [
        mapper.DIRECTION.FORWARD,
        mapper.DIRECTION.BACKWARD,
        mapper.DIRECTION.LEFT,
        mapper.DIRECTION.RIGHT,
        mapper.DIRECTION.BACKWARD_LEFT,
        mapper.DIRECTION.BACKWARD_RIGHT,
        mapper.DIRECTION.FORWARD_RIGHT,
        mapper.DIRECTION.FORWARD_LEFT,
    ]

    scores = []
    for i in range(train_iterations):
        try:
            scores.append(0)
            training_ctrl.set_environment({"iteration": i})

            for j in range(steps_per_iteration):
                if stop_event.is_set():
                    training_ctrl.stop()
                    print("trainning stopped")
                    return

                _ = training_ctrl.get_state()

                # The sensors would feed our training model, which would return the next action
                action = possible_actions[random.randint(
                    0, len(possible_actions) - 1)]

                # Then we pass the action to our update method
                result = training_ctrl.update(action)
                # Now we should reward our model with the reward value
                scores[i] += result["reward"]
                if result["done"]:
                    # No more steps
                    print(f"End of train_iteration {i}, score:", scores[i])
                    break

        except Exception as e:
            traceback.print_exc()
            print(f"error during training session:", e.__traceback__)

    training_ctrl.stop()
    print("Training is over, scores:", scores)

