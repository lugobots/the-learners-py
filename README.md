# The Learners Py

**This is a beta project**

The Learners PY is a template of trainable bot to training models to [Lugo Bots](https://beta.lugobots.dev/).

## How to use the code


1. You may find the implementation of a [PyEnvironment](https://www.tensorflow.org/agents/api_docs/python/tf_agents/environments/PyEnvironment) on
[bot_environment.py](./bot_environment.py). It implements a basic wrapper to make easier to use TensorFlow. You likelly won't need to change it.
2. You definitly want to change the `training` function on [main.py](./main.py) file, though.  If you are newbie as whom wrotes that, leave it alone, but if you now what are you doing, that code will be not good for you. Just re-write your own code. 
Checkout and change the :
Checkout and change the :
3. (The most important): change the [Bot Trainer](./my_bot.py) to control the trainning session based on your model goals

![Drag Racing](reinforcement_learning_diagram.png)

```python
class BotTrainer:

    # (not seen at the picture above)
    # Sets up the initial game state (e.g. you may change the initial player positions, ball position, etc)
    def set_environment(self, data):
        pass

    # The "interprefer" at the picture above
    # Receives the game snapshot and returns the state of whatever you are training (e.g. coordinates of an element, opponents distance, etc) 
    def get_state(self, snapshot: lugo.GameSnapshot):
        pass
    
    # The "interprefer" at the picture above
    # Receives the previous and the new snapshot and should return the evaluation/result. 
    # Note that this method returns `Any`. Depending on what framework/algorithm you use, you may return `{reward, done}` or something
    # else.
    def evaluate(self, previous_snapshot: lugo.GameSnapshot, new_snapshot: lugo.GameSnapshot) -> Any:
        pass
    
    # The "agent" at the picture above, but instead of chosing an action, executes it on the environment (game)
    # Receives the game snapshot and the action chosen by your Reinforcement Learning framework/algorithm and executes the Lubo bot orders that 
    # the action represents (e.g. move forward, kick, etc)
    def play(self, order_set: lugo.OrderSet, snapshot: lugo.GameSnapshot, action) -> lugo.OrderSet:
        pass

```


## If you are not going to use TensorFlow

See a simpler example at [https://github.com/lugobots/lugo4py/tree/master/example/rl](https://github.com/lugobots/lugo4py/tree/master/example/rl)

## Set the environment up
1. Initialize your venv `virtualenv venv --python=python3.9`
2. Activate your virtual environment `source venv/bin/activate`
3. Install the requirements `pip install -r requirements.txt`

## Initializing the trainning

### 1st Step: Start the Game Server

The Game Server must be initialized with the flags `--dev-mode --timer-mode=remote` to allow the game be fully controlled by the trainning session.

You may run the Game Server as a container:

```shell
docker run -p 8080:8080 -p 5000:5000 lugobots/server:latest play --dev-mode --timer-mode=remote
```

Or, you may [download the Game Server binaries at https://hub.docker.com/r/lugobots/server](https://hub.docker.com/r/lugobots/server) that will get a significantly higher performance:


```shell
# Executin the binary on Linux
./lugo_server play --dev-mode --timer-mode=remote
```


### 2nd Step: (optional) Run another bot 

**Note** The initial example is design to training with static bots. Run against another bot if your BotTrainer expected real players.

Run `docker compose -f docker-compose-away-team.yml up` to run against the official bot Level-1 (aka The Dummies Go)

### 3rd Step: Start your trainning script
```shell
python main.py
```

## After training your model

This code is only meant to training your model, it won't build a final bot.


When your model is ready to play, create a bot using the bot template ([The dummies](https://github.com/lugobots/the-dummies-py)) and use the model
to help your bot takes its decisions.
