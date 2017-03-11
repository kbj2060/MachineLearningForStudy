Chapter 01-02. 수업 소개 &
 OpenAI Gym
========

Sung kim님의 [유뷰트 강의](https://www.youtube.com/watch?v=dZ4vw6v3LcA&index=1&list=PLlMkM4tgfjnKsCWav-Z2F-MMFRx-2gMGG&t=2s)에서 공부한 내용을 정리하고자 한다. 강화 학습은 공부할 수 있는 기회가 별로 없기 때문에 소중한 자료이기도 하다. 

대부분이 알듯이 게임 분야에만 특화된 것이 아니라 회계, 비즈니스, 메디아 등 여러 분야에 투입될 수 있다.

예상 독자는 의지만 있으면 누구나 가능하다고 하니 다행이다. tensorflow와 python 으로 예제를 수행하기 때문에 나와 매우 적합하다고 판단하여 이 강의를 선택하게 됐다.

[OpenAI Gym](https://gym.openai.com/)은 강화학습에서 환경 설정을 해주는 프레임 워크이다. 

특정한 환경 안에서 Action과 State를 가지고 Agent가 학습하게 된다. 

```python
"""
 * Reinforcement Learning
"""
import gym
import tensorflow as tf
from gym.envs.registration import register
import sys, tty, termios 

class _Getch:
    def __call__(self):
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(sys.stdin.fileno())
            ch = sys.stdin.read(3)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN,old_settings)
        return ch

inkey = _Getch()

LEFT = 0
DOWN = 1
RIGHT = 2
UP = 3

arrow_keys = {
    '\x1b[A' : UP,
    '\x1b[B': DOWN,
    '\x1b[C': RIGHT,
    '\x1b[D': LEFT,
}


register(id = 'FrozenLake-v3',
         entry_point = 'gym.envs.toy_text:FrozenLakeEnv',
         kwargs = {'map_name' : '4x4', 'is_slippery' : False}
)

env = gym.make('FrozenLake-v3')
env.render()

while True:
    key = inkey()
    if key not in arrow_keys.keys():
        print "Game aborted!"
        break
    
    action = arrow_keys[key]
    state, reward, done, info = env.step(action)
    env.render()
    print "State : :", state, "Action : ",action,"Reward : ",reward,"Info : ",info

    if done:
        print "Finished with reward", reward
        break
```

FrozenLake-v3의 게임 환경에서 사용자의 키보드를 읽어 reward를 받는 스크립트이다. 기계가 학습하는 것이 아니라 그냥 게임 같은 느낌이다. 이런 환경 위에서 기계가 reward, state, info, action 정보를 받으면서 학습하는 것이 q-learning 이다.