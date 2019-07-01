# -------------------------
# Project: Deep Q-Learning on Flappy Bird
# Author: Flood Sung
# Date: 2016.3.21
# -------------------------

import cv2
import sys
sys.path.append("game/")
import wrapped_flappy_bird as game
from BrainDQN_Nature import BrainDQN
import numpy as np

# 辅助函数：将80*80大小的图像进行灰度二值化处理
def preprocess(observation):
	observation = cv2.cvtColor(cv2.resize(observation, (80, 80)), cv2.COLOR_BGR2GRAY)
	ret, observation = cv2.threshold(observation,1,255,cv2.THRESH_BINARY)
	return np.reshape(observation,(80,80,1))

# 主函数：初始化DQN和游戏，并开始游戏进行训练
def playFlappyBird():
	# Step 1: init BrainDQN
	actions = 2
	brain = BrainDQN(actions)
	# Step 2: init Flappy Bird Game
	flappyBird = game.GameState()
	# Step 3: play game

	# Step 3.1: 得到初始状态
	action0 = np.array([1,0])  # do nothing
	observation0, reward0, terminal = flappyBird.frame_step(action0)
	observation0 = cv2.cvtColor(cv2.resize(observation0, (80, 80)), cv2.COLOR_BGR2GRAY)
	ret, observation0 = cv2.threshold(observation0,1,255,cv2.THRESH_BINARY)
	brain.setInitState(observation0)

	# Step 3.2: run the game
	while 1!= 0:
		# 得到一个动作
		action = brain.getAction()
		# 通过游戏接口得到动作后返回的下一帧图像、回报和终止标志
		nextObservation,reward,terminal = flappyBird.frame_step(action)
		# 图像灰度二值化处理
		nextObservation = preprocess(nextObservation)
		# 将动作后得到的下一帧图像放入到新状态newState，然后将新状态、当前状态、动作、回报和终止标志放入都游戏回放记忆序列
		brain.setPerception(nextObservation,action,reward,terminal)

def main():
	playFlappyBird()

if __name__ == '__main__':
	main()