# http://mnemstudio.org/path-finding-q-learning-tutorial.htm
import numpy as np;

R = np.array([
	    [-1,     -1,     -1,    -1,     0,      -1],
	    [-1,     -1,     -1,     0,     -1,     100],
	    [-1,     -1,     -1,     0,     -1,     -1],
	    [-1,       0,      0,     -1,    0,      -1],
	    [ 0,      -1,     -1,     0,     -1,     100], 
	    [ -1,      0,      -1,   -1,     0,      100],
]); # row => state, column => action

Q = np.zeros((6,6))
gamma = 0.8;
learning_rate = .05;

print R;

is_goal = False;
while is_goal <> True:
	is_goal = True;
	for state in range(6):
		for action in range(6):
			if R[state][action] == -1:
				continue;
			Q_temp = Q[state, action];
			next_state = action;
			
			R_target = R[next_state,] <> -1; #R(next_state,:) ==> Meaning:  all actions of next_state row.
			Q_values_for_R_true = Q[next_state, R_target]; # Q values in that row when R values are True as same indexes.
			
			# Q[state, action] = R[state, action] + gamma * np.max(Q_values_for_R_true); # ==> When learning rate is 1.
			# https://en.wikipedia.org/wiki/Q-learning
			Q[state, action] = Q[state, action] + learning_rate * (R[state, action] + gamma * np.max(Q_values_for_R_true) - Q[state, action]);
			
			# check whether Q state is updated or not.
			if Q_temp <> Q[state, action]:
				 is_goal = False;

#normalize
Q = Q / np.max(Q.flatten(1)) * 100;
print Q;


             








