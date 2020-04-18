import gym
from gym import error, spaces, utils
import numpy as np

'''
================
game explanation
================
this is a game where an agent has to solve a `problem` of the form:
  func(a, b) = (c, d)

b and d are optional, decided by num_inputs and num_outputs respectively.

the agent's action space is its best guess for c[, d]. it gets some rewards for getting it right. it's allowed to not be sure yet of the answer.
there's a constant small negative reward to encourage calculating fast. more reward details can be found below.

each episode randomly initializes a[, b] and calculates the corresponding c[, d] using the `problem` function. then, on each step, the agent does its best guess
(or hopefully in most steps, no guesses but just thinking), until we hit the win/lose conditions explained in win/lose explanation below.

the point is to see if we can learn difficult number crunching algorithms. we'll start simple with addition, but cool future ideas include:
- multiplication : lambda a, b = a * b
- square root : lambda a = int(a ** 0.5)
- comparison : lambda a, b = min(a, b), max(a, b)
- sorting bytes : lambda a = sorted([a & (16**i) for i in range(len(hex(a))-2)]) # just splitting a up into bytes.. need to convert that back to an int oops - shows we might have room to extend this to improve speed and avoid converting back and forth..
- fourier transform : lambda a = fourier_real(a), fourier_imaginary(a) # split a up into bytes like above
- primality testing ?! linear programming ?1 matrix multiplication ?! so many options!!

====================
win/lose explanation
===================
episode finishes if:
- you win because you got the right answer
- you lose because -output_size * 20 > cumulative reward (switched whole answer back and forth ~5 times)
- you lose because 10k steps without positive reward (taking too long for your next answer)
- you lose because 10k * output_size steps total (not sure how you even got here; just in case there's some weird haack)

==================
reward explanation
==================
every step you get reward of:
- base reward of -1.0 / 10000 (get more reward for solving faster, and 10k thinking steps compensates for a right character. we def optimize over less than 10k steps)
- if a character is right this time, and it was right last time, add no reward for this
- if a character is right this time, and it was wrong/unconfident last time, add reward of 1.0
- if the whole answer is right, add reward of 3.0 (and episode is done)
- if a character is unconfident this time, and it was right last time, add reward of -2.0
- if a character is unconfident this time, and it was wrong/unconfident last time, add no reward for this
- if a character is wrong this time, and it was right last time, add reward of -2.0 (will combine with below)
- if a character is ever wrong, no matter what it was last time, add a reward of -(abs(right-wrong)/base + 1.0)

reward exploits to keep in mind:
- going back and forth between right and wrong should be bad, not neutral
- keeping a mostly right answer around for a while does not increase your reward, because then agent would rather do that then find the full answer
- encourage solving it faster, but don't be too harsh about it
- the more wrong an answer, the more harsh, for signal
- if you're really close to the answer and get it wrong, still -1.0 reward. and if it was right before, then -3.0 reward. stability of correct answers is important
- getting the whole thing totally right gives a bigger reward

========================
action space explanation
========================
the "best guess" for the answer. has length output_size. a 0 indicates unconfident in answer for this character, so it's usually ignored in reward (see reward above)

=============================
observation space explanation
=============================
the observation includes the inputs for the problem (e.g, `a` and `b` if solving `a+b=c`), and the previous action.
we include previous action to enable the idea of "difficulty": low difficulty episodes will start off where the initial observation has some of the right answer. 


'''


# this will likely be too slow. probably want to implement it in C later if it works

class MathEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, base: int = 2, num_inputs: int = 2, num_outputs: int = 1, input_size: int = 32, output_size: int = 32, arg_sizes=(1, 16), problem=lambda a, b: a + b, difficulty=1.0, seed=None, verbose=1):
        '''
        base: 2 is binary, 10 is decimal, 16 hex, etc. other args are using this base.
        num_inputs: 1 or 2. sqrt is 1, sum is 2.
        num_outputs: 1 or 2. sqrt is 1, a,b=b,a is 2.
        input_size: the max number of characters input into this problem, adding up size for *all* the inputs. even if num_inputs=2.
        output_size: the max number of characters output from this problem, most likely a function of input_size. even if num_outputs=2.
        arg_sizes: used to figure out how many characters to randomly come up with. minimum of 1, max is what fits in input_size. (min, max) sampled uniformly
        problem: a function which takes in regular numbers and outputs the answer, for ease of trying multiple things out.
        difficulty: (0.0, 1.0] - what percent of random characters to give away for free in initial obs. should create a curriculum. these chars are forgotten after the first observation. weird part about this idea is hidden state of LSTM...
        seed: allows reproducibility in running this env. None means randomize.
        verbose: 0, 1, or 2. 0: no printing. 1: print in reset only. 2: print in step.
        '''

        assert base >= 1, 'invalid base, base={}'.format(base)
        assert num_inputs in [1, 2], 'invalid num_inputs; num_inputs={} type(num_inputs)={}'.format(num_inputs, type(num_inputs))
        assert num_outputs in [1, 2], 'invalid num_outputs; num_outputs={} type(num_outputs)={}'.format(num_outputs, type(num_outputs))
        assert num_inputs != 2 or input_size % 2 == 0, 'input_size must be even if num_inputs is 2; input_size={} num_inputs={}'.format(input_size, num_inputs)
        assert num_outputs != 2 or output_size % 2 == 0, 'output_size must be even if num_outputs is 2; output_size={} num_outputs={}'.format(output_size, num_outputs)
        assert arg_sizes[0] <= arg_sizes[1], 'arg_sizes[0] must be less than arg_sizes[1]; arg_sizes={}'.format(arg_sizes)
        assert type(arg_sizes[0]) == int and type(arg_sizes[1]) == int, 'arg_sizes[0] and [1] must be int; arg_sizes={}'.format(arg_sizes)
        assert arg_sizes[0] >= 1, 'arg_sizes[0] cannot be so low; arg_sizes[0]={}'.format(arg_sizes[0])
        assert arg_sizes[1] * num_inputs <= input_size, 'arg_sizes[1] cannot be so high; arg_sizes[1]={}, num_inputs={}, input_size={}'.format(arg_sizes[1], num_inputs, input_size)

        self.base = base
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.input_size = input_size
        self.output_size = output_size
        self.arg_sizes = arg_sizes
        self.problem = problem
        self.difficulty = difficulty
        self.verbose = verbose

        # TODO: when sampling action_space, it's a big waste to do it completely randomly - much better to maintain some random percent of previous action_space; and perhaps
        # explicitly avoid sampling the confident outputs. is this what trust regions are? I think yes; but I haven't thought about it enough to be sure. look into it later
        self.action_space = spaces.MultiDiscrete([base + 1] * output_size)

        self.observation_space = spaces.MultiDiscrete([base] * input_size + [base + 1] * output_size)

        self.reward_range = (-output_size * 4, output_size + 3)

        # avoid reallocating np arrays to calculate reward each step : don't mess with these... supposed to make things fast? who knows..
        self.step_zeros = np.zeros(output_size, dtype=int)
        self.step_unconfidents = np.zeros(output_size, dtype=int)
        self.step_rights = np.zeros(output_size, dtype=int)
        self.step_wrongs = np.zeros(output_size, dtype=int)
        self.previous_rights = np.zeros(output_size, dtype=int)
        self.previous_nonrights = np.zeros(output_size, dtype=int)
        self.step_reward_rights = np.zeros(output_size, dtype=int)
        self.step_reward_wrong_was_right = np.zeros(output_size, dtype=int)
        self.step_reward_unconfidents = np.zeros(output_size, dtype=int)
        self.step_reward_error = np.zeros(output_size, dtype=np.float32)

        self.observation = np.zeros(self.input_size + self.output_size, dtype=int)

        self.seed(seed)
        self.reset()

    def seed(self, seed=None):
        self.np_random, seed = utils.seeding.np_random(seed)
        return [seed]

    def reset(self):
        # np.array holding the correct set of actions. no 0s here; 1s are padding. used to calculate reward.
        self.correct_action = None
        # this episode's input; i.e, the question, like concat(a, b) if we're doing a+b=c, or like a if we're doing sqrt(a)=c
        self.episode_question = None
        # np.array holding the last set of actions done, which were also provided as the observation last step. initialize this using difficulty. with difficulty=1.0, it starts as all 0s.
        self.last_action = None

        # problem(a[, b]) = c, d
        # ok this code can definitely be written better lmao...
        # the shorter ways of writing this aren't as clear though and more prone to bugs, so I'll leave it like this for now..
        if self.num_outputs == 2 and self.num_inputs == 2:
            input_a = self.np_random.randint(self.base**(self.arg_sizes[0] - 1), self.base**self.arg_sizes[1])
            input_b = self.np_random.randint(self.base**(self.arg_sizes[0] - 1), self.base**self.arg_sizes[1])
            output_c, output_d = self.problem(input_a, input_b)
            if self.verbose:
                print('input_a=', input_a, 'input_b=', input_b, 'output_c=', output_c, 'output_d=', output_d)
            problem_input = input_a * (self.base ** (self.input_size / 2)) + input_b
            problem_output = output_c * (self.base ** (self.output_size / 2)) + output_d
        elif self.num_outputs == 1 and self.num_inputs == 2:
            input_a = self.np_random.randint(self.base**(self.arg_sizes[0] - 1), self.base**self.arg_sizes[1])
            input_b = self.np_random.randint(self.base**(self.arg_sizes[0] - 1), self.base**self.arg_sizes[1])
            output_c = self.problem(input_a, input_b)
            if self.verbose:
                print('input_a=', input_a, 'input_b=', input_b, 'output_c=', output_c)
            problem_input = input_a * (self.base ** (self.input_size / 2)) + input_b
            problem_output = output_c
        elif self.num_outputs == 2 and self.num_inputs == 1:
            input_a = self.np_random.randint(self.base**(self.arg_sizes[0] - 1), self.base**self.arg_sizes[1])
            output_c, output_d = self.problem(input_a)
            if self.verbose:
                print('input_a=', input_a, 'output_c=', output_c, 'output_d=', output_d)
            problem_input = input_a
            problem_output = output_c * (self.base ** (self.output_size / 2)) + output_d
        elif self.num_outputs == 1 and self.num_inputs == 1:
            input_a = self.np_random.randint(self.base**(self.arg_sizes[0] - 1), self.base**self.arg_sizes[1])
            output_c = self.problem(input_a)
            if self.verbose:
                print('input_a=', input_a, 'output_c=', output_c)
            problem_input = input_a
            problem_output = output_c

        input_characters = []
        for i in range(self.input_size):
            next_character = problem_input % self.base
            input_characters.append(next_character)
            problem_input = problem_input // self.base
        self.episode_question = np.array(input_characters, dtype=int)

        output_characters = []
        for i in range(self.output_size):
            # oh i am definitely gonna mess this up somewhere... be really careful about always adding / subtracting 1 here to get the actual answer... 
            next_character = (problem_output % self.base) + 1
            output_characters.append(next_character)
            problem_output = problem_output // self.base
        self.correct_action = np.array(output_characters, dtype=int)

        number_freebies = int(self.output_size * (1.0 - self.difficulty))
        freebies = list(self.np_random.choice(self.output_size, size=number_freebies, replace=False))
        free_characters = []
        for i, character in enumerate(list(self.correct_action)):
            if i in freebies:
                free_characters.append(character)
                # don't give reward for just passing something through, to minimize difference in difficulty settings..
                self.previous_rights[i] = 1
            else:
                # 0 is unconfident
                free_characters.append(0)
                self.previous_rights[i] = 0
        self.last_action = np.array(free_characters, dtype=int)

        self.last_reward = 0.0
        self.episode_total_reward = 0.0
        self.steps = 0
        self.steps_since_positive_reward = 0
        self.done = False

        # self.observation_space = spaces.MultiDiscrete([base] * input_size + [base + 1] * output_size)
        np.concatenate((self.episode_question, self.last_action), out=self.observation)
        if self.verbose:
            print('reset.observation(concat(episode_question[{}], last_action[{}]))='.format(self.episode_question.shape[0], self.last_action.shape[0]), self.observation)
        return self.observation

    def step(self, action):
        # comment this out to make code faster
        assert self.action_space.contains(action)
        if self.done:
            raise Exception('tried to step in an env which was already done')

        # see "reward explanation" at the top to understand the reward calculations...

        done = False

        # base reward
        reward = -1.0 / 10000

        # right mask, in current action
        np.equal(action, self.correct_action, self.step_rights, dtype=int)

        # if a character is right this time, and it was wrong/unconfident last time, add reward of 1.0
        np.subtract(1, self.previous_rights, self.previous_nonrights)
        np.bitwise_and(self.previous_nonrights, self.step_rights, self.step_reward_rights)
        reward_rights = np.sum(self.step_reward_rights)
        if self.verbose > 1:
            print('step.reward_rights=', reward_rights)
        reward += reward_rights

        reward_whole_right = 0.0
        # if the whole answer is right, add reward of 3.0 (and episode is done)
        if np.sum(self.step_rights) == self.output_size:
            # got the right answer; congrats
            reward_whole_right = 3.0
            reward += reward_whole_right
            if self.verbose:
                print('done=True because you won!')
            done = True
        
        if self.verbose > 1:
            print('step.reward_whole_right=', reward_whole_right)

        # unconfident mask, in current action
        np.equal(action, self.step_zeros, out=self.step_unconfidents, dtype=int)

        # if a character is unconfident this time, and it was right last time, add reward of -2.0
        np.bitwise_and(self.step_unconfidents, self.previous_rights, out=self.step_reward_unconfidents)
        # step_unconfidents is a mask of all unconfidents in current step
        # step_reward_unconfidents is 1 iff above condition holds else 0
        reward_unconfidents = np.sum(self.step_reward_unconfidents) * -2.0
        if self.verbose > 1:
            print('step.reward_unconfidents=', reward_unconfidents)
        reward += reward_unconfidents

        # wrong mask, in current action
        np.add(self.step_unconfidents, self.step_rights, out=self.step_wrongs)
        np.subtract(1, self.step_wrongs, self.step_wrongs)

        # if a character is wrong this time, and it was right last time, add reward of -2.0 (will combine with below)
        np.bitwise_and(self.step_wrongs, self.previous_rights, out=self.step_reward_wrong_was_right)
        reward_wrong_was_right = np.sum(self.step_reward_wrong_was_right) * -2.0
        if self.verbose > 1:
            print('step.reward_wrong_was_right=', reward_wrong_was_right)
        reward += reward_wrong_was_right

        # if a character is ever wrong, no matter what it was last time, add a reward of -(abs(right-wrong)/base + 1.0)
        np.subtract(self.correct_action, action, out=self.step_reward_error)
        np.divide(self.step_reward_error, self.base, out=self.step_reward_error)
        np.abs(self.step_reward_error, out=self.step_reward_error)
        np.add(1.0, self.step_reward_error, out=self.step_reward_error)
        np.multiply(self.step_reward_error, self.step_wrongs, out=self.step_reward_error)
        reward_error = -1.0 * np.sum(self.step_reward_error)
        if self.verbose > 1:
            print('step.reward_error=', reward_error)
        reward += reward_error

        self.episode_total_reward += reward
        # you lose because -output_size * 20 > cumulative reward (switched whole answer back and forth ~5 times)
        if self.output_size * -20 >= self.episode_total_reward:
            if self.verbose:
                print('done=True because -output_size * 20 > cumulative reward')
            done = True

        # you lose because 10k steps without positive reward (taking too long for your next answer)
        if reward < 0:
            self.steps_since_positive_reward += 1
            if self.steps_since_positive_reward >= 10000:
                if self.verbose:
                    print('done=True because 10k steps without +ve reward')
                done = True
        else:
            self.steps_since_positive_reward = 0

        # you lose because 10k * output_size steps total (not sure how you even got here; just in case there's some weird haack)
        self.steps += 1
        if self.steps >= 10000 * self.output_size:
            if self.verbose:
                print('done=True because 10k * output_size steps total')
            done = True

        # set previous to current but make sure to keep them both as np arrays; current is now dirty.. don't reuse it lol
        self.previous_rights, self.step_rights = self.step_rights, self.previous_rights

        self.last_action = action
        self.last_reward = reward
        self.done = done

        # self.observation_space = spaces.MultiDiscrete([base] * input_size + [base + 1] * output_size)
        np.concatenate((self.episode_question, self.last_action), out=self.observation)

        if self.verbose > 1 or (self.verbose and done):
            print('step.observation(concat(episode_question[{}], last_action[{}]))='.format(self.episode_question.shape[0], self.last_action.shape[0]), self.observation)
            print('reward={}; done={}'.format(reward, done))

        return self.observation, reward, done, {}

    def render(self, mode='human'):
        print('last_action={}\ncorrect_action={}\nepisode_question={}\nlast_reward={}\nepisode_total_reward={}\nsteps={}\nsteps_since_positive_reward={}'.format(
            self.last_action, self.correct_action, self.episode_question, self.last_reward, self.episode_total_reward, self.steps, self.steps_since_positive_reward))

    def close(self):
        pass
