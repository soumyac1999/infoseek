import numpy as np


class CardEnv(object):
    def __init__(self, trial_type, sampling_rew=None, win_rew=None, lose_rew=None):
        try:
            sampling_rew[0]
            self.sampling_rew = sampling_rew
        except:
            self.sampling_rew = [0, -10, -15, -20]

        if win_rew:
            self.win_rew = win_rew
        else:
            self.win_rew = 60

        if lose_rew:
            self.lose_rew = lose_rew
        else:
            self.lose_rew = -50
        self.trial_type = trial_type

    def guess(self, action):
        """
        0 -> guess A, 1 -> guess B
        """
        assert action in [0, 1]
        if action == 0:
            reward = self.win_rew if self.correct_row == 0 else self.lose_rew
        elif action == 1:
            reward = self.win_rew if self.correct_row == 1 else self.lose_rew
        return None, reward, True

    def step1(self, action):
        """
        Action can be sample/guessA/guessB
        """
        assert action in [0, 1, 2]
        if action == 0:
            card_val = 0
            if self.aa:
                card_val = self.flip_card(1)
            else:
                card_val = self.flip_card(2)
            return card_val, self.sampling_rew[self.cur_step], False
        else:
            return self.guess(action - 1)

    def step2(self, action):
        """
        aa: action sample/guessA/guessB
        ab: action sampleA/sampleB/guessA/guessB
        """
        if self.aa == 1:
            assert action in [0, 1, 2]
        else:
            assert action in [0, 1, 2, 3]
        if self.aa == 1:
            if action == 0:
                return self.flip_card(2), self.sampling_rew[self.cur_step], False
            else:
                return self.guess(action - 1)
        else:
            if action == 0:
                return self.flip_card(1), self.sampling_rew[self.cur_step], False
            elif action == 1:
                return self.flip_card(3), self.sampling_rew[self.cur_step], False
            else:
                return self.guess(action - 2)

    def step3(self, action):
        """
        action - sample / guessA / guessB
        """
        assert action in [0, 1, 2]
        if action == 0:
            return (
                self.flip_card(np.where(self.seen == 0)[0][0]),
                self.sampling_rew[self.cur_step],
                False,
            )
        else:
            return self.guess(action - 1)

    def flip_card(self, loc):
        assert self.seen[loc] != 1
        self.seen[loc] = 1
        return self.cards[loc]

    def step(self, action):
        if self.cur_step == 1:
            ret = self.step1(action)
        elif self.cur_step == 2:
            ret = self.step2(action)
        elif self.cur_step == 3:
            ret = self.step3(action)
        elif self.cur_step == 4:
            ret = self.guess(action)
        self.cur_step += 1
        return (*ret, {"step": self.cur_step})

    def reset(self):
        self.cards = np.random.randint(1, 11, 4)
        self.seen = np.array([0, 0, 0, 0])
        self.cur_step = 1
        self.aa = np.random.randint(2)
        if self.trial_type == 0:
            self.correct_row = np.argmax(self.cards) // 2
        elif self.trial_type == 1:
            self.correct_row = np.argmin(self.cards) // 2
        elif self.trial_type == 2:
            self.correct_row = np.argmax(self.cards.reshape(2, 2).sum(1))
        elif self.trial_type == 3:
            self.correct_row = np.argmin(self.cards.reshape(2, 2).sum(1))
        elif self.trial_type == 4:
            self.correct_row = np.argmax(self.cards.reshape(2, 2).prod(1))
        elif self.trial_type == 5:
            self.correct_row = np.argmin(self.cards.reshape(2, 2).prod(1))
        return (self.flip_card(0), self.aa)

    def reset_with_cards(self, aa, cards):
        self.cards = cards
        self.seen = np.array([0, 0, 0, 0])
        self.cur_step = 1
        self.aa = aa
        if self.trial_type == 0:
            self.correct_row = np.argmax(self.cards) // 2
        elif self.trial_type == 1:
            self.correct_row = np.argmin(self.cards) // 2
        elif self.trial_type == 2:
            self.correct_row = np.argmax(self.cards.reshape(2, 2).sum(1))
        elif self.trial_type == 3:
            self.correct_row = np.argmin(self.cards.reshape(2, 2).sum(1))
        elif self.trial_type == 4:
            self.correct_row = np.argmax(self.cards.reshape(2, 2).prod(1))
        elif self.trial_type == 5:
            self.correct_row = np.argmin(self.cards.reshape(2, 2).prod(1))
        return (self.flip_card(0), self.aa)
