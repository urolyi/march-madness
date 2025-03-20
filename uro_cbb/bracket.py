"""Module to help simulating bracket win probabilities using Kaggle 2025 submission data."""

import abc
import dataclasses
import functools

import pandas as pd


@dataclasses.dataclass(slots=True)
class Team:
    _id: int
    name: str

    @classmethod
    def from_id(cls, id: int, team_df: pd.DataFrame):
        """team_df: maps team_id to team_name"""
        id_map = team_df[team_df["TeamID"] == id]
        assert len(id_map) == 1, f"id_map: {id_map}"
        return cls(id, id_map.TeamName.item())

    def __hash__(self):
        return self._id


def get_win_prob(team1: Team, team2: Team, win_probs_df: pd.DataFrame):
    """win_prob_df: is in kaggle 2025 submission format of two columns: ID and Pred.
    ID column is formatted as 2025_{team1_id}_{team2_id} and Pred is the win probability of team1.
    """
    if team1._id < team2._id:
        prob = win_probs_df[win_probs_df["ID"] == f"2025_{team1._id}_{team2._id}"]
        assert len(prob) == 1, f"prob: {prob}"
        return prob.Pred.item()
    else:
        prob = win_probs_df[win_probs_df["ID"] == f"2025_{team2._id}_{team1._id}"]
        assert len(prob) == 1, f"prob: {prob}"
        return 1 - prob.Pred.item()


class AbstractGame(abc.ABC):
    @property
    @abc.abstractmethod
    def win_prob(self) -> float:
        pass

    @property
    @abc.abstractmethod
    def all_win_probs(self) -> dict[Team, float]:
        pass

    @property
    @abc.abstractmethod
    def teams(self) -> Team:
        pass


@dataclasses.dataclass
class Game(AbstractGame):
    team1: Team
    team2: Team

    @functools.cached_property
    def win_prob(self) -> float:
        return get_win_prob(self.team1, self.team2)

    @functools.cached_property
    def winner(self) -> Team:
        return self.team1 if self.win_prob > 0.5 else self.team2

    @property
    def teams(self) -> list[Team]:
        return [self.team1, self.team2]

    @property
    def all_win_probs(self) -> dict[Team, float]:
        return {self.team1: self.win_prob, self.team2: 1 - self.win_prob}


@dataclasses.dataclass
class HyperGame:
    prev_game1: AbstractGame
    prev_game2: AbstractGame

    @property
    def deterministic_win_prob(self):
        game1_winner = self.prev_game1.winner
        game2_winner = self.prev_game2.winner
        return Game(game1_winner, game2_winner).win_prob

    @functools.cached_property
    def winner(self):
        game1_winner = self.prev_game1.winner
        game2_winner = self.prev_game2.winner
        game = Game(game1_winner, game2_winner)
        if game.win_prob > 0.5:
            return game.team1
        return game.team2

    @functools.cached_property
    def teams(self):
        return self.prev_game1.teams + self.prev_game2.teams

    @functools.cached_property
    def all_win_probs(self) -> dict[Team, float]:
        win_probs = {}
        for team in self.prev_game1.all_win_probs:
            for team2 in self.prev_game2.all_win_probs:
                prob_game_happens = (
                    self.prev_game1.all_win_probs[team]
                    * self.prev_game2.all_win_probs[team2]
                )
                win_probs[team] = (
                    win_probs.get(team, 0)
                    + get_win_prob(team, team2) * prob_game_happens
                )
                win_probs[team2] = (
                    win_probs.get(team2, 0)
                    + get_win_prob(team2, team) * prob_game_happens
                )
        return win_probs
