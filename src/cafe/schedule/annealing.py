import numpy as np
from math import exp
import random
from typing import Literal
from abc import ABC, abstractmethod
import pandas as pd
from pathlib import Path

import matplotlib.pyplot as plt

from cafe.schedule.turns import TurnList
from .base import SchedulerBase, Sheets
from .turns import Turn
from .linear_prog import LinearProgSched


class TempStrategy(ABC):
    num_steps: int

    @abstractmethod
    def get_temp(self, i: int) -> float:
        pass

class TempExpDecay(TempStrategy):
    def __init__(self, num_steps, t1, decay_point, t2=0):
        self.num_steps = num_steps
        self.t1 = t1
        self.t2 = t2
        self.decay_point = decay_point
        self.decay = self.decay_point * self.num_steps

    def get_temp(self, i):
        return (self.t1 - self.t2) * exp(-i/self.decay) + self.t2

class TempScalingLaw(TempStrategy):
    def __init__(self, num_steps, t1, exponent):
        self.num_steps = num_steps
        self.t1 = t1
        self.exponent = exponent

    def get_temp(self, i):
        return self.t1 * (1 + i)**(-self.exponent)

class TempConst(TempStrategy):
    def __init__(self, t, num_steps):
        self.t = t
        self.num_steps = num_steps

    def get_temp(self, i):
        return self.t

class SystemParams:
    def __init__(self, k_disp, k_border, k_fix_people_gt, k_fix_people_sm, k_fix_people_sm_peak, k_lunch, k_continuos_lunch,
        k_no_people, k_work_load, k_overflow_work_load, k_pref, temp_strat: TempStrategy | list[TempStrategy], lunch_min_free_turns=2):
        '''
        Parameters for the annealing schedule.
        
        Parameters
        ----------
        k_disp: 
            Coefficient to not violate availability.

        k_border: 
            Coefficient to minimize granular schedules.
            
        k_fix_people_gt: 
            Coefficient to penalize more people than the target people number per turn.
            
        k_fix_people_sm: 
            Coefficient to penalize less people than the target people number per turn.
        
        k_fix_people_sm_peak: 
            Coefficient to penalize less people than the target people number for peak turns.
        
        k_lunch: 
            Coefficient to give lunch hours.
        
        k_continuos_lunch: 
            Coefficient to give continuos lunch hours.

        k_no_people: 
            Coefficient to penalize turns with no people.

        k_work_load: 
            Coefficient to distribute hours among people fairly.

        k_overflow_work_load: 
            Coefficient to penalize when a people receives more turns than asked.

        k_pref: 
            Coefficient for preferring hours in the preference.
        
        temp_strat: 
            Temperature decay strategy. It can be a list of strategies, is this
            case every strategy is executed in order.

        lunch_min_free_turns: 
            Number of turns everyone needs for lunch.
        '''
        self.k_disp = k_disp
        self.k_border = k_border
        self.k_fix_people_gt = k_fix_people_gt
        self.k_fix_people_sm = k_fix_people_sm
        self.k_fix_people_sm_peak = k_fix_people_sm_peak
        self.k_no_people = k_no_people
        self.k_lunch = k_lunch
        self.k_continuos_lunch = k_continuos_lunch
        self.k_work_load = k_work_load 
        self.k_overflow_work_load = k_overflow_work_load
        self.k_pref = k_pref
        self.lunch_min_free_turns = lunch_min_free_turns 
        self.temp_strat = temp_strat 


class ScheduleSystem:
    def __init__(self, 
        x: np.ndarray, pref: np.ndarray, disp: np.ndarray, target_work_load: np.ndarray,
        turn_capacity: np.ndarray, weights: np.ndarray, 
        params: SystemParams, lunch_ids, peak_ids,
        ):
        self.pref = pref
        self.disp = disp
        self.params = params
        self.turn_capacity = turn_capacity
        
        self.lunch_ids = lunch_ids
        self.peak_ids = peak_ids
        self.weights = weights
        self.x = x

        disp_work_load = np.sum(disp * weights[None, ...], axis=(1, 2))

        total_wl = (self.turn_capacity * self.weights).sum()
        num_p = x.shape[0]
        avg_wl = total_wl / num_p 
        self.target_work_load = np.empty_like(target_work_load, dtype=float)
        self.target_work_load[target_work_load == None] = avg_wl * 10
        self.target_work_load[target_work_load != None] = target_work_load[target_work_load != None]

        mask = self.target_work_load > disp_work_load
        self.target_work_load[mask] = disp_work_load[mask]

        target_sorted = np.sort(self.target_work_load)
        for id, t in enumerate(target_sorted):
            total_asked = target_sorted[:id].sum() + target_sorted[id:].size * t
            if total_asked > total_wl:
                self.max_work_load = (total_wl - target_sorted[:id].sum()) / target_sorted[id:].size
                break
        else:
            self.max_work_load = self.target_work_load.max()

        self.target_work_load[self.target_work_load > self.max_work_load] = self.max_work_load


        self.disp_mask = 1 - self.disp

        num_masks = len(self.lunch_ids) - self.params.lunch_min_free_turns + 1
        self.lunch_masks = np.zeros((num_masks, len(self.lunch_ids)))
        for i in range(num_masks):
            self.lunch_masks[i, i:i+self.params.lunch_min_free_turns] = 1

        self.sum_pref = np.sum(self.pref, axis=(1, 2))
        self.k_pref = self.sum_pref.min()  / self.sum_pref

        self.k_sm_fix_people_matrix = np.full((self.x.shape[1:]), self.params.k_fix_people_sm, dtype=int)
        for peak_id in self.peak_ids:
            self.k_sm_fix_people_matrix[peak_id, :] = self.params.k_fix_people_sm_peak

        self.problems = {}

    def max_pref_energy(self):
        return (np.sum(np.abs(self.x - self.pref), axis=(1, 2)) * self.k_pref).sum()
    
    def pref_energy(self):
        return - self.params.k_pref * (self.x * self.pref).sum()
    
    def work_load_energy(self):
        g = np.sum(self.x * self.weights[None, ...], axis=(1, 2))

        # lower_max = g < self.max_work_load
        lower_max = g < self.target_work_load
        bigger_max = ~lower_max

        k = self.params.k_work_load
        k_of = self.params.k_overflow_work_load
        g_max = self.max_work_load
        g_t = self.target_work_load[bigger_max]

        smaller_term = k/2 * ((g[lower_max] - g_max)**2).sum()
        bigger_term = k_of/2 * ((g[bigger_max] - g_t)**2).sum() + k/2 * ((g_t - g_max)**2).sum()

        return smaller_term + bigger_term
    
    def disp_energy(self):
        return np.sum(np.abs(self.x - self.disp) * self.disp_mask) * self.params.k_disp

    def boarder_energy(self):
        return np.sum(np.abs(self.x[:, 1:, :] - self.x[:, :-1, :])) * self.params.k_border

    def fix_people_energy(self):
        p_num = self.x.sum(axis=0)
        p_diff = p_num - self.turn_capacity
        gt_mask = p_diff > 0

        e1 = self.params.k_fix_people_gt * p_diff[gt_mask].sum() 
        e2 = (np.abs((p_diff * self.k_sm_fix_people_matrix)[~gt_mask])**1).sum()

        e3 = (p_num[self.turn_capacity != 0] == 0).sum() * self.params.k_no_people

        # return self.params.k_fix_people/2 * ((p_num - self.people_number)**2).sum()
        return e1 + e2 + e3

    def lunch_energy(self):
        min_free_turns = self.params.lunch_min_free_turns

        x_lunch = self.x[:, self.lunch_ids, :]

        has_continuos_lunch = np.full((self.x.shape[0], self.x.shape[2]), False)
        for mask in self.lunch_masks:
            continuos_lunch = (x_lunch * mask[..., None]).sum(axis=1) < 1
            has_continuos_lunch |= continuos_lunch

        lunch_free = len(self.lunch_ids) - x_lunch.sum(axis=1)
        lunch_free[lunch_free >= min_free_turns] = min_free_turns

        lunch_energies = (min_free_turns - lunch_free)**2 * self.params.k_lunch
        
        return lunch_energies.sum() + (~has_continuos_lunch).sum() * self.params.k_continuos_lunch

    def energy(self):
        return self.work_load_energy() + self.pref_energy() + self.disp_energy() + self.boarder_energy() + self.fix_people_energy() + self.lunch_energy()
    
    def run(self):
        strats: list[TempStrategy] = self.params.temp_strat
        if not isinstance(strats, list):
            strats = [strats]

        energy = [self.energy()]
        for strat in strats:
            for i in range(strat.num_steps):
                temp = strat.get_temp(i)

                for idx in np.ndindex(self.x.shape):
                    e1 = self.energy()
                    self.x[idx] = 1 - self.x[idx]
                    e2 = self.energy()

                    de = e2 - e1
                    
                    accept = False
                    if de > 0:
                        if random.random() < exp(-1/temp * de):
                            accept = True
                    else:
                        accept = True

                    if not accept:                
                        self.x[idx] = 1 - self.x[idx]

                energy.append(self.energy())

        return energy
    
    def get_problems(self, mappers):
        problems = {}
        problems["lunch"] = self.launch_problem(mappers)
        problems["availability"] = self.disp_problem(mappers)
        problems["turn_capacity"] = self.number_people_problem(mappers)
        return problems

    def launch_problem(self, mappers):
        x_lunch = self.x[:, self.lunch_ids, :]
        has_continuos_lunch = np.full((self.x.shape[0], self.x.shape[2]), False)
        for mask in self.lunch_masks:
            continuos_lunch = (x_lunch * mask[..., None]).sum(axis=1) < 1
            has_continuos_lunch |= continuos_lunch

        # problems = lunch_free < self.params.lunch_min_free_turns
        problems = ~has_continuos_lunch

        info = []
        for p_id, p in mappers.id_to_person.items():
                for d_id, d in mappers.id_to_week_day.items():
                    if problems[p_id, d_id]:
                        info.append((p, d))
        return pd.DataFrame(info, columns=["Pessoa", "Dia"])
    
    def disp_problem(self, mappers):
        problems = np.logical_and(self.x != self.disp, self.disp == 0)
        
        info = {}
        for p_id, p in mappers.id_to_person.items():
            for t_id, t in mappers.id_to_turn.items():
                for d_id, d in mappers.id_to_week_day.items():
                    if problems[p_id, t_id, d_id]:
                        if p not in info:
                            info[p] = []
                        info[p].append((d, t))
        
        info_df = pd.DataFrame.from_dict(info, orient='index').transpose()

        return info_df

    def number_people_problem(self, mappers):
        count = np.sum(self.x, axis=0)
        problems = count != self.turn_capacity

        info = []
        for t_id, t in mappers.id_to_turn.items():
            for d_id, d in mappers.id_to_week_day.items():
                if problems[t_id, d_id]:
                    info.append((d, t, int(count[t_id, d_id]), int(self.turn_capacity[t_id, d_id])))
        return pd.DataFrame(info, columns=["Dia", "Turno", "Atual", "Alvo"])

def sheets_to_matrix(sheet, sheets: Sheets):
        num_people = len(sheets.people)
        num_work_turns = len(sheets.turns)
        num_week_days = len(sheets.week_days)
        mappers = sheets.mappers
        matrix = np.zeros(shape=(num_people, num_work_turns, num_week_days), dtype=int)
        for p, days in sheet.items():
            p_id = mappers.person_to_id[p]
            for d, p_turns in days.items():
                d_id = mappers.week_day_to_id[d]
                for t in p_turns:
                    t_id = mappers.turn_to_id[str(Turn(t))]
                    matrix[p_id, t_id, d_id] = 1
        
        return matrix

class AnnealingSched(SchedulerBase):
    def __init__(self, 
        sheets: Sheets, params: SystemParams, 
        lunch_turns: TurnList, peak_turns: TurnList, 
        init_state: Literal["random", "linear_sched"]="random", turns_weights: dict=None):
        '''
        Scheduler based on the annealing algorithm.

        Parameters
        ----------
        sheets:
            Dada from sheets (availability, preference, etc). See docs for `Sheets` for
            more info.
        
        params:
            Parameters used by the annealing algorithm. See docs for `SystemParams` for more info.
        
        lunch_turns:
            Turns that one can use to lunch.
        
        peak_turns:
            Turns wih high priority than the other.
        
        init_state:
            Init state used by the annealing algorithm.
            - "random": random state
            - "linear_sched": init state give by the linear scheduler.
        
        turns_weights:
            Turns weight, a higher weight mean the turn count more hours than a turn with
            a lower weight. It should be given as a dictionary with the fallowing key-value pairs:
            - key: (week_day, turn)
            - value: turn weight  
        '''
        super().__init__(sheets, turns_weights)
        self.params = params
        
        self.lunch_ids = [self.sheets.mappers.turn_to_id[str(t)] for t in lunch_turns]
        self.peak_ids = [self.sheets.mappers.turn_to_id[str(t)] for t in peak_turns]
        self.init_state = init_state

        self.anneal_system: ScheduleSystem = None
        self.energy: np.ndarray = None

    def get_linear_scheduler(self):
        sched = LinearProgSched(
            sheets=self.sheets,
            turns_weights=self.turns_weights,
        )

        for day in self.sheets.turn_capacity.columns:
            turn_3 = [turn for turn, value in self.sheets.turn_capacity[day].items() if value == 3]
            if len(turn_3) > 0:
                sched.add_fix_people_turns(
                    turns=[turn for turn, value in self.sheets.turn_capacity[day].items() if value == 3],
                    capacity=3,
                    days=[day],
                )

        return sched
    
    def schedule_to_x_mat(self, schedule):
        num_people = len(self.sheets.people)
        num_work_turns = len(self.sheets.turns)
        num_week_days = len(self.sheets.week_days)
        mappers = self.sheets.mappers
        x_mat = np.zeros(shape=(num_people, num_work_turns, num_week_days), dtype=int)
        for d, day_sched in schedule.items():
            for t, people in day_sched.items():
                for p in people:
                    d_id = mappers.week_day_to_id[d]
                    t_id = mappers.turn_to_id[str(Turn(t))]
                    p_id = mappers.person_to_id[p]

                    x_mat[p_id, t_id, d_id] = 1
        return x_mat
    
    def x_mat_to_schedule(self, x):
        week_days = self.sheets.week_days
        turns = self.sheets.turns
        mappers = self.sheets.mappers

        schedule = {d: {str(t): [] for t in turns} for d in week_days}
        for (p_id, t_id, d_id) in np.ndindex(x.shape):
            p = mappers.id_to_person[p_id]
            t = mappers.id_to_turn[t_id]
            d = mappers.id_to_week_day[d_id]
            if x[p_id, t_id, d_id]:
                schedule[d][t].append(p)

        return schedule

    def generate(self):
        if self.init_state == "linear_sched":
            linear_sched = self.get_linear_scheduler()
            
            linear_sched.generate()
            # print("Problemas (Linear):")
            # print(self.linear_sched.problems.availability)
            
            x_mat = self.schedule_to_x_mat(linear_sched.schedule)
        elif self.init_state == "random":
            shape = (len(self.sheets.people), len(self.sheets.turns), len(self.sheets.week_days))
            x_mat = np.random.randint(0, 2, size=shape, dtype=int)
        elif isinstance(self.init_state, dict):
            x_mat = self.schedule_to_x_mat(self.init_state)
        else:
            raise ValueError("Init State not valid! It should be a schedule (dict) or a valid string.")

        aval_mat = sheets_to_matrix(self.sheets.avail, self.sheets)
        pref_mat = sheets_to_matrix(self.sheets.pref, self.sheets)

        mappers = self.sheets.mappers

        shape = (len(self.sheets.turns), len(self.sheets.week_days))
        turn_capacity_mat = np.empty(shape, dtype=int)
        for t, row in self.sheets.turn_capacity.iterrows():
            for day, value in row.items():
                turn_capacity_mat[mappers.turn_to_id[t], mappers.week_day_to_id[day]] = value

        target_work_load_mat = np.full(x_mat.shape[0], None)
        for name, wl in self.sheets.target_work_load.items():
            target_work_load_mat[mappers.person_to_id[name]] = wl

        turns_wights_mat = np.ones_like(turn_capacity_mat)
        for (day, turn), w in self.turns_weights.items():
            turns_wights_mat[mappers.turn_to_id[str(turn)], mappers.week_day_to_id[day]] = w

        self.anneal_system = ScheduleSystem(
            x_mat, pref_mat, aval_mat, target_work_load_mat, 
            turn_capacity_mat, turns_wights_mat, 
            self.params, self.lunch_ids, self.peak_ids,
        )
        self.energy = self.anneal_system.run()

        self.schedule = self.x_mat_to_schedule(self.anneal_system.x)

    def show_problems(self, path=None):
        "Show schedule problems. If `path` is a folder path, problems will be saved inside `path`."
        if path is None:
            print("\nProblemas:")
        
        # print(sched.check_availability(sched.schedule, sched.availability))
        for p_name, p_info in self.anneal_system.get_problems(self.sheets.mappers).items():
            if path is None:
                print(f"{p_name}:")
                print(p_info)
                print()
            else:
                p_info.to_csv(Path(path) / f"{p_name}.csv", index=False)

    def show_energy(self):
        plt.plot(self.energy, marker='o', linestyle='-')
        plt.title("Energy Over Iterations")
        plt.xlabel("Iteration")
        plt.ylabel("Energy")
        plt.grid(True)
        plt.show()

    def save_work_load(self, path):
        target_work_load = {}
        for p, id in self.sheets.mappers.person_to_id.items():
            target_work_load[p] = self.anneal_system.target_work_load[id]
        return super().save_work_load(path, target_work_load)

    # def save_work_load(self, path):
    #     target_work_load = {}
    #     for p, id in mappers.person_to_id.items():
    #         target_work_load[p] = anneal_system.target_work_load[id]
    #     self.save_work_load("work_load.csv", target_work_load)


if __name__ == "__main__":
    pass