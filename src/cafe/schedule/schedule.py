from pulp import LpProblem, LpVariable, lpSum, LpMaximize, LpStatus, PULP_CBC_CMD
import pandas as pd
from collections import namedtuple
import logging
import numpy as np
from pathlib import Path

from . import reader
from .turns import TurnList
from .constants import *

# Define a logger for this module
logger = logging.getLogger(__name__)

FixPeopleHours = namedtuple('FixPeopleHours', ['turns', 'people_number'])

class ScheduleProblems:
    def __init__(self):
        self.availability: dict = None

class Scheduler:
    def __init__(self, open_turn: str, close_turn: str, 
        people_boost:dict[str, float]=None, desired_work_load: dict[str, float]=None,
        max_open_per_people=2, max_close_per_people=2, 
        max_load_per_day=None, max_load_per_week=None, min_load_per_week=None,
        default_people_number_per_turn=2,
    ):
        '''
        System to generate a schedule. After creating this object, one should call `self.generate()`
        to create the schedule.

        OBS: 
            A turn is a string with the following structure: "HH:MM-HH:MM".
            Example: "07:30-08:00" 

        Parameters
        ----------
        open_turn:
            When the cafe opens
        
        close_turn:
            When the cafe closes

        people_boost:
            Map between people and its boost in the objective function. The name
            used should be the same as the one in the preference/availability sheets.
            If the boost is greater/less than 1, for a give person, this person will tend to gain more/less
            work hours. Example boosting work hours for Marcos:
            
            >>> shed = Scheduler(people_boost={"Marcos": 1.5})

        desired_work_load:
            Map between people and its desired total work load in the week in hours.
        
        max_open_per_people:
            Maximum number of times a given person open the cafe.
        
        max_close_per_people:
            Maximum number of times a given person close the cafe.
        
        max_load_per_day:
            Maximum number of hours a given person works per day. If None there is no limit.

        max_load_per_week, min_load_per_week:
            Maximum/Minimum number of hours a person should work in a week. If `None`, there is no limit.
        
        default_people_number_per_turn:
            Default number of people per turn. This constraint is override for turns
            specified by `add_fix_people_turns()`
        '''
        self.open_turn = open_turn
        self.close_turn = close_turn
        
        self.work_turns = TurnList.from_start_end(
            start_turn=self.open_turn,
            end_turn=self.close_turn,
        )

        self.max_open_per_people = max_open_per_people
        self.max_close_per_people = max_close_per_people
        self.max_load_per_day = max_load_per_day
        self.max_load_per_week = max_load_per_week
        self.min_load_per_week = min_load_per_week
        self.default_people_number_per_turn = default_people_number_per_turn

        if desired_work_load is None:
            desired_work_load = {}
        self.work_load = desired_work_load
        
        if people_boost is None:
            people_boost = {}
        self.people_boost = people_boost

        self.fix_people_hours: list[FixPeopleHours] = []

        self.preference_work_load: dict = None
        self.preference = None
        self.availability = None

        self.x = None
        self.schedule = None
        self.problems = ScheduleProblems()

    def add_fix_people_turns(self, turns: TurnList, people_number: int):
        "Add turn where there should be `people_number` people at the cafe."
        if isinstance(turns, list):
            turns = TurnList(turns)

        self.fix_people_hours.append(
            FixPeopleHours(turns=turns, people_number=people_number)
        )

    def generate(self, preference_path, availability_path, preference_sheet_name=None, availability_sheet_name=None):
        '''
        Generates schedule trying to maximize peoples preference and respecting
        peoples availability. After running this method, one can save the
        schedule calling `save()`.
        '''
        logging.info("Gerando a escala..")

        preference: dict = reader.read_schedule(preference_path, preference_sheet_name)
        availability: dict = reader.read_schedule(availability_path, availability_sheet_name)

        self.preference = preference
        self.availability = availability

        work_turns = self.work_turns.turns_str

        for p, day_turn in preference.items():
            if p not in availability.keys():
                availability[p] = preference[p]
                continue

            for day, turns in day_turn.items():
                for t in turns:
                    if t not in availability[p][day]:
                        availability[p][day].add(t)

        people = list(availability.keys())

        # If a person does not have a preference, is assumed 
        # that his preference is the same as his availability.
        for p in people:
            if p not in preference:
                preference[p] = availability[p]

        # Creating decision variables x[person][day][hour]
        x = {
            p: {
                d: {h: LpVariable(f"x_{p}_{d}_{h}", cat="Binary") for h in work_turns}
                for d in week_days
            }
            for p in people
        }

        for p in people:
            if p not in preference.keys():
                logging.warning(f"A pessoa {p} não preencheu a preferência.")

        # Creating the maximization problem (prioritizing preferences)
        prob = LpProblem("Weekly_Schedule_Generation", LpMaximize)

        preference_work_load = {}
        for p in preference: 
            preference_work_load[p] = 0
            for day_turns in preference[p].values(): 
                preference_work_load[p] += len(day_turns)
        self.preference_work_load = preference_work_load
            
        # preference_work_load["Floriano"] = 1000

        # Objective function: Maximize allocations in preferred hours
        prob += lpSum(
            x[p][d][h] * (1/preference_work_load[p] * self.people_boost.get(p, 1)) for p in preference for d in week_days for h in preference[p][d] if h in x[p][d]
        ), "Maximize Preferred Hours"

        # Constraint: Each shift needs 2 or 3 people
        turns_with_fix_people = set()
        for fix_people_hours in self.fix_people_hours:
            people_number = fix_people_hours.people_number
            turns = fix_people_hours.turns.turns_str

            turns_with_fix_people.update(turns)

            for d in week_days:
                for h in work_turns:
                    if h in turns:
                        prob += lpSum(x[p][d][h] for p in people) == people_number, f"Shift_{d}_{h}_{people_number}_People"

        # Constraint: Default shift needs 2 people 
        for d in week_days:
            for h in work_turns:
                if h in turns_with_fix_people:
                    continue
                prob += lpSum(x[p][d][h] for p in people) == self.default_people_number_per_turn, f"Shift_{d}_{h}_max_default_people"

        # Constraint: Ensure that each person is only scheduled when available
        for p in people:
            for d in week_days:
                for h in work_turns:
                    if h not in availability[p][d]: 
                        prob += x[p][d][h] == 0, f"Unavailability_{p}_{d}_{h}"

        # for p in people:
        #     prob += lpSum(x[p][d][h] for d in week_days for h in work_turns) >= 2 * 2, f"{p}_minimal_work"

        # Constraint: Ensure that each person works close to the desired weekly workload
        delta_h = 1
        for p in people:
            if p not in self.work_load:
                continue
            
            prob += (
                lpSum(x[p][d][h] for d in week_days for h in work_turns) >= self.work_load[p] - 2*delta_h,
                f"Load_Min_{p}"
            )
            prob += (
                lpSum(x[p][d][h] for d in week_days for h in work_turns) <= self.work_load[p] + 2*delta_h,
                f"Load_Max_{p}"
            )


        # Constraint: Distribute who opens and closes the day evenly throughout the week
        for p in people:
            prob += lpSum(
                x[p][d][self.open_turn] for d in week_days if self.open_turn in availability[p].get(d, set())
            ) <= self.max_open_per_people, f"Balanced_Weekly_Opening_{p}"
            
            prob += lpSum(
                x[p][d][self.close_turn] for d in week_days if self.close_turn in availability[p].get(d, set())
            ) <= self.max_close_per_people, f"Balanced_Weekly_Closing_{p}"


        # Constraint: Ensure everyone doesn't exceed the maximum work load per day
        if self.max_load_per_day is not None:
            for p in people:
                for d in week_days:
                    prob += lpSum(x[p][d][h] for h in work_turns) <= self.max_load_per_day*2, f"Max_Load_{p}_{d}"
        
        # Constraint: Ensure everyone doesn't exceed the maximum/minimum work load per week
        if self.min_load_per_week is not None:
            for p in people:
                prob += lpSum(x[p][d][h] for d in week_days for h in work_turns) >= self.min_load_per_week*2, f"Min_Load_Week_{p}"
        
        if self.max_load_per_week is not None:
            for p in people:
                prob += lpSum(x[p][d][h] for d in week_days for h in work_turns) <= self.max_load_per_week*2, f"Max_Load_Week_{p}"

        prob.solve(PULP_CBC_CMD(msg=False))

        logging.info("Escala gerada!")
        logging.info(f"Status da solução: {LpStatus[prob.status]}\n")

        # Generate schedule
        schedule = {d: {h: [] for h in work_turns} for d in week_days}
        for p in people:
            for d in week_days:
                for h in work_turns:
                    if x[p][d][h].value() == 1:
                        schedule[d][h].append(p)
        
        self.problems.availability = self.check_availability(x, availability)

        self.schedule = schedule
        self.x = x

    def check_availability(self, variables: dict, availability: dict):
        "Returns a list of where availability was violated."
        problems = {}
        for people, days in variables.items():
            days: dict
            for day, turns in days.items():
                turns: dict
                for turn, var in turns.items() :
                    was_summoned = var.value() == 1
                    is_available = turn in availability[people].get(day, set()) 
                    if was_summoned and not is_available:
                        if people not in problems:
                            problems[people] = []

                        problems[people].append((day, turn))
        return problems                    

    def save_sheet_turns(self, sheet, path):
        data = np.full((len(self.work_turns), len(week_days)), "", dtype=object)

        for person, week_turns in sheet.items():
            for day, day_turns in week_turns.items():
                for turn in day_turns:
                    col_id = week_days.index(day)
                    row_id = self.work_turns.turns_str.index(turn)
                    data[row_id, col_id] += f"{person}, " 

        df = pd.DataFrame(data, columns=week_days, index=self.work_turns.turns_str)
        df.to_csv(path)


    def show(self):
        "Show schedule generated."
        for d in week_days:
            print(f"\nEscala para {d.capitalize()}:")
            for h in self.work_turns.turns_str:
                print(f"{h}: {', '.join(self.schedule[d].get(h, []))}")

    def save(self, path, save_pref_avail=True):
        "Save schedule generated at `path` as a .csv"
        data = []
        for h in self.work_turns.turns_str:
            row = h.split("-")
            for d in week_days:
                row.append(", ".join(self.schedule[d][h]))
            data.append(row)

        df = pd.DataFrame(data, columns=["Início", "Fim"] + week_days)
        df.to_csv(path, index=False) 

        if save_pref_avail:
            self.save_sheet_turns(self.preference, Path(path).parent / "preferencia_cafe.csv")
            self.save_sheet_turns(self.availability, Path(path).parent / "disponibilidade_cafe.csv")

        import os
        absolute_path = os.path.abspath(path)
        logging.info(f"Escala salva em: {absolute_path}")

    def calc_work_load(self, path):
        "Calculates total work load per person and save it at `path` as a .csv"
        total_work = []
        for p in self.x.keys():
            p_work = 0
            for d in week_days:
                for h in self.work_turns.turns_str:
                    p_work += self.x[p][d][h].value()

            if p not in self.preference_work_load:
                w_rel = None
            else:
                w_rel = p_work / self.preference_work_load[p]
            
            total_work.append([p, p_work/2, self.preference_work_load[p]/2, w_rel])
        
        df = pd.DataFrame(total_work, columns=["Pessoa", "Carga Ganha (Horas)", "Carga Solicitada (Horas)", "Carga Relativa"])
        df.to_csv(path, index=False)
