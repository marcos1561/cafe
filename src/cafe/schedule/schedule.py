from pulp import LpProblem, LpVariable, lpSum, LpMaximize, LpStatus
import pandas as pd
from collections import namedtuple

from . import reader
from .turns import TurnList
from .constants import *

FixPeopleHours = namedtuple('FixPeopleHours', ['turns', 'people_number'])

class ScheduleProblems:
    def __init__(self):
        self.availability: dict = None

class Scheduler:
    def __init__(self, open_turn: str, close_turn: str, desired_work_load: dict=None,
        max_open_per_people=2, max_close_per_people=2, max_load_per_day=None,
        default_people_number_per_turn=2,         
    ):
        '''
        System to generate a schedule.

        OBS: 
            A turn is a string with the following structure: "HH:MM-HH:MM".
            Example: "07:30-08:00" 

        Parameters
        ----------
        open_turn:
            When the cafe opens
        
        close_turn:
            When the cafe closes

        desired_work_load:
            Map between people and its desired total work load in the week in hours.
        
        max_open_per_people:
            Maximum number of times a given person open the cafe
        
        max_close_per_people:
            Maximum number of times a given person close the cafe
        
        default_people_number_per_turn:
            Default number of people per turn. This constraint is override for turns
            specified by `add_fix_people_turns()`

        max_load_per_day:
            Maximum number of hours a given person works per day. If None there is no limit.
        '''
        self.open_turn = open_turn
        self.close_turn = close_turn
        
        self.work_turns = TurnList.from_start_end(
            start_turn=self.open_turn,
            end_turn=self.close_turn,
        )

        self.max_open_per_people = max_open_per_people
        self.max_close_per_people = max_close_per_people
        self.default_people_number_per_turn = default_people_number_per_turn

        if desired_work_load is None:
            desired_work_load = {}
        self.work_load = desired_work_load

        self.max_load_per_day = max_load_per_day

        self.fix_people_hours: list[FixPeopleHours] = []

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

    def generate(self, preference_path, availability_path, sheet_name):
        '''
        Generates schedule trying to maximize peoples preference and respecting
        peoples availability. After running this method, one can save the
        schedule calling `save()`.
        '''
        preference: dict = reader.read_schedule(preference_path, sheet_name)
        availability: dict = reader.read_schedule(availability_path, sheet_name)

        work_turns = self.work_turns.turns_str

        for p in preference.keys():
            if p not in availability.keys():
                availability[p] = preference[p]

        peoples = list(availability.keys())

        # Creating decision variables x[person][day][hour]
        x = {
            p: {
                d: {h: LpVariable(f"x_{p}_{d}_{h}", cat="Binary") for h in work_turns}
                for d in week_days
            }
            for p in peoples
        }

        # Creating the maximization problem (prioritizing preferences)
        prob = LpProblem("Weekly_Schedule_Generation", LpMaximize)

        # Objective function: Maximize allocations in preferred hours
        prob += lpSum(
            x[p][d][h] for p in preference for d in week_days for h in preference[p][d] if h in x[p][d]
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
                        prob += lpSum(x[p][d][h] for p in peoples) == people_number, f"Shift_{d}_{h}_{people_number}_People"

        # Constraint: Default shift needs 2 people 
        for d in week_days:
            for h in work_turns:
                if h in turns_with_fix_people:
                    continue
                prob += lpSum(x[p][d][h] for p in peoples) == self.default_people_number_per_turn, f"Shift_{d}_{h}_max_default_people"

        # Constraint: Ensure that each person is only scheduled when available
        for p in peoples:
            for d in week_days:
                for h in work_turns:
                    if h not in availability[p][d]: 
                        prob += x[p][d][h] == 0, f"Unavailability_{p}_{d}_{h}"


        # Constraint: Ensure that each person works close to the desired weekly workload
        delta_h = 1
        for p in peoples:
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
        for p in peoples:
            prob += lpSum(
                x[p][d][self.open_turn] for d in week_days if self.open_turn in availability[p].get(d, set())
            ) <= self.max_open_per_people, f"Balanced_Weekly_Opening_{p}"
            
            prob += lpSum(
                x[p][d][self.close_turn] for d in week_days if self.close_turn in availability[p].get(d, set())
            ) <= self.max_close_per_people, f"Balanced_Weekly_Closing_{p}"


        # Constraint: Ensure everyone doesn't exceed the maximum work load per day
        if self.max_load_per_day is not None:
            for p in peoples:
                for d in week_days:
                    prob += lpSum(x[p][d][h] for h in work_turns) <= self.max_load_per_day*2, f"Max_Load_{p}_{d}"

        prob.solve()

        # Display solution status
        print(f"Status da solução: {LpStatus[prob.status]}\n")

        # Generate schedule
        schedule = {d: {h: [] for h in work_turns} for d in week_days}
        for p in peoples:
            for d in week_days:
                for h in work_turns:
                    if x[p][d][h].value() == 1:
                        schedule[d][h].append(p)
        
        self.problems.availability = self.check_availability(x, availability)

        self.schedule = schedule
        self.x = x

    def check_availability(self, variables: dict, availability: dict):
        "Returns a list of where availability was violated"
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

    def show(self):
        "Show schedule generated."
        for d in week_days:
            print(f"\nEscala para {d.capitalize()}:")
            for h in self.work_turns.turns_str:
                print(f"{h}: {', '.join(self.schedule[d].get(h, []))}")

    def save(self, path):
        "Save schedule generated at `path` as a .csv"
        data = []
        for h in self.work_turns.turns_str:
            row = h.split("-")
            for d in week_days:
                row.append(", ".join(self.schedule[d][h]))
            data.append(row)

        df = pd.DataFrame(data, columns=["Início", "Fim"] + week_days)
        df.to_csv(path, index=False) 

    def calc_work_load(self, path):
        "Calculates total work load per person and save it at `path` as a .csv"
        total_work = []
        for p in self.x.keys():
            p_work = 0
            for d in week_days:
                for h in self.work_turns.turns_str:
                    p_work += self.x[p][d][h].value()
            total_work.append([p, p_work / 2])
        
        df = pd.DataFrame(total_work, columns=["Pessoa", "Carga Horária (Horas)"])
        df.to_csv(path, index=False)
